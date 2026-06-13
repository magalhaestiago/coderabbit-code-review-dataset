"""
For every repo in coderabbit_results.csv that has a CodeRabbit config file,
fetch the raw content from GitHub and save it locally under:

    configs/<owner>__<repo>__<filename>

Only the first config file found per repo is downloaded (priority follows TARGETS order).
Requires GITHUB_TOKEN_1 (and optionally GITHUB_TOKEN_2) in .env.
"""

import base64
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV = "coderabbit_results.csv"
OUTPUT_DIR = Path("configs")
PROGRESS_JSONL = "progress_download.jsonl"
MAX_WORKERS = 10

TARGETS = [
    ".coderabbit.yml",
    ".coderabbit.yaml",
    "coderabbit.yml",
    "coderabbit.yaml",
    ".coderabbit",
    "coderabbit"
]

# ---------------------------------------------------------------------------
# Token rotation
# ---------------------------------------------------------------------------

GITHUB_TOKEN_1 = os.environ.get("GITHUB_TOKEN_1", "")
GITHUB_TOKEN_2 = os.environ.get("GITHUB_TOKEN_2", "")
GITHUB_TOKEN_3 = os.environ.get("GITHUB_TOKEN_3", "")

_tokens = [t for t in [GITHUB_TOKEN_1, GITHUB_TOKEN_2, GITHUB_TOKEN_3] if t]
if not _tokens:
    print("Warning: no GITHUB_TOKEN_1 / GITHUB_TOKEN_2 / GITHUB_TOKEN_3 set — unauthenticated (60 req/hr limit).")

_token_lock = threading.Lock()
_current_token_index = 0


def _headers_for(token: str) -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _rotate_token(exhausted_index: int) -> int:
    with _token_lock:
        global _current_token_index
        if _current_token_index == exhausted_index:
            _current_token_index = (exhausted_index + 1) % len(_tokens)
            print(f"  Switching to token {_current_token_index + 1}.")
        return _current_token_index


# ---------------------------------------------------------------------------
# Fetch file content from GitHub Contents API
# ---------------------------------------------------------------------------

def fetch_file_content(owner_repo: str, path: str) -> str | None:
    """Return decoded file content string, or None on error/not found."""
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    token_index = _current_token_index
    for attempt in range(6):
        token = _tokens[token_index] if _tokens else ""
        try:
            response = requests.get(url, headers=_headers_for(token), timeout=15)
        except requests.exceptions.ConnectTimeout:
            wait = 10 * (attempt + 1)
            print(f"  Timeout (attempt {attempt + 1}). Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Request error for {owner_repo}/{path}: {e}")
            return None
        if response.status_code == 200:
            data = response.json()
            # Directory entries return a list — skip
            if isinstance(data, list):
                return None
            encoded = data.get("content", "")
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
        elif response.status_code == 404:
            return None
        elif response.status_code == 403:
            remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
            if remaining == 0 and len(_tokens) > 1:
                token_index = _rotate_token(token_index)
            else:
                reset = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
        else:
            print(f"  Unexpected status {response.status_code} for {owner_repo}/{path}")
            return None
    return None


# ---------------------------------------------------------------------------
# Per-repo worker
# ---------------------------------------------------------------------------

def download_repo(repo: str, targets_found: list[str]) -> dict:
    """Download the first available config file for the repo."""
    for target in targets_found:
        content = fetch_file_content(repo, target)
        if content is None:
            continue
        safe_name = repo.replace("/", "__")
        repo_dir = OUTPUT_DIR / safe_name
        repo_dir.mkdir(parents=True, exist_ok=True)
        filename = repo_dir / target
        filename.write_text(content, encoding="utf-8")
        return {"repo_name": repo, "file": target, "status": "ok", "saved_as": str(filename)}
    return {"repo_name": repo, "file": None, "status": "not_found"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read repos that have a config and note which targets were found
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        repos = []
        for row in reader:
            if row.get("has_coderabbit_config", "").strip().lower() != "true":
                continue
            found_targets = [t for t in TARGETS if row.get(t, "").strip().lower() == "true"]
            repos.append((row["repo_name"], found_targets))

    print(f"Repos to download: {len(repos)}")

    # Load already-downloaded repos from progress file
    done = set()
    if os.path.exists(PROGRESS_JSONL):
        with open(PROGRESS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["repo_name"])
        print(f"Resuming: {len(done)} already downloaded, skipping.")

    remaining = [(r, t) for r, t in repos if r not in done]
    total = len(repos)

    lock = threading.Lock()
    completed_count = 0

    progress_file = open(PROGRESS_JSONL, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_repo, repo, targets): repo for repo, targets in remaining}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"repo_name": repo, "file": None, "status": f"error: {e}"}
                with lock:
                    completed_count += 1
                    i = len(done) + completed_count
                    progress_file.write(json.dumps(result) + "\n")
                    progress_file.flush()
                    elapsed = time.time() - start_time
                    avg = elapsed / completed_count
                    eta = avg * (total - i)
                    status = result["status"]
                    saved = f" → {result['file']}" if result.get("file") else ""
                    print(f"[{i}/{total}] {repo}{saved} [{status}]")
                    print(f"  Elapsed: {elapsed:.1f}s | Avg: {avg:.1f}s/repo | ETA: {eta:.0f}s")
    finally:
        progress_file.close()

    total_elapsed = time.time() - start_time
    print(f"\nDone. Files saved to '{OUTPUT_DIR}/'")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
