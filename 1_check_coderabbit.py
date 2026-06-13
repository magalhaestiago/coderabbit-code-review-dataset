"""
Check which repos in repos.csv contain CodeRabbit configuration files or directories:
  .coderabbit.yml
  .coderabbit.yaml
  coderabbit.yml
  coderabbit.yaml
  .coderabbit/

Additional heuristics:
  - branches with prefix "coderabbit/"  -> has_coderabbit_branch, coderabbit_branch_created_at
  - commits with git author name containing "coderabbit" -> has_coderabbit_author

Timestamps (ISO 8601):
  - coderabbit_config_created_at: date of the oldest commit that introduced the config file
  - coderabbit_branch_created_at: commit date of the oldest coderabbit/ branch HEAD (proxy for creation)

Uses the GitHub Contents API. Set GITHUB_TOKEN env variable to avoid rate limiting.
"""

import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv

load_dotenv()

TARGETS = [
    ".coderabbit.yml",
    ".coderabbit.yaml",
    "coderabbit.yml",
    "coderabbit.yaml",
    ".coderabbit",
]

INPUT_CSV = "repos.csv"
OUTPUT_CSV = "coderabbit_results.csv"
PROGRESS_JSONL = "progress.jsonl"
MAX_WORKERS = 5  # concurrent repo checks; increase carefully to avoid rate limits

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
    """Switch to the next token; returns the new index."""
    with _token_lock:
        global _current_token_index
        if _current_token_index == exhausted_index:
            _current_token_index = (exhausted_index + 1) % len(_tokens)
            print(f"  Switching to token {_current_token_index + 1}.")
        return _current_token_index


def _get_root_tree(owner_repo: str) -> set | None:
    """Fetch the root-level file/dir names via the Git Trees API. Returns a set of names, or None on error."""
    url = f"https://api.github.com/repos/{owner_repo}/git/trees/HEAD"
    token_index = _current_token_index
    for attempt in range(6):
        token = _tokens[token_index] if _tokens else ""
        try:
            response = requests.get(url, headers=_headers_for(token), timeout=15)
        except requests.exceptions.ConnectTimeout:
            wait = 10 * (attempt + 1)
            print(f"  Connection timed out (attempt {attempt + 1}). Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Request error for {owner_repo}: {e}")
            return None
        if response.status_code == 200:
            return {item["path"] for item in response.json().get("tree", [])}
        elif response.status_code == 404:
            return set()
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
            print(f"  Unexpected status {response.status_code} for {owner_repo}")
            return None
    return None


def _parse_last_link(link_header: str) -> str | None:
    """Extract the URL marked rel='last' from a GitHub Link response header."""
    match = re.search(r'<([^>]+)>;\s*rel="last"', link_header)
    return match.group(1) if match else None


def _get_commit_date(owner_repo: str, sha: str, token_index: int | None = None) -> str | None:
    """Return the git author date (ISO 8601) for the given commit SHA, or None on error."""
    if token_index is None:
        token_index = _current_token_index
    url = f"https://api.github.com/repos/{owner_repo}/commits/{sha}"
    for attempt in range(6):
        token = _tokens[token_index] if _tokens else ""
        try:
            response = requests.get(url, headers=_headers_for(token), timeout=15)
        except requests.exceptions.ConnectTimeout:
            wait = 10 * (attempt + 1)
            print(f"  Connection timed out (attempt {attempt + 1}). Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Request error fetching commit {sha} for {owner_repo}: {e}")
            return None
        if response.status_code == 200:
            return response.json().get("commit", {}).get("author", {}).get("date")
        elif response.status_code in (404, 422):
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
            print(f"  Unexpected status {response.status_code} fetching commit {sha}")
            return None
    return None


def _check_coderabbit_branches(owner_repo: str) -> str | None:
    """Return the ISO 8601 timestamp of the oldest 'coderabbit/' branch commit, or None."""
    url = f"https://api.github.com/repos/{owner_repo}/git/matching-refs/heads/coderabbit/"
    token_index = _current_token_index
    for attempt in range(6):
        token = _tokens[token_index] if _tokens else ""
        try:
            response = requests.get(url, headers=_headers_for(token), timeout=15)
        except requests.exceptions.ConnectTimeout:
            wait = 10 * (attempt + 1)
            print(f"  Connection timed out (attempt {attempt + 1}). Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Request error for {owner_repo} (branches): {e}")
            return None
        if response.status_code == 200:
            refs = response.json()
            if not refs:
                return None
            dates = []
            for ref in refs:
                sha = ref.get("object", {}).get("sha", "")
                if sha:
                    date = _get_commit_date(owner_repo, sha, token_index)
                    if date:
                        dates.append(date)
            return min(dates) if dates else None
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
            print(f"  Unexpected status {response.status_code} for {owner_repo} (branches)")
            return None
    return None


def _get_config_created_at(owner_repo: str, file_path: str) -> str | None:
    """Return the ISO 8601 timestamp when file_path was first introduced (oldest commit)."""
    url = f"https://api.github.com/repos/{owner_repo}/commits"
    params = {"path": file_path, "per_page": 1}
    token_index = _current_token_index
    for attempt in range(6):
        token = _tokens[token_index] if _tokens else ""
        try:
            response = requests.get(url, headers=_headers_for(token), params=params, timeout=15)
        except requests.exceptions.ConnectTimeout:
            wait = 10 * (attempt + 1)
            print(f"  Connection timed out (attempt {attempt + 1}). Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Request error for {owner_repo} (config created_at): {e}")
            return None
        if response.status_code == 200:
            last_url = _parse_last_link(response.headers.get("Link", ""))
            if last_url:
                # More than one page: fetch the last page to get the oldest commit
                try:
                    last_resp = requests.get(last_url, headers=_headers_for(token), timeout=15)
                    if last_resp.status_code == 200:
                        commits = last_resp.json()
                        if commits:
                            return commits[-1].get("commit", {}).get("author", {}).get("date")
                except requests.exceptions.RequestException:
                    pass
            else:
                # Single page: oldest commit is the last item in the current response
                commits = response.json()
                if commits:
                    return commits[-1].get("commit", {}).get("author", {}).get("date")
            return None
        elif response.status_code in (404, 422):
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
            print(f"  Unexpected status {response.status_code} for {owner_repo} (config created_at)")
            return None
    return None


def _check_coderabbit_author(owner_repo: str) -> bool:
    """Return True if any recent commit has a git author name containing 'coderabbit'."""
    url = f"https://api.github.com/repos/{owner_repo}/commits"
    params = {"per_page": 100}
    token_index = _current_token_index
    for attempt in range(6):
        token = _tokens[token_index] if _tokens else ""
        try:
            response = requests.get(url, headers=_headers_for(token), params=params, timeout=15)
        except requests.exceptions.ConnectTimeout:
            wait = 10 * (attempt + 1)
            print(f"  Connection timed out (attempt {attempt + 1}). Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Request error for {owner_repo} (commits): {e}")
            return False
        if response.status_code == 200:
            commits = response.json()
            return any(
                "coderabbit" in (c.get("commit", {}).get("author", {}).get("name", "") or "").lower()
                for c in commits
            )
        elif response.status_code == 404:
            return False
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
            print(f"  Unexpected status {response.status_code} for {owner_repo} (commits)")
            return False
    return False


def check_repo(repo: str) -> dict:
    row = {"repo_name": repo}
    root_names = _get_root_tree(repo)
    found_any = False
    first_found_target = None
    for target in TARGETS:
        exists = target in root_names if root_names is not None else False
        row[target] = exists
        if exists:
            found_any = True
            if first_found_target is None:
                first_found_target = target
    row["has_coderabbit_config"] = found_any
    row["coderabbit_config_created_at"] = (
        _get_config_created_at(repo, first_found_target) if first_found_target else None
    )
    branch_created_at = _check_coderabbit_branches(repo)
    row["has_coderabbit_branch"] = branch_created_at is not None
    row["coderabbit_branch_created_at"] = branch_created_at
    row["has_coderabbit_author"] = _check_coderabbit_author(repo)
    return row


def main():
    start_time = time.time()

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        repos = [
            row["repo_name"] for row in reader
            if row.get("engineered_project", "").strip().lower() == "true"
        ]

    # Load already-processed repos from progress file
    done = {}
    if os.path.exists(PROGRESS_JSONL):
        with open(PROGRESS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done[entry["repo_name"]] = entry
        print(f"Resuming: {len(done)} repos already processed, skipping them.")

    results = list(done.values())
    total = len(repos)
    remaining = [r for r in repos if r not in done]

    lock = threading.Lock()
    completed_count = 0

    progress_file = open(PROGRESS_JSONL, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(check_repo, repo): repo for repo in remaining}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    row = future.result()
                except Exception as e:
                    print(f"  ERROR processing {repo}: {e}")
                    continue
                with lock:
                    completed_count += 1
                    i = len(done) + completed_count
                    results.append(row)
                    progress_file.write(json.dumps(row) + "\n")
                    progress_file.flush()
                    elapsed = time.time() - start_time
                    found_label = " FOUND" if row["has_coderabbit_config"] else ""
                    avg = elapsed / completed_count
                    eta = avg * (total - i)
                    print(f"[{i}/{total}] {repo}{found_label}")
                    print(f"  Elapsed: {elapsed:.1f}s | Avg: {avg:.1f}s/repo | ETA: {eta:.0f}s")
    finally:
        progress_file.close()

    # Write CSV in original repo order, keeping only repos with at least one positive signal
    repo_order = {r: idx for idx, r in enumerate(repos)}
    results.sort(key=lambda r: repo_order.get(r["repo_name"], 0))

    positive = [
        r for r in results
        if r.get("has_coderabbit_config") or r.get("has_coderabbit_branch") or r.get("has_coderabbit_author")
    ]

    fieldnames = ["repo_name"] + TARGETS + [
        "has_coderabbit_config", "coderabbit_config_created_at",
        "has_coderabbit_branch", "coderabbit_branch_created_at",
        "has_coderabbit_author",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(positive)

    found_count = sum(1 for r in results if r["has_coderabbit_config"])
    total_elapsed = time.time() - start_time
    print(f"\nDone. {len(positive)}/{total} repos have at least one CodeRabbit signal ({found_count} with config file).")
    print(f"Results saved to {OUTPUT_CSV} ({len(positive)} rows)")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

