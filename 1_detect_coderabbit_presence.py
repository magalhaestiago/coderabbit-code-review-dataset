"""
Check which repos in repos.csv contain CodeRabbit configuration files or directories:
  .coderabbit.yml
  .coderabbit.yaml
  coderabbit.yml
  coderabbit.yaml
  .coderabbit/

Additional heuristics:
  - branches with prefix "coderabbit/" or "coderabbitai/"  -> heuristic = "Branch"
  - commits with git author name containing "coderabbit" -> heuristic = "Author"

Outputs repositories.parquet with repo metadata (language, license, stars, forks, etc.).
Uses the GitHub REST API. Set GITHUB_TOKEN_1 / _2 / _3 env variables to avoid rate limiting.
"""

import csv
import datetime
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
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

INPUT_CSV = "ai_config/repos.csv"
OUTPUT_PARQUET = "results/repositories.parquet"
PROGRESS_JSONL = "results/progress.jsonl"
MAX_WORKERS = 5 

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



def _check_coderabbit_branches(owner_repo: str) -> bool:
    """Return True if any 'coderabbit/' or 'coderabbitai/' branch exists."""
    for prefix in ("coderabbit", "coderabbitai"):
        url = f"https://api.github.com/repos/{owner_repo}/git/matching-refs/heads/{prefix}/"
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
                break
            if response.status_code == 200:
                if len(response.json()) > 0:
                    return True
                break  # no matches for this prefix, try next
            elif response.status_code == 404:
                break  # try next prefix
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
                break  # try next prefix
    return False


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


def _get_repo_metadata(owner_repo: str) -> dict:
    """Fetch repo-level metadata (language, license, dates, counts, topics) from the Repos API."""
    url = f"https://api.github.com/repos/{owner_repo}"
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
            print(f"  Request error for {owner_repo} (metadata): {e}")
            return {}
        if response.status_code == 200:
            data = response.json()
            lic = data.get("license") or {}
            return {
                "language": data.get("language"),
                "license": lic.get("spdx_id") or lic.get("name"),
                "created_at": data.get("created_at"),
                "forks": data.get("forks_count"),
                "watchers": data.get("watchers_count"),
                "stargazers": data.get("stargazers_count"),
                "topics": ",".join(data.get("topics") or []),
            }
        elif response.status_code == 404:
            return {}
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
            print(f"  Unexpected status {response.status_code} for {owner_repo} (metadata)")
            return {}
    return {}


def _get_pagination_count(owner_repo: str, endpoint: str, extra_params: dict | None = None) -> int | None:
    """Return total item count by fetching per_page=1 and reading the 'last' page number from the Link header."""
    url = f"https://api.github.com/repos/{owner_repo}/{endpoint}"
    params = {"per_page": 1}
    if extra_params:
        params.update(extra_params)
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
            print(f"  Request error for {owner_repo} ({endpoint}): {e}")
            return None
        if response.status_code == 200:
            last_url = _parse_last_link(response.headers.get("Link", ""))
            if last_url:
                m = re.search(r"[?&]page=(\d+)", last_url)
                return int(m.group(1)) if m else None
            return len(response.json())
        elif response.status_code in (404, 409):
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
            print(f"  Unexpected status {response.status_code} for {owner_repo} ({endpoint})")
            return None
    return None


def check_repo(repo_id: int, repo: str) -> dict:
    mined_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root_names = _get_root_tree(repo)
    has_config = any(target in root_names for target in TARGETS) if root_names is not None else False
    if has_config:
        heuristic = "Configuration File"
    else:
        if _check_coderabbit_branches(repo):
            heuristic = "Branch"
        elif _check_coderabbit_author(repo):
            heuristic = "Author"
        else:
            heuristic = None
    meta = _get_repo_metadata(repo)
    commits = _get_pagination_count(repo, "commits")
    contributors = _get_pagination_count(repo, "contributors", {"anon": "1"})
    return {
        "repo_id": repo_id,
        "repo_name": repo,
        "heuristic": heuristic,
        "mined_at": mined_at,
        "github_link": f"https://github.com/{repo}",
        "language": meta.get("language"),
        "license": meta.get("license"),
        "created_at": meta.get("created_at"),
        "commits": commits,
        "forks": meta.get("forks"),
        "watchers": meta.get("watchers"),
        "stargazers": meta.get("stargazers"),
        "contributors": contributors,
        "topics": meta.get("topics"),
    }


def main():
    start_time = time.time()

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        repos = [
            (repo_id + 1, row["repo_name"])
            for repo_id, row in enumerate(reader)
            if row.get("engineered_project", "").strip().lower() == "true"
        ]

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
    remaining = [(rid, r) for rid, r in repos if r not in done]

    lock = threading.Lock()
    completed_count = 0

    progress_file = open(PROGRESS_JSONL, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(check_repo, rid, repo): repo for rid, repo in remaining}
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
                    found_label = f" [{row['heuristic']}]" if row["heuristic"] else ""
                    avg = elapsed / completed_count
                    eta = avg * (total - i)
                    print(f"[{i}/{total}] {repo}{found_label}")
                    print(f"  Elapsed: {elapsed:.1f}s | Avg: {avg:.1f}s/repo | ETA: {eta:.0f}s")
    finally:
        progress_file.close()

    # Write parquet in original repo order, keeping only repos with at least one positive signal
    repo_order = {r: idx for idx, (_, r) in enumerate(repos)}
    results.sort(key=lambda r: repo_order.get(r["repo_name"], 0))

    positive = [r for r in results if r.get("heuristic")]

    columns = [
        "repo_id", "repo_name", "heuristic", "mined_at", "github_link",
        "language", "license", "created_at", "commits",
        "forks", "watchers", "stargazers", "contributors", "topics",
    ]
    df = pd.DataFrame(positive, columns=columns)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    total_elapsed = time.time() - start_time
    print(f"\nDone. {len(positive)}/{total} repos have at least one CodeRabbit signal.")
    print(f"Results saved to {OUTPUT_PARQUET} ({len(positive)} rows)")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

