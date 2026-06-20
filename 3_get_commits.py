"""
For each repo in repositories.parquet, fetch all commits authored by
coderabbitai[bot] via the GitHub REST API:
  GET /repos/{owner}/{repo}/commits?author=coderabbitai[bot]

Outputs results/commits.parquet with commit metadata and a repo_id foreign
key linking back to results/repositories.parquet.

Set GITHUB_TOKEN_1 / _2 / _3 / _4 env variables to avoid rate limiting.
"""

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

INPUT_PARQUET = "results/repositories.parquet"
OUTPUT_PARQUET = "results/commits.parquet"
MAX_WORKERS = 5
PER_PAGE = 100

GITHUB_TOKEN_1 = os.environ.get("GITHUB_TOKEN_1", "")
GITHUB_TOKEN_2 = os.environ.get("GITHUB_TOKEN_2", "")
GITHUB_TOKEN_3 = os.environ.get("GITHUB_TOKEN_3", "")
GITHUB_TOKEN_4 = os.environ.get("GITHUB_TOKEN_4", "")
_tokens = [t for t in [GITHUB_TOKEN_1, GITHUB_TOKEN_2, GITHUB_TOKEN_3, GITHUB_TOKEN_4] if t]
if not _tokens:
    print("Warning: no GITHUB_TOKEN set — unauthenticated (60 req/hour limit).")

_token_lock = threading.Lock()
_current_token_index = 0


def _next_token() -> str:
    global _current_token_index
    with _token_lock:
        if not _tokens:
            return ""
        tok = _tokens[_current_token_index % len(_tokens)]
        _current_token_index += 1
        return tok


def _headers() -> dict:
    token = _next_token()
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(url: str, params: dict = None, retries: int = 5) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 403 or resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                print(f"  Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            if resp.status_code in (404, 422):
                return None
            print(f"  HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            print(f"  Request error: {e}. Retrying ({attempt+1}/{retries})...")
            time.sleep(5)
    return None


API_BASE = "https://api.github.com"

# Ordered: first match wins.
_COMMIT_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("autofix",                  re.compile(r'\bautofix\b|\bauto[\s_-]?fix\b', re.IGNORECASE)),
    ("generate unit tests",      re.compile(r'\bgenerate\s+unit[\s_-]?tests?\b|\bunit[\s_-]?tests?\b|\badd\s+unit\b', re.IGNORECASE)),
    ("generate docstrings",      re.compile(r'\bgenerate\s+docstrings?\b|\bdocstrings?\b|\badd\s+docs?\b', re.IGNORECASE)),
    ("resolve merge conflicts",  re.compile(r'\bresolve\s+merge[\s_-]?conflicts?\b|\bmerge[\s_-]?conflicts?\b', re.IGNORECASE)),
    ("custom recipes",           re.compile(r'\bcustom[\s_-]?recipes?\b', re.IGNORECASE)),
    ("simplify code",            re.compile(r'\bsimplify[\s_-]?code\b|\bsimplif(y|ied|ication)\b', re.IGNORECASE)),
]


def classify_commit(message: str) -> str:
    """Return the commit type for a coderabbitai commit message."""
    if not message:
        return "Other"
    for type_name, pattern in _COMMIT_TYPE_PATTERNS:
        if pattern.search(message):
            return type_name
    return "Other"


def _item_to_record(item: dict, repo_id: int, repo_name: str) -> dict:
    commit = item.get("commit", {})
    author = commit.get("author", {})
    committer = commit.get("committer", {})
    return {
        "repo_id": repo_id,
        "repo": repo_name,
        "sha": item.get("sha"),
        "message": commit.get("message"),
        "coderabbit_type_of_commit": classify_commit(commit.get("message", "")),
        "author_name": author.get("name"),
        "author_email": author.get("email"),
        "author_date": author.get("date"),
        "committer_name": committer.get("name"),
        "committer_email": committer.get("email"),
        "committer_date": committer.get("date"),
        "url": item.get("html_url"),
    }


def fetch_commits_for_repo(repo_id: int, repo_name: str) -> list[dict]:
    """Fetch all commits authored by coderabbitai[bot] for a repo (paginated)."""
    url = f"{API_BASE}/repos/{repo_name}/commits"
    results = []
    page = 1

    while True:
        resp = _get(url, params={"author": "coderabbitai[bot]", "per_page": PER_PAGE, "page": page})
        if resp is None:
            break

        items = resp.json()
        if not isinstance(items, list) or not items:
            break

        for i in items:
            results.append(_item_to_record(i, repo_id, repo_name))

        if len(items) < PER_PAGE:
            break

        page += 1
        time.sleep(0.3)

    return results


def load_repos(parquet_path: str) -> list[tuple[int, str]]:
    df = pd.read_parquet(parquet_path, columns=["repo_id", "repo_name"])
    return list(zip(df["repo_id"], df["repo_name"]))


def main():
    repos = load_repos(INPUT_PARQUET)
    print(f"Loaded {len(repos)} repos from {INPUT_PARQUET}")

    all_commits: list[dict] = []
    completed = 0

    # Load existing output to support resuming — track repo names already processed
    already_done: set[str] = set()
    if os.path.exists(OUTPUT_PARQUET):
        existing_df = pd.read_parquet(OUTPUT_PARQUET)
        all_commits = existing_df.to_dict("records")
        already_done = set(existing_df["repo"].unique())
        repos = [(rid, repo) for rid, repo in repos if repo not in already_done]
        print(f"Resuming: {len(already_done)} repos already processed, {len(repos)} remaining.")

    lock = threading.Lock()

    def process_repo(repo_id: int, repo_name: str):
        nonlocal completed
        print(f"Fetching commits: {repo_name}")
        commits = fetch_commits_for_repo(repo_id, repo_name)
        print(f"  -> {len(commits)} commits for {repo_name}")
        with lock:
            all_commits.extend(commits)
            completed += 1
            if completed % 50 == 0:
                pd.DataFrame(all_commits).to_parquet(OUTPUT_PARQUET, index=False)
                print(f"  [checkpoint] Saved {len(all_commits)} commits so far.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_repo, rid, repo): repo
            for rid, repo in repos
        }
        for future in as_completed(futures):
            repo = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  Error processing {repo}: {e}")

    final_df = pd.DataFrame(all_commits)
    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nDone. {len(all_commits)} total commits saved to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()

