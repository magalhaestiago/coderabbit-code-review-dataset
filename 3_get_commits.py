"""
For each PR in pr_links.parquet, fetch all commits via the GitHub REST API:
  GET /repos/{owner}/{repo}/pulls/{pr_number}/commits

This is scoped to PRs already collected in pr_links.parquet, avoiding the
GitHub search API's 1000-result cap entirely.

Outputs commits.parquet with commit metadata and foreign keys (repo_id, repo,
pr_number) linking back to pr_links.parquet.

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

INPUT_PARQUET = "results/pull_requests.parquet"
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
    ("autofix",     re.compile(r'\bautofix\b|\bauto[\s_-]?fix\b', re.IGNORECASE)),
    ("unit tests",  re.compile(r'\bunit[\s_-]?tests?\b|\badd\s+unit\b|\bgenerate\s+tests?\b', re.IGNORECASE)),
    ("docstring",   re.compile(r'\bdocstrings?\b|\badd\s+docs?\b|\bgenerate\s+docstrings?\b', re.IGNORECASE)),
]


def classify_commit(message: str) -> str:
    """Return the commit type for a coderabbitai commit message."""
    if not message:
        return "Other"
    for type_name, pattern in _COMMIT_TYPE_PATTERNS:
        if pattern.search(message):
            return type_name
    return "Other"


def _item_to_record(item: dict, repo_id: int, repo_name: str, pr_number: int) -> dict:
    commit = item.get("commit", {})
    author = commit.get("author", {})
    committer = commit.get("committer", {})
    return {
        "pr_id": f"{repo_id}_{pr_number}",
        "repo_id": repo_id,
        "repo": repo_name,
        "pr_number": pr_number,
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


def fetch_commits_for_pr(repo_id: int, repo_name: str, pr_number: int) -> list[dict]:
    """Fetch all commits for a single PR via the REST API (paginated)."""
    url = f"{API_BASE}/repos/{repo_name}/pulls/{pr_number}/commits"
    results = []
    page = 1

    while True:
        resp = _get(url, params={"per_page": PER_PAGE, "page": page})
        if resp is None:
            break

        items = resp.json()
        if not isinstance(items, list) or not items:
            break

        for i in items:
            record = _item_to_record(i, repo_id, repo_name, pr_number)
            if (i.get("author") or {}).get("login", "").lower() == "coderabbitai[bot]":
                results.append(record)

        if len(items) < PER_PAGE:
            break

        page += 1
        time.sleep(0.3)

    return results


def load_prs(parquet_path: str) -> list[tuple[int, str, int]]:
    df = pd.read_parquet(parquet_path, columns=["repo_id", "repo", "pr_number"])
    return list(zip(df["repo_id"], df["repo"], df["pr_number"]))


def main():
    prs = load_prs(INPUT_PARQUET)
    print(f"Loaded {len(prs)} PRs from {INPUT_PARQUET}")

    all_commits: list[dict] = []
    completed = 0

    # Load existing output to support resuming — track (repo, pr_number) pairs
    already_done: set[tuple[str, int]] = set()
    if os.path.exists(OUTPUT_PARQUET):
        existing_df = pd.read_parquet(OUTPUT_PARQUET)
        all_commits = existing_df.to_dict("records")
        already_done = set(zip(existing_df["repo"], existing_df["pr_number"]))
        prs = [(rid, repo, pr_num) for rid, repo, pr_num in prs if (repo, pr_num) not in already_done]
        print(f"Resuming: {len(already_done)} PRs already processed, {len(prs)} remaining.")

    lock = threading.Lock()

    def process_pr(repo_id: int, repo_name: str, pr_number: int):
        nonlocal completed
        print(f"Fetching commits: {repo_name}#{pr_number}")
        commits = fetch_commits_for_pr(repo_id, repo_name, pr_number)
        print(f"  -> {len(commits)} commits for {repo_name}#{pr_number}")
        with lock:
            all_commits.extend(commits)
            completed += 1
            if completed % 50 == 0:
                pd.DataFrame(all_commits).to_parquet(OUTPUT_PARQUET, index=False)
                print(f"  [checkpoint] Saved {len(all_commits)} commits so far.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_pr, rid, repo, pr_num): (repo, pr_num)
            for rid, repo, pr_num in prs
        }
        for future in as_completed(futures):
            repo, pr_num = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  Error processing {repo}#{pr_num}: {e}")

    final_df = pd.DataFrame(all_commits)
    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nDone. {len(all_commits)} total commits saved to {OUTPUT_PARQUET}")

    enrich_pull_requests(final_df, INPUT_PARQUET)


def enrich_pull_requests(commits_df: pd.DataFrame, pr_parquet_path: str) -> None:
    """
    Update pull_requests.parquet so the `activity` column becomes a list that
    combines the original activity value ("Reviewed" / "Authored") with the
    commit types found for that PR.  If coderabbit made commits on a PR where
    it only appeared as a reviewer, "Authored" is added automatically.
    """
    if not os.path.exists(pr_parquet_path):
        return
    if commits_df.empty:
        return

    # Map pr_id -> sorted list of unique commit types
    commit_types_by_pr: dict[str, list[str]] = (
        commits_df.groupby("pr_id")["coderabbit_type_of_commit"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )

    pr_df = pd.read_parquet(pr_parquet_path)

    # Compute pr_id on the fly in case the file pre-dates script 2's change
    if "pr_id" not in pr_df.columns:
        pr_df["pr_id"] = pr_df["repo_id"].astype(str) + "_" + pr_df["pr_number"].astype(str)

    def build_activity(row: pd.Series) -> list[str]:
        commit_types = commit_types_by_pr.get(row["pr_id"], [])
        activities: list[str] = [row["activity"]]
        # Coderabbit made commits but this row is "Reviewed" only
        if row["activity"] == "Reviewed" and commit_types:
            activities.append("Authored")
        return activities + commit_types

    pr_df["activity"] = pr_df.apply(build_activity, axis=1)
    pr_df.to_parquet(pr_parquet_path, index=False)
    print(f"Enriched {pr_parquet_path} with commit type info.")


if __name__ == "__main__":
    main()

