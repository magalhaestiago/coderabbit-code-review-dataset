"""
For each repo in repositories.parquet, search GitHub for closed issues commented on by
coderabbitai[bot] using the query: is:issue is:closed commenter:coderabbitai[bot] repo:<owner>/<repo>

To overcome GitHub's 1000-result cap per query, the search is split into
monthly time windows using the `closed:` qualifier. If a month still reports
>1000 results it is split into weekly windows recursively, ensuring complete
coverage regardless of issue volume.

Outputs issue_links.parquet with issue metadata and repo_id foreign key linking to repositories.parquet.

Set GITHUB_TOKEN_1 / _2 / _3 env variables to avoid rate limiting.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

INPUT_PARQUET = "repositories.parquet"
OUTPUT_PARQUET = "issue_links.parquet"
MAX_WORKERS = 3  # search API is stricter on rate limits
PER_PAGE = 100   # max allowed by GitHub search API

GITHUB_TOKEN_1 = os.environ.get("GITHUB_TOKEN_1", "")
GITHUB_TOKEN_2 = os.environ.get("GITHUB_TOKEN_2", "")
GITHUB_TOKEN_3 = os.environ.get("GITHUB_TOKEN_3", "")
GITHUB_TOKEN_4 = os.environ.get("GITHUB_TOKEN_4", "")

_tokens = [t for t in [GITHUB_TOKEN_1, GITHUB_TOKEN_2, GITHUB_TOKEN_3, GITHUB_TOKEN_4] if t]
if not _tokens:
    print("Warning: no GITHUB_TOKEN set — unauthenticated (10 req/min search limit).")

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
            if resp.status_code == 422:
                # Unprocessable entity — usually repo not accessible or query invalid
                return None
            print(f"  HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            print(f"  Request error: {e}. Retrying ({attempt+1}/{retries})...")
            time.sleep(5)
    return None


SEARCH_URL = "https://api.github.com/search/issues"
# GitHub caps at 1000 results per query; we split windows when approaching this
RESULT_CAP = 1000


def _fetch_window(query: str) -> list[dict]:
    """Fetch all pages for a single query string (up to RESULT_CAP)."""
    results = []
    page = 1

    while True:
        params = {"q": query, "per_page": PER_PAGE, "page": page}
        resp = _get(SEARCH_URL, params=params)
        if resp is None:
            break

        data = resp.json()
        items = data.get("items", [])
        total = data.get("total_count", 0)

        results.extend(items)

        fetched = (page - 1) * PER_PAGE + len(items)
        if fetched >= total or len(items) < PER_PAGE or fetched >= RESULT_CAP:
            break

        page += 1
        time.sleep(1)  # respect secondary rate limits

    return results


def _count_window(query: str) -> int:
    """Return total_count for a query without fetching all pages."""
    params = {"q": query, "per_page": 1, "page": 1}
    resp = _get(SEARCH_URL, params=params)
    if resp is None:
        return 0
    return resp.json().get("total_count", 0)


def _date_windows(start: date, end: date, delta_days: int) -> list[tuple[date, date]]:
    """Split [start, end] into windows of at most delta_days days."""
    windows = []
    cur = start
    while cur <= end:
        win_end = min(cur + timedelta(days=delta_days - 1), end)
        windows.append((cur, win_end))
        cur = win_end + timedelta(days=1)
    return windows


def _search_window(base_query: str, start: date, end: date, repo_name: str) -> list[dict]:
    """
    Recursively fetch issues for a date window, splitting further if the
    window would hit the 1000-result cap.
    """
    date_range = f"{start.isoformat()}..{end.isoformat()}"
    query = f"{base_query} closed:{date_range}"

    total = _count_window(query)

    if total == 0:
        return []

    if total >= RESULT_CAP:
        # Split the window in half and recurse
        mid = start + (end - start) / 2
        mid = date(mid.year, mid.month, mid.day)
        if mid == start:
            # Window is a single day; can't split further — fetch what we can
            print(f"  [{repo_name}] Warning: single-day window {start} still has {total} results (capped at {RESULT_CAP})")
        else:
            left = _search_window(base_query, start, mid, repo_name)
            right = _search_window(base_query, mid + timedelta(days=1), end, repo_name)
            return left + right

    return _fetch_window(query)


def _item_to_record(item: dict, repo_id: int, repo_name: str) -> dict:
    return {
        "repo_id": repo_id,
        "repo": repo_name,
        "issue_number": item.get("number"),
        "title": item.get("title"),
        "url": item.get("html_url"),
        "state": item.get("state"),
        "author": item.get("user", {}).get("login"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "closed_at": item.get("closed_at"),
        "labels": [lbl.get("name") for lbl in item.get("labels", [])],
        "comments": item.get("comments"),
    }


def search_issues_for_repo(repo_id: int, repo_name: str) -> list[dict]:
    """
    Search closed issues commented on by coderabbitai[bot] for a single repo.
    Uses monthly time windows with recursive bisection to stay under
    GitHub's 1000-result-per-query cap.
    """
    base_query = f"is:issue is:closed commenter:coderabbitai[bot] repo:{repo_name}"

    # First check total without date filter
    total = _count_window(base_query)
    print(f"  [{repo_name}] ~{total} total issues reported by GitHub")

    if total == 0:
        return []

    if total < RESULT_CAP:
        # Simple case: fetch all at once
        items = _fetch_window(base_query)
        return [_item_to_record(i, repo_id, repo_name) for i in items]

    # Split by month from GitHub's launch (2023-01) up to today
    start = date(2023, 1, 1)
    end = date.today()
    monthly_windows = _date_windows(start, end, delta_days=31)

    seen_ids: set[int] = set()
    results = []

    for win_start, win_end in monthly_windows:
        items = _search_window(base_query, win_start, win_end, repo_name)
        for item in items:
            issue_id = item.get("number")
            if issue_id not in seen_ids:
                seen_ids.add(issue_id)
                results.append(_item_to_record(item, repo_id, repo_name))
        time.sleep(0.5)  # be polite between windows

    return results


def load_repos(parquet_path: str) -> list[tuple[int, str]]:
    df = pd.read_parquet(parquet_path, columns=["repo_id", "repo_name"])
    return list(zip(df["repo_id"], df["repo_name"]))


def main():
    repos = load_repos(INPUT_PARQUET)
    print(f"Loaded {len(repos)} repos from {INPUT_PARQUET}")

    all_issues: list[dict] = []
    completed = 0

    # Load existing output to support resuming
    if os.path.exists(OUTPUT_PARQUET):
        existing_df = pd.read_parquet(OUTPUT_PARQUET)
        all_issues = existing_df.to_dict("records")
        already_done = set(existing_df["repo"].unique())
        repos = [(rid, r) for rid, r in repos if r not in already_done]
        print(f"Resuming: {len(already_done)} repos already processed, {len(repos)} remaining.")

    lock = threading.Lock()

    def process_repo(repo_id: int, repo_name: str):
        nonlocal completed
        print(f"Searching issues: {repo_name}")
        issues = search_issues_for_repo(repo_id, repo_name)
        print(f"  -> {len(issues)} issues found for {repo_name}")
        with lock:
            all_issues.extend(issues)
            completed += 1
            # Save incrementally every 10 repos
            if completed % 10 == 0:
                pd.DataFrame(all_issues).to_parquet(OUTPUT_PARQUET, index=False)
                print(f"  [checkpoint] Saved {len(all_issues)} issues so far.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_repo, rid, repo): repo for rid, repo in repos}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  Error processing {repo}: {e}")

    # Final save
    pd.DataFrame(all_issues).to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\nDone. {len(all_issues)} total issues saved to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
