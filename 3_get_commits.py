"""
For each unique PR in pull_requests.parquet, fetch all commits via the
GitHub REST API:
  GET /repos/{owner}/{repo}/pulls/{pr_number}/commits

Only commits where author.login == "coderabbitai[bot]" are kept.

Outputs results/commits.parquet with commit metadata and foreign keys
linking back to results/repositories.parquet and results/pull_requests.parquet.

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
_COMMIT_ACTIVITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("autofix",                  re.compile(r'\bautofix(es)?\b|\bauto[\s_-]?fix(es)?\b', re.IGNORECASE)),
    ("generate unit tests",      re.compile(r'\bgenerate\s+unit[\s_-]?tests?\b|\bunit[\s_-]?tests?\b|\badd\s+unit\b|\badd\s+pytest\s+tests?\b|\badd\s+.*test\s+files?\b|\bupdate\s+tests?[/\w.-]*|\bUTG\b', re.IGNORECASE)),
    ("generate docstrings",      re.compile(r'\bgenerate\s+docstrings?\b|\bdocstrings?\b|\badd\s+docs?\b', re.IGNORECASE)),
    ("resolve merge conflicts",  re.compile(r'\bresolve\s+merge[\s_-]?conflicts?\b|\bmerge[\s_-]?conflicts?\b', re.IGNORECASE)),
    ("config update",            re.compile(r'\.coderabbit\.ya?ml|\bcoderabbit\s+config\b', re.IGNORECASE)),
    ("style cleanup",            re.compile(r"\breplace\s+'?var'?\s+with\s+'?const'?\b|\blint\b|\bformat(ted|ting)?\b", re.IGNORECASE)),
    ("custom recipes",           re.compile(r'\bcustom[\s_-]?recipes?\b', re.IGNORECASE)),
    ("simplify code",            re.compile(r'\bsimplify[\s_-]?code\b|\bsimplif(y|ied|ication)\b', re.IGNORECASE)),
    ("targeted code change",      re.compile(r'\bCodeRabbit\s+Chat:\s+(Update|Fix|Remove|Replace)\b|^(fix|feat|refactor|chore)(\(.+\))?:', re.IGNORECASE)),
    ("apply requested changes",   re.compile(r'\bimplement\s+requested\s+code\s+changes\b|\bcode\s+changes\s+was\s+requested\b', re.IGNORECASE)),
]

_COMMIT_INTERACTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("CodeRabbit Chat requested change", re.compile(r'\bCodeRabbit\s+Chat:', re.IGNORECASE)),
    ("autofix",                         re.compile(r'\bautofix(es)?\b|\bauto[\s_-]?fix(es)?\b', re.IGNORECASE)),
    ("generated unit tests",            re.compile(r'\bgenerate\s+unit[\s_-]?tests?\b|\bunit[\s_-]?tests?\b|\bUTG\b', re.IGNORECASE)),
    ("generated docstrings",            re.compile(r'\bgenerate\s+docstrings?\b|\bdocstrings?\b', re.IGNORECASE)),
]


def classify_commit_activity(message: str) -> str:
    """Return what kind of change the commit made."""
    if not message:
        return "Other"
    for type_name, pattern in _COMMIT_ACTIVITY_PATTERNS:
        if pattern.search(message):
            return type_name
    return "Other"


def classify_commit_interaction(message: str) -> str:
    """Return how the CodeRabbit commit appears to have been produced."""
    if not message:
        return "unknown"
    for interaction_name, pattern in _COMMIT_INTERACTION_PATTERNS:
        if pattern.search(message):
            return interaction_name
    return "unknown"


def classify_commit(message: str) -> str:
    """Return the commit activity type for backwards compatibility."""
    return classify_commit_activity(message)


def add_commit_classifications(record: dict) -> dict:
    message = record.get("message", "")
    record["commit_activity_type"] = classify_commit_activity(message)
    record["commit_interaction_type"] = classify_commit_interaction(message)
    return record


def refresh_commit_classification_columns(df):
    """Recompute classification columns for old or resumed commit datasets."""
    if df.empty or "message" not in df.columns:
        return df
    messages = df["message"].fillna("")
    df["commit_activity_type"] = messages.map(classify_commit_activity)
    df["commit_interaction_type"] = messages.map(classify_commit_interaction)
    return df


def _item_to_record(item: dict, repo_id: int, repo_name: str, pr_id: str, pr_number: int) -> dict | None:
    """Return a record only if the commit was authored by coderabbitai[bot]."""
    gh_author = item.get("author") or {}
    if gh_author.get("login") != "coderabbitai[bot]":
        return None
    commit = item.get("commit", {})
    author = commit.get("author", {})
    return add_commit_classifications({
        "repo_id": repo_id,
        "repo": repo_name,
        "pr_id": pr_id,
        "pr_number": pr_number,
        "sha": item.get("sha"),
        "message": commit.get("message"),
        "author_name": author.get("name"),
        "author_date": author.get("date"),
        "url": item.get("html_url"),
    })


def fetch_commits_for_pr(repo_id: int, repo_name: str, pr_id: str, pr_number: int) -> list[dict]:
    """Fetch all commits for a PR authored by coderabbitai[bot] (paginated)."""
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
            record = _item_to_record(i, repo_id, repo_name, pr_id, pr_number)
            if record:
                results.append(record)

        if len(items) < PER_PAGE:
            break

        page += 1
        time.sleep(0.3)

    return results


def load_prs(parquet_path: str) -> list[tuple[int, str, str, int]]:
    df = pd.read_parquet(parquet_path, columns=["repo_id", "repo", "pr_number", "pr_id"])
    unique = df.drop_duplicates(subset=["repo", "pr_number"])
    return list(zip(unique["repo_id"], unique["repo"], unique["pr_id"], unique["pr_number"]))


def main():
    prs = load_prs(INPUT_PARQUET)
    print(f"Loaded {len(prs)} authored PRs from {INPUT_PARQUET}")

    all_commits: list[dict] = []
    completed = 0

    # Load existing output to support resuming — track PR keys already processed
    already_done: set[tuple] = set()
    if os.path.exists(OUTPUT_PARQUET):
        existing_df = pd.read_parquet(OUTPUT_PARQUET)
        existing_df = refresh_commit_classification_columns(existing_df)
        all_commits = existing_df.to_dict("records")
        already_done = set(zip(existing_df["repo"], existing_df["pr_number"]))
        prs = [(rid, repo, pr_id, pr) for rid, repo, pr_id, pr in prs if (repo, pr) not in already_done]
        print(f"Resuming: {len(already_done)} PRs already processed, {len(prs)} remaining.")

    lock = threading.Lock()

    def process_pr(repo_id: int, repo_name: str, pr_id: str, pr_number: int):
        nonlocal completed
        print(f"Fetching commits: {repo_name}#{pr_number} (pr_id={pr_id})")
        commits = fetch_commits_for_pr(repo_id, repo_name, pr_id, pr_number)
        print(f"  -> {len(commits)} commits for {repo_name}#{pr_number}")
        with lock:
            all_commits.extend(commits)
            completed += 1
            if completed % 20 == 0:
                pd.DataFrame(all_commits).to_parquet(OUTPUT_PARQUET, index=False)
                print(f"  [checkpoint] Saved {len(all_commits)} commits so far.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_pr, rid, repo, pr_id, pr): (repo, pr)
            for rid, repo, pr_id, pr in prs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  Error processing {key}: {e}")

    final_df = pd.DataFrame(all_commits)
    final_df = refresh_commit_classification_columns(final_df)
    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nDone. {len(all_commits)} commits saved to {OUTPUT_PARQUET}")

    # --- Enrich pull_requests.parquet with commit activity types ---
    print("\nEnriching pull_requests.parquet with commit activity types...")
    pr_df = pd.read_parquet(INPUT_PARQUET)

    # Use the full commits.parquet for enrichment (covers resumed runs too)
    full_commits_df = pd.read_parquet(OUTPUT_PARQUET, columns=["pr_id", "commit_activity_type"])
    commit_types = (
        full_commits_df
        .groupby("pr_id")["commit_activity_type"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
        .rename(columns={"commit_activity_type": "commit_types"})
    )

    pr_df = pr_df.merge(commit_types, on="pr_id", how="left")

    def build_activity_list(row):
        base = row["coderabbit_activity"]
        base_list = list(base) if hasattr(base, "__iter__") and not isinstance(base, str) else [base]
        # strip any already-flattened string (re-entrancy safe)
        if len(base_list) == 1 and ":" in base_list[0]:
            base_list = [base_list[0].split(":")[0].split(",")[0].strip()]
        extra = list(row["commit_types"]) if isinstance(row["commit_types"], (list, tuple)) else []
        combined = base_list + [t for t in extra if t not in base_list]
        if combined == ["Authored"]:
            combined = ["Authored", "Other"]
        return combined

    def flatten_activity(value) -> str:
        items = list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else [str(value)]
        if not items:
            return ""
        first = items[0]
        rest = items[1:]
        if rest:
            authored_part = ", ".join(rest)
            return f"Reviewed, Authored: {authored_part}" if first == "Reviewed" else f"{first}: {authored_part}"
        return first

    pr_df["coderabbit_activity"] = pr_df.apply(build_activity_list, axis=1).apply(flatten_activity)
    pr_df = pr_df.drop(columns=["commit_types"])
    pr_df.to_parquet(INPUT_PARQUET, index=False)

    print(f"Updated {INPUT_PARQUET}")
    print(pr_df["coderabbit_activity"].value_counts().head(20))


if __name__ == "__main__":
    main()
