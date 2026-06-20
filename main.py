"""
Pipeline runner — executes all 5 data-collection scripts in order.

Steps:
  1. Detect CodeRabbit presence in repos          (1_detect_coderabbit_presence.py)
  2. Fetch pull requests                           (2_get_pull_requests.py)
  3. Fetch commits authored by coderabbitai[bot]  (3_get_commits.py)
  4. Fetch issues                                  (4_get_issues.py)
  5. Download CodeRabbit config file contents      (5_get_repos_and_config_file_content.py)

Each step is imported and its main() is called directly (no subprocess overhead).
The pipeline stops immediately if any step raises an unhandled exception.
"""

import time

import importlib, sys
from pathlib import Path

# Ensure the project root is on sys.path so sibling modules resolve correctly.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = [
    ("1_detect_coderabbit_presence", "Detect CodeRabbit presence"),
    ("2_get_pull_requests",          "Fetch pull requests"),
    ("3_get_commits",                "Fetch commits"),
    ("4_get_issues",                 "Fetch issues"),
    ("5_get_repos_and_config_file_content", "Download config file contents"),
]


def run_pipeline():
    total_start = time.time()

    for i, (module_name, description) in enumerate(STEPS, start=1):
        print(f"\n{'='*60}")
        print(f"Step {i}/{len(STEPS)}: {description}")
        print(f"{'='*60}")
        step_start = time.time()

        module = importlib.import_module(module_name)
        module.main()

        elapsed = time.time() - step_start
        print(f"\nStep {i} finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete. Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_pipeline()
