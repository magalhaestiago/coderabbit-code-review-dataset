import json

INPUT_FILE = "pr_links.json"
OUTPUT_FILE = "prs_authored_by_coderabbit.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    prs = json.load(f)

coderabbit_prs = [pr for pr in prs if pr.get("author") == "coderabbitai[bot]"]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(coderabbit_prs, f, indent=2)

print(f"Found {len(coderabbit_prs)} PRs authored by coderabbitai[bot] out of {len(prs)} total.")
print(f"Saved to {OUTPUT_FILE}")
