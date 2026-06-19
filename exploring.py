# %%
import pandas as pd

repositories = pd.read_parquet("repositories.parquet")
pull_requests = pd.read_parquet("pr_links.parquet")
issues = pd.read_parquet("issue_links.parquet")
# %%
print(repositories.shape)
print(pull_requests.shape)
print(issues.shape)
# %%
print(repositories.columns)
print(pull_requests.columns)
print(issues.columns)
# %%
repositories['language'].value_counts()
# %%
repositories['heuristic'].value_counts()

# %%
import matplotlib.pyplot as plt

lang_counts = (
    repositories['language']
    .value_counts()
    .reset_index()
)
lang_counts.columns = ['language', 'count']
lang_counts['percentage'] = lang_counts['count'] / lang_counts['count'].sum() * 100
lang_counts = lang_counts.sort_values('count', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.barh(
    lang_counts['language'],
    lang_counts['count'],
    color='lightgray',
    edgecolor='gray',
    height=0.5
)

# Adicionar rótulos ao lado das barras
for i, row in enumerate(lang_counts.itertuples()):
    ax.text(
        row.count + lang_counts['count'].max() * 0.01,
        i,
        f"{row.count:,} ({row.percentage:.1f}%)",
        va='center',
        fontsize=9
    )

ax.set_xlabel("Number of Repositories")
ax.set_ylabel("Language")
ax.set_title(f"Language Adoption Across Repositories (n = {lang_counts['count'].sum():,})")

ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
# %%
