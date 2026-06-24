# CodeRabbit Code Review Dataset

This repository collects and explores public GitHub activity related to
CodeRabbit usage. It detects repositories with CodeRabbit configuration or
activity, fetches pull requests, commits, and issues involving
`coderabbitai[bot]`, and stores the resulting dataset as Parquet files for
analysis in Python/Jupyter.

## Repository Layout

| File | Purpose |
| --- | --- |
| `main.py` | Runs the full five-step data collection pipeline. |
| `1_detect_coderabbit_presence.py` | Detects repositories with CodeRabbit config files, branches, or commit authorship signals. |
| `2_get_pull_requests.py` | Fetches merged pull requests reviewed or authored by `coderabbitai[bot]`. |
| `3_get_commits.py` | Fetches commits authored by `coderabbitai[bot]` and classifies commit activity/interactions. |
| `4_get_issues.py` | Fetches closed issues commented on by `coderabbitai[bot]`. |
| `5_get_repos_and_config_file_content.py` | Downloads CodeRabbit configuration file contents for detected repositories. |
| `exploring.ipynb` | Jupyter notebook for exploratory analysis of the generated Parquet files. |
| `results/` | Local output directory for generated `.parquet` files and downloaded config files. |

## Setup

Use Python 3.12 or newer. This project is configured to use
[`uv`](https://docs.astral.sh/uv/) for dependency management.

Install dependencies from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

To include the Jupyter dependencies, install the development dependency group:

```bash
uv sync --group dev
```

Create a `.env` file with one or more GitHub tokens to reduce rate-limit
pressure:

```bash
GITHUB_TOKEN_1=...
GITHUB_TOKEN_2=...
GITHUB_TOKEN_3=...
GITHUB_TOKEN_4=...
```

The scripts can run without tokens, but GitHub rate limits will make collection
much slower.

## Input Data

The first pipeline step expects:

```text
ai_config/repos.csv
```

The CSV should contain the target repositories to inspect. The scripts expect
repository names in GitHub `owner/repo` format.

## Running the Pipeline

Run every data collection step in order:

```bash
uv run python main.py
```

Or run an individual step:

```bash
uv run python 3_get_commits.py
```

Each script writes its output to `results/`. Some steps support resuming from
existing result files.

## Output Files

The main generated datasets are:

| Output | Description |
| --- | --- |
| `results/repositories.parquet` | Repository metadata and CodeRabbit presence heuristic. |
| `results/pull_requests.parquet` | Pull requests reviewed/authored by CodeRabbit, enriched with commit activity labels. |
| `results/commits.parquet` | Commits authored by `coderabbitai[bot]`, including activity and interaction classifications. |
| `results/issues.parquet` | Closed issues with comments by `coderabbitai[bot]`. |
| `results/coderabbit_configuration_files/` | Downloaded CodeRabbit configuration files. |

## Commit Classification

`3_get_commits.py` classifies CodeRabbit-authored commits with two separate
fields:

### `commit_activity_type`

Describes what the commit changed or generated.

Current examples include:

- `generate docstrings`
- `generate unit tests`
- `autofix`
- `resolve merge conflicts`
- `apply requested changes`
- `targeted code change`
- `style cleanup`
- `config update`
- `Other`

### `commit_interaction_type`

Describes the apparent mechanism or interaction that caused the commit.

Current examples include:

- `generated docstrings`
- `generated unit tests`
- `autofix`
- `CodeRabbit Chat requested change`
- `unknown`

The classifications are regex-based heuristics over commit messages. They are
intended for exploratory analysis, so new repositories may reveal additional
message patterns that should be reviewed and added over time.

## Exploring the Dataset

Open `exploring.ipynb` in Jupyter and run the cells after generating or loading
the `results/*.parquet` files.

Start Jupyter through the project environment:

```bash
uv run jupyter notebook
```

If the notebook UI does not automatically pick the project environment, install
an IPython kernel for it:

```bash
uv run python -m ipykernel install --user --name coderabbit-code-review-dataset --display-name "CodeRabbit Dataset"
```

Useful starting points:

```python
commits["commit_activity_type"].value_counts()
commits["commit_interaction_type"].value_counts()
pull_requests["coderabbit_activity"].value_counts().head(20)
```

If Jupyter cannot read Parquet files, make sure the notebook is using the
`uv` environment. `pyarrow` is already included in the project dependencies.

As a quick notebook-side fallback, install it in the active kernel:

```python
%pip install pyarrow
```

Then restart the kernel and rerun the notebook.

## Notes

- GitHub Search API has a 1,000-result cap per query. The PR and issue scripts
  split searches into monthly and weekly windows to avoid missing high-volume
  repositories.
- The pipeline is designed for public GitHub data and depends on repository
  accessibility at collection time.
- Generated results may be large and are treated as local analysis artifacts.
