[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![gitlint](https://img.shields.io/badge/commit_lint-gitlint-blue)](https://github.com/jorisroovers/gitlint)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)


# Setup commands

```shell
pip install uv
uv pip install -r requirements.txt
```

# Lint:

```shell
python -m ruff check --select I,TC --fix
python -m ruff format
python -m ruff check .
python -m flake8 .
python -m mypy .
```


