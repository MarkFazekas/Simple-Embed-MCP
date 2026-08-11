[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![gitlint](https://img.shields.io/badge/commit_lint-gitlint-blue)](https://github.com/jorisroovers/gitlint)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)


# Setup

## Pyenv

How to install:   
https://github.com/pyenv/pyenv/wiki#suggested-build-environment

```shell
curl https://pyenv.run | bash
```

How to update:  
```shell
cd ~/.pyenv/plugins/python-build/../.. && git pull && cd -
pyenv install --list
```

Install:

```shell
env PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install 3.12.3
pyenv virtualenv 3.12.3 sem3123
pyenv local sem3123
```

## Python

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

# Run

## Embedding

```shell
ollama pull qwen3-embedding:4b
```

## FastMCP

```shell
fastmcp dev inspector -m app.main --no-reload
```
