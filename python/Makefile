.PHONY: test-core test-polars test-numpy test-pandas

test-core:
	uv pip install -e ".[test-core]"
	uv run pytest -s -m core --disable-warnings 

test-polars:
	uv pip install -e ".[polars, test-core]"
	uv run pytest -s -vv -m polars

test-pandas:
	uv pip install -e ".[pandas, test-core]"
	uv run pytest -s -vv -m pandas

test-numpy:
	uv pip install -e ".[numpy, test-core]"
	uv run pytest -s -vv -m numpy

test-all: test-core test-polars test-numpy test-pandas

format:
	uv pip install ruff
	uv run ruff format .

format-diff:
	uv pip install ruff
	uv run ruff format . --diff