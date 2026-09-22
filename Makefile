.PHONY: install-local test build

install-local:
	uv tool install --force .

test:
	uv run pytest -q

build:
	uv build
