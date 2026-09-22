PYTHON_DIR := python

.PHONY: \
	develop-python install install-deps lint test test-func test-func-ci \
	test-python test-rust test-static test-typescript upgrade version \
	version-full version-pypi

version:
	@bash ./cicd/version.sh -g . -c

version-full:
	@bash ./cicd/version.sh -g . -c -m

version-pypi:
	@bash ./cicd/version.sh -g .

install: install-deps develop-python

install-deps:
	uv sync --project $(PYTHON_DIR) --all-extras
	npm ci --prefix typescript --ignore-scripts

upgrade:
	uv lock --project $(PYTHON_DIR) --upgrade --refresh

develop-python:
	uv run --project $(PYTHON_DIR) maturin develop \
		--manifest-path $(PYTHON_DIR)/native/Cargo.toml

test: test-rust test-python test-typescript

test-rust:
	cargo fmt --all -- --check
	cargo clippy --locked --workspace --all-targets -- -D warnings
	cargo test --locked --workspace --all-targets
	cargo test --locked --doc --package hpke-http
	RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --package hpke-http

test-python: test-static test-func

test-static:
	cd $(PYTHON_DIR) && uv run ruff format --check .
	cd $(PYTHON_DIR) && uv run ruff check .
	cd $(PYTHON_DIR) && uv run pyright .
	cd $(PYTHON_DIR) && uv run -m vulture .

test-func:
	cd $(PYTHON_DIR) && uv run pytest tests/

test-func-ci: test-func

test-typescript:
	npm --prefix typescript run build
	npm --prefix typescript run check
	npm --prefix typescript test

lint:
	cd $(PYTHON_DIR) && uv run ruff format .
	cd $(PYTHON_DIR) && uv run ruff check --fix .
