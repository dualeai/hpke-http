PYTHON_DIR := python
TYPESCRIPT_DIR := typescript
ARTIFACT_DIR ?= artifacts
PYTHON_VERSION ?= 3.10
NPM_VERSION ?= 11.19.1
WASM_BINDGEN_VERSION ?= 0.2.128
MATURIN_VERSION ?= v1.15.0
CODSPEED_VERSION ?= 5.0.2
EXPECTED_VERSION ?=
CARGO_PACKAGE_FLAGS ?=
PYTHON_BUILD_FLAGS ?=
PYTHON_WHEEL_INTERPRETER ?=
PYTHON_WHEEL_OUT ?= $(ARTIFACT_DIR)/python
PYTHON_WHEEL_INPUT ?= $(ARTIFACT_DIR)/python/*.whl
PYTHON_WHEEL_PIP_FLAGS ?=
PYTHON_SMOKE_DIR ?= .artifact-smoke/python
NPM_SMOKE_DIR ?= .artifact-smoke/npm
PLATFORM_WHEEL_DIR ?= dist
PLATFORM_PYTHON ?= python
# GitHub Actions sets CI=true. Local Python commands may update uv.lock.
UV_RUN_FLAGS ?= $(if $(filter true 1,$(CI)),--frozen --all-extras,--all-extras)

PYTHON_WHEEL_ARGS = --release --locked --compatibility pypi --out $(PYTHON_WHEEL_OUT) \
	$(if $(PYTHON_WHEEL_INTERPRETER),--interpreter $(PYTHON_WHEEL_INTERPRETER)) \
	--manifest-path $(PYTHON_DIR)/native/Cargo.toml

.PHONY: \
	benchmark-python benchmark-rust benchmark-typescript build \
	build-benchmark-rust build-python build-rust build-typescript \
	check-python-package check-release-candidate check-release-portable-artifacts \
	develop-python install install-ci install-ci-npm install-deps \
	install-benchmark-python install-codspeed install-deps-python \
	install-deps-python-ci install-deps-typescript \
	install-wasm-bindgen lint \
	package package-python-sdist package-python-wheel package-rust \
	package-typescript print-maturin-version print-python-wheel-args \
	smoke-python-platform-wheel smoke-python-sdist smoke-python-wheel \
	smoke-typescript test test-func test-func-ci \
	test-python test-rust test-static test-typescript upgrade version \
	version-full version-pypi

version:
	@bash ./cicd/version.sh -g . -c

version-full:
	@bash ./cicd/version.sh -g . -c -m

version-pypi:
	@bash ./cicd/version.sh -g .

install: install-deps build-python

install-deps: install-deps-python install-deps-typescript

install-deps-python:
	uv sync --project $(PYTHON_DIR) --all-extras

install-deps-python-ci:
	uv sync --project $(PYTHON_DIR) --frozen --all-extras

install-benchmark-python:
	uv sync --project $(PYTHON_DIR) --frozen --extra dev --extra httpx

install-deps-typescript:
	npm ci --prefix $(TYPESCRIPT_DIR) --ignore-scripts

install-wasm-bindgen:
	cargo install wasm-bindgen-cli --version $(WASM_BINDGEN_VERSION) --locked

install-codspeed:
	cargo install cargo-codspeed --version $(CODSPEED_VERSION) --locked

install-ci-npm:
	npm install --global "npm@$(NPM_VERSION)"

install-ci: install-wasm-bindgen install-deps-python-ci install-ci-npm install-deps-typescript

upgrade:
	uv lock --project $(PYTHON_DIR) --upgrade --refresh

build: build-rust build-python build-typescript

build-rust:
	cargo build --locked --package hpke-http --all-targets

build-python:
	uv run --project $(PYTHON_DIR) $(UV_RUN_FLAGS) maturin develop \
		$(PYTHON_BUILD_FLAGS) --locked \
		--manifest-path $(PYTHON_DIR)/native/Cargo.toml

develop-python: build-python

build-typescript:
	npm --prefix $(TYPESCRIPT_DIR) run build

build-benchmark-rust: install-codspeed
	cargo codspeed build -p hpke-http -m simulation -m memory

benchmark-rust:
	cargo codspeed run -p hpke-http

benchmark-python:
	uv run --project $(PYTHON_DIR) --frozen --no-sync pytest \
		$(PYTHON_DIR)/benchmarks/ --codspeed -q -o addopts=

benchmark-typescript:
	npm --prefix $(TYPESCRIPT_DIR) run benchmark

test: test-rust test-python test-typescript

test-rust:
	cargo fmt --all -- --check
	cargo clippy --locked --workspace --all-targets -- -D warnings
	cargo test --locked --workspace --all-targets
	cargo test --locked --doc --package hpke-http
	RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --package hpke-http

test-python: test-static test-func

test-static:
	cd $(PYTHON_DIR) && uv run $(UV_RUN_FLAGS) ruff format --check .
	cd $(PYTHON_DIR) && uv run $(UV_RUN_FLAGS) ruff check .
	cd $(PYTHON_DIR) && uv run $(UV_RUN_FLAGS) pyright .
	cd $(PYTHON_DIR) && uv run $(UV_RUN_FLAGS) -m vulture .

test-func: build-python
	cd $(PYTHON_DIR) && uv run $(UV_RUN_FLAGS) pytest tests/

test-func-ci: test-func

test-typescript: build-typescript
	npm --prefix $(TYPESCRIPT_DIR) run check
	npm --prefix $(TYPESCRIPT_DIR) test

package: package-rust package-python-wheel package-python-sdist package-typescript

package-rust:
	cargo package --locked $(CARGO_PACKAGE_FLAGS) --package hpke-http

print-python-wheel-args:
	@printf '%s\n' '$(strip $(PYTHON_WHEEL_ARGS))'

print-maturin-version:
	@printf '%s\n' '$(MATURIN_VERSION)'

package-python-wheel:
	mkdir -p "$(PYTHON_WHEEL_OUT)"
	uv run --project $(PYTHON_DIR) $(UV_RUN_FLAGS) maturin build $(PYTHON_WHEEL_ARGS)

package-python-sdist:
	mkdir -p "$(ARTIFACT_DIR)/python"
	uv run --project $(PYTHON_DIR) $(UV_RUN_FLAGS) maturin sdist \
		--manifest-path $(PYTHON_DIR)/native/Cargo.toml \
		--out $(ARTIFACT_DIR)/python

check-python-package:
	uv run --project $(PYTHON_DIR) $(UV_RUN_FLAGS) twine check $(ARTIFACT_DIR)/python/*

check-release-portable-artifacts: check-python-package
	test "$$(find $(ARTIFACT_DIR)/rust -maxdepth 1 -name '*.crate' -type f | wc -l)" -eq 1
	test "$$(find $(ARTIFACT_DIR)/python -maxdepth 1 -name '*.tar.gz' -type f | wc -l)" -eq 1
	test "$$(find $(ARTIFACT_DIR)/npm -maxdepth 1 -name '*.tgz' -type f | wc -l)" -eq 1
	tar -tzf $(ARTIFACT_DIR)/npm/*.tgz > /tmp/hpke-http-npm-files.txt
	grep -Fxq 'package/_wasm/browser/hpke_http_wasm_bg.wasm' /tmp/hpke-http-npm-files.txt
	grep -Fxq 'package/_wasm/node/hpke_http_wasm_bg.wasm' /tmp/hpke-http-npm-files.txt

check-release-candidate:
	test "$$(find $(ARTIFACT_DIR)/python -maxdepth 1 -name '*.whl' -type f | wc -l)" -eq 3
	test "$$(find $(ARTIFACT_DIR)/python -maxdepth 1 -name '*.tar.gz' -type f | wc -l)" -eq 1
	ls $(ARTIFACT_DIR)/python/*manylinux*_x86_64.whl >/dev/null
	ls $(ARTIFACT_DIR)/python/*manylinux*_aarch64.whl >/dev/null
	ls $(ARTIFACT_DIR)/python/*universal2.whl >/dev/null
	test "$$(find $(ARTIFACT_DIR)/rust -maxdepth 1 -name '*.crate' -type f | wc -l)" -eq 1
	test "$$(find $(ARTIFACT_DIR)/npm -maxdepth 1 -name '*.tgz' -type f | wc -l)" -eq 1

package-typescript:
	mkdir -p "$(ARTIFACT_DIR)/npm"
	cd $(TYPESCRIPT_DIR) && npm pack --pack-destination "$(abspath $(ARTIFACT_DIR)/npm)"

smoke-python-wheel:
	uv venv $(PYTHON_SMOKE_DIR) --python "$(PYTHON_VERSION)"
	@set -- $(PYTHON_WHEEL_INPUT); test "$$#" -eq 1; \
		uv pip install --python $(PYTHON_SMOKE_DIR)/bin/python $(PYTHON_WHEEL_PIP_FLAGS) \
		"$$1[fastapi,aiohttp,httpx]"
	$(PYTHON_SMOKE_DIR)/bin/python $(PYTHON_DIR)/tests/artifact_smoke.py $(if $(EXPECTED_VERSION),"$(EXPECTED_VERSION)")

smoke-python-platform-wheel:
	$(PLATFORM_PYTHON) -c "import pathlib; wheels = list(pathlib.Path('$(PLATFORM_WHEEL_DIR)').glob('*.whl')); assert len(wheels) == 1, f'expected one wheel, got {len(wheels)}'"
	$(PLATFORM_PYTHON) -m pip install --disable-pip-version-check "$(wildcard $(PLATFORM_WHEEL_DIR)/*.whl)[fastapi,aiohttp,httpx]"
	$(PLATFORM_PYTHON) $(PYTHON_DIR)/tests/artifact_smoke.py $(if $(EXPECTED_VERSION),"$(EXPECTED_VERSION)")

smoke-python-sdist:
	uv venv --clear .artifact-smoke/python-sdist --python "$(PYTHON_VERSION)"
	uv pip install --python .artifact-smoke/python-sdist/bin/python \
		"maturin==$(patsubst v%,%,$(MATURIN_VERSION))"
	@set -- $(ARTIFACT_DIR)/python/*.tar.gz; test "$$#" -eq 1; \
		uv pip install --python .artifact-smoke/python-sdist/bin/python \
		--no-build-isolation --no-cache "$$1[fastapi,aiohttp,httpx]"
	.artifact-smoke/python-sdist/bin/python \
		$(PYTHON_DIR)/tests/artifact_smoke.py $(if $(EXPECTED_VERSION),"$(EXPECTED_VERSION)")

smoke-typescript:
	@set -- $(ARTIFACT_DIR)/npm/*.tgz; test "$$#" -eq 1
	mkdir -p $(NPM_SMOKE_DIR)
	cd $(NPM_SMOKE_DIR) && \
		npm init --yes >/dev/null && \
		npm install --ignore-scripts --no-audit --no-fund \
			"$(abspath $(ARTIFACT_DIR)/npm)"/*.tgz && \
		cp "$(abspath $(TYPESCRIPT_DIR)/test/artifact-smoke.mjs)" . && \
		$(if $(EXPECTED_VERSION),EXPECTED_VERSION="$(EXPECTED_VERSION)" )node artifact-smoke.mjs
	node $(TYPESCRIPT_DIR)/test/vite-artifact-smoke.mjs "$(abspath $(ARTIFACT_DIR)/npm)"/*.tgz

lint:
	cd $(PYTHON_DIR) && uv run ruff format .
	cd $(PYTHON_DIR) && uv run ruff check --fix .
