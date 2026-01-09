# Misc
name ?= hpke_http
python_version ?= 3.10  # Lowest compatible version (see pyproject.toml requires-python)

# Versions
version_full ?= $(shell $(MAKE) --silent version-full)
version_small ?= $(shell $(MAKE) --silent version)

version:
	@bash ./cicd/version.sh -g . -c

version-full:
	@bash ./cicd/version.sh -g . -c -m

version-pypi:
	@bash ./cicd/version.sh -g .

install:
	uv venv --python $(python_version) --allow-existing
	$(MAKE) install-deps

install-deps:
	uv sync --extra dev --extra fastapi --extra aiohttp --extra httpx --extra zstd

# Test vectors
VECTORS_DIR := tests/vectors

# RFC 9180 HPKE vectors from CFRG
CFRG_URL := https://raw.githubusercontent.com/cfrg/draft-irtf-cfrg-hpke/master/test-vectors.json
CFRG_RAW := $(VECTORS_DIR)/rfc9180_all.json
CFRG_PSK := $(VECTORS_DIR)/rfc9180_psk_x25519_chacha.json

# Wycheproof vectors for primitives
WYCHEPROOF_BASE := https://raw.githubusercontent.com/C2SP/wycheproof/master/testvectors_v1
WYCHEPROOF_X25519 := $(VECTORS_DIR)/wycheproof_x25519.json
WYCHEPROOF_CHACHA := $(VECTORS_DIR)/wycheproof_chacha20_poly1305.json
WYCHEPROOF_HKDF := $(VECTORS_DIR)/wycheproof_hkdf_sha256.json
WYCHEPROOF_HMAC := $(VECTORS_DIR)/wycheproof_hmac_sha256.json

upgrade:
	uv lock --upgrade --refresh
	$(MAKE) download-vectors

# Download all test vectors
download-vectors:
	$(MAKE) download-vectors-cfrg
	$(MAKE) download-vectors-wycheproof

# Download official CFRG HPKE vectors and extract our cipher suite
download-vectors-cfrg:
	@echo "Downloading RFC 9180 test vectors from CFRG..."
	@mkdir -p $(VECTORS_DIR)
	@curl -sL "$(CFRG_URL)" -o $(CFRG_RAW)
	@echo "Extracting PSK mode + X25519 + HKDF-SHA256 + ChaCha20-Poly1305..."
	@uv run python -c "\
import json; \
data = json.load(open('$(CFRG_RAW)')); \
filtered = [v for v in data if v['mode']==1 and v['kem_id']==32 and v['kdf_id']==1 and v['aead_id']==3]; \
json.dump(filtered, open('$(CFRG_PSK)', 'w'), indent=2); \
print(f'  Extracted {len(filtered)} HPKE vector(s)')"
	@rm $(CFRG_RAW)

# Download Wycheproof vectors for underlying primitives
download-vectors-wycheproof:
	@echo "Downloading Wycheproof vectors..."
	@mkdir -p $(VECTORS_DIR)
	@curl -sL "$(WYCHEPROOF_BASE)/x25519_test.json" -o $(WYCHEPROOF_X25519)
	@curl -sL "$(WYCHEPROOF_BASE)/chacha20_poly1305_test.json" -o $(WYCHEPROOF_CHACHA)
	@curl -sL "$(WYCHEPROOF_BASE)/hkdf_sha256_test.json" -o $(WYCHEPROOF_HKDF)
	@curl -sL "$(WYCHEPROOF_BASE)/hmac_sha256_test.json" -o $(WYCHEPROOF_HMAC)
	@uv run python -c "\
import json; \
x = json.load(open('$(WYCHEPROOF_X25519)')); \
c = json.load(open('$(WYCHEPROOF_CHACHA)')); \
h = json.load(open('$(WYCHEPROOF_HKDF)')); \
m = json.load(open('$(WYCHEPROOF_HMAC)')); \
print(f'  X25519: {sum(len(g[\"tests\"]) for g in x[\"testGroups\"])} tests'); \
print(f'  ChaCha20-Poly1305: {sum(len(g[\"tests\"]) for g in c[\"testGroups\"])} tests'); \
print(f'  HKDF-SHA256: {sum(len(g[\"tests\"]) for g in h[\"testGroups\"])} tests'); \
print(f'  HMAC-SHA256: {sum(len(g[\"tests\"]) for g in m[\"testGroups\"])} tests')"

test:
	$(MAKE) test-static
	$(MAKE) test-func

test-static:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright .
	uv run -m vulture .

# All tests together for accurate coverage measurement
test-func:
	uv run pytest tests/ -v -n auto

# CI-friendly tests (no root required, parallel execution)
test-func-ci:
	uv run pytest tests/ -v -n auto -m "not requires_root"

# Root-required tests only (tcpdump network capture, must run serial)
# Usage: sudo make test-func-root
test-func-root:
	uv run pytest tests/ -v -n 0 -m "requires_root" --no-cov

# Property-based fuzz tests (slower, more thorough)
test-fuzz:
	uv run pytest tests/ -v -m "fuzz" --hypothesis-show-statistics

lint:
	uv run ruff format .
	uv run ruff check --fix .

# Build and publish
build:
	rm -rf dist/
	uv build
	uv run twine check dist/*

publish-test:
	uv publish --publish-url https://test.pypi.org/legacy/

publish:
	uv publish
