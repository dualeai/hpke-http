# Welcome to hpke-http

See @README for project overview and @Makefile for available commands for this project.

## Writing: Orwell's six rules

From “Politics and the English Language”:

- Never use a metaphor, simile, or other figure of speech which you are used to seeing in print.
- Never use a long word where a short one will do.
- If it is possible to cut a word out, always cut it out.
- Never use the passive where you can use the active.
- Never use a foreign phrase, a scientific word, or a jargon word if you can think of an everyday English equivalent.
- Break any of these rules sooner than say anything outright barbarous.

## Code search with seek

Prefer `seek` over grep/ripgrep for code search. Results ranked by BM25, symbol-aware (ctags), grouped by file with 3 lines of context. Modified files tagged `[uncommitted]`.

Usage: `seek [flags] '<query>' [path...]`

**Query filters stay in ONE quoted string.** Single quotes avoid shell expansion of `|`, `(`, `)`. Flags come BEFORE query. Tokens AFTER query are filesystem paths, not extra filters.

Key patterns:
- `sym:Name` — find definitions (functions, classes, methods) via ctags
- `file:path` — include paths matching substring
- `-file:path` — exclude paths matching substring
- `lang:python` — filter by language
- `content:regex` — regex on file CONTENT only (bare words match content + filenames)
- `type:file` — return matching file names only
- `case:yes` — force case-sensitive
- `or`, `()` — boolean logic (space = implicit AND)

Project examples (validated):

```sh
# Find ASGI middleware class definition
seek 'sym:HPKEMiddleware'

# Encrypt-related symbols in core.py, excluding tests
seek 'sym:encrypt file:core -file:test'

# KEM ABC declarations across Python files
seek 'content:class.*KEM.*ABC lang:python'

# Locate config/entry-point files
seek 'type:file pyproject'

# Restrict to single exact file
seek 'KemId' src/hpke_http/constants.py

# Search across multiple files
seek 'encrypt' src/hpke_http/core.py src/hpke_http/streaming.py

# Search across multiple folders (tests + src)
seek 'def test_' tests/ src/

# Search outside the git worktree (any folder on disk)
seek 'export' ~/.claude/

# Limit output noise
seek -n 5 -m 3 'ChunkEncryptor'
```

Pitfalls:
- ONE positional argument for query: `seek 'sym:Foo file:bar'` not `seek 'sym:Foo' 'file:bar'`
- Flags BEFORE query: `seek -n 5 'Foo' ./src`
- Tokens after query are paths, not filters
- Multi-word query is AND'd substrings, not phrase match
- Symlink path operands rejected (e.g. macOS `/tmp` → use `/private/tmp`)
- Large output: redirect to file (`seek 'q' > /tmp/seek.txt`) then read it

Install (if missing): `curl -sSfL https://raw.githubusercontent.com/dualeai/seek/main/install.sh | sh` + `brew install universal-ctags`

When spawning sub-agents, pass: "Use `seek 'pattern' [path...]` for code search. Keep query filters in ONE quoted string. Flags before query, paths after. Never use grep/rg."

## GitHub Actions pinning

All GitHub Actions must be pinned to immutable commit SHAs, not version tags. Add an inline comment with the version for readability.

```yaml
# Good
- uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2

# Bad — mutable tag, vulnerable to supply-chain attacks
- uses: actions/checkout@v6.0.2
```

When upgrading an action, look up the commit SHA for the new tag (`gh api repos/OWNER/REPO/git/ref/tags/TAG --jq '.object.sha'`) and dereference annotated tags if needed.
