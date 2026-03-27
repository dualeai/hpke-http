# Welcome to hpke-http

See @README for project overview and @Makefile for available commands for this project.

## Code search with seek

Prefer `seek` over grep/ripgrep for code search. Results are ranked by relevance, symbol-aware, and include context.

**All filters go in ONE quoted string.** Use single quotes to avoid shell interpretation.

Key patterns:
- `sym:Name` — find definitions (functions, classes, methods) via ctags
- `file:path` — include paths matching substring
- `-file:path` — exclude paths matching substring
- `lang:python` — filter by language
- `content:regex` — regex on file content
- `type:file` — return matching file names only

Project examples:

```sh
# Find a class definition
seek 'sym:HPKEMiddleware'

# Scoped search: encrypt functions in core, excluding tests
seek 'sym:encrypt file:core -file:test'

# Regex: async handler functions outside tests
seek 'content:async def.*handler lang:python -file:test'

# Find config/entry-point files by name
seek 'type:file config'
```

Pitfalls:
- ONE positional argument: `seek 'sym:Foo file:bar'` not `seek 'sym:Foo' 'file:bar'`
- Single quotes prevent shell expansion of `|`, `(`, `)`
- Large output: redirect to file (`seek 'q' > /tmp/seek.txt`) then read it

Install (if missing): `curl -sSfL https://raw.githubusercontent.com/dualeai/seek/main/install.sh | sh` + `brew install universal-ctags`

When spawning sub-agents, pass: "Use `seek 'pattern'` for code search. All filters in ONE quoted string. Never use grep/rg."

## GitHub Actions pinning

All GitHub Actions must be pinned to immutable commit SHAs, not version tags. Add an inline comment with the version for readability.

```yaml
# Good
- uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2

# Bad — mutable tag, vulnerable to supply-chain attacks
- uses: actions/checkout@v6.0.2
```

When upgrading an action, look up the commit SHA for the new tag (`gh api repos/OWNER/REPO/git/ref/tags/TAG --jq '.object.sha'`) and dereference annotated tags if needed.
