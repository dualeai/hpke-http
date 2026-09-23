# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities via [GitHub Security Advisories](https://github.com/dualeai/hpke-http/security/advisories/new).

Do not open a public issue for a suspected vulnerability. Include the affected
package and version, a minimal reproducer, and the impact when possible.

### Response timeline

- **48 hours**: Initial acknowledgment
- **7 days**: Assessment and action plan
- **90 days**: Target for fix and disclosure

### Third-party dependencies

Report a dependency vulnerability to its upstream project first. Also notify us
when it affects an `hpke-http` release or one of its default build paths.

## Supported versions

| Release line | Status |
| --- | --- |
| 2.x | Supported published line |
| 1.x and earlier | Unsupported |

The current source tree targets the next major release with protocol ID
`hpke-http/2`. It has no earlier wire decoder or binding fallback. Security
fixes ship as coordinated Rust, Python, and TypeScript releases from one
source tag.

Protocol gzip/zstd body compression is disabled by default. Enabling it can
reveal information about mixed secret and attacker-controlled body content
through ciphertext length, even though the body remains authenticated and
encrypted. Do not enable it for such bodies; the HTTP `Content-Encoding` field
is not a substitute for the protocol transform.
