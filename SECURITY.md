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

The current source tree implements protocol ID `hpke-http/3`. Security fixes
ship as coordinated Rust, Python, and TypeScript releases from one source tag.

The Rust engine tries zstd on clear payload parts before encryption and sends
raw parts when zstd does not save bytes. The protocol encrypts content but does
not hide payload size, record count, or timing. It adds no random padding.
Traffic size hiding is outside the protocol's scope.
