# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/dualeai/hpke-http/security/advisories/new).

Do not open a public issue for a suspected vulnerability. When possible, include:

- The affected package and version.
- A small code sample or steps that show the issue.
- The possible impact.

### Response timeline

- **48 hours**: Initial acknowledgment
- **7 days**: Assessment and action plan
- **90 days**: Target for fix and disclosure

### Third-party dependencies

Report a dependency vulnerability to that dependency's maintainers first.
Also notify us when it affects an `hpke-http` release or its default build steps.

## Supported versions

Only the latest published major version receives security fixes.

| Release line | Status |
| --- | --- |
| 4.x | Supported current line |
| 3.x and earlier | Unsupported |

Security fixes ship as coordinated Rust, Python, and TypeScript releases
from one source tag.

See [what observers can see](README.md#what-observers-can-see) and the
[protection limits](README.md#protection-limits) when assessing a report.
