# Security policy

## Reporting

Please report suspected vulnerabilities privately through GitHub Security
Advisories on `krishgolcha/neoform`. Do not open a public issue containing API
keys, private prompts, checkpoint URLs, or exploit details.

## Credential model

NEOFORM reads `TINKER_API_KEY` in the Python backend. It never intentionally
stores that value in SQLite, sends it to the web client, or writes it to logs.
The default API binds to `127.0.0.1`; operators who expose it to a network are
responsible for adding authentication, TLS, and network policy.

Only the latest `0.1.x` release line receives security fixes during the preview.
