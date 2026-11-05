# Security

This is research and teaching code. It has no network service and handles no
credentials, and it is not meant to run against live markets.

The code most exposed to untrusted input is the parsing: the CSV readers and the
ITCH decoders in Python, Rust and C++. A crash, memory error or hang from a
crafted input file counts as a security bug. FastBook's id hash is not keyed, so
ids from an untrusted source need remapping first; that is documented behaviour
rather than a bug.

## Reporting

Please report a vulnerability privately through GitHub: the repository's
**Security** tab, then **Report a vulnerability**. Do not open a public issue for
it. Include the input or steps that trigger the problem and the commit you tested.

There is one maintainer and no response-time guarantee. Fixes go into the latest
release only.
