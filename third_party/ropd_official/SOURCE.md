# ROPD Official Source

- Repository: https://github.com/Peregrine123/ROPD_official
- Commit: `6b8d364de4e2cd02672c5d225295fda3bed82128`
- Commit date: 2026-05-08
- License: Apache-2.0

The `algo/` and `prompts/` trees are vendored without semantic changes. Project
integration code lives outside this directory so the vendored implementation
can be compared directly with the upstream commit.

The two templates are also mirrored under `algo/prompts/`. The pinned
`algo/ropd/prompts.py` resolves that path even though the upstream release
stores the canonical files in the repository-level `prompts/` directory. The
mirrored files are byte-identical to the canonical upstream templates.
