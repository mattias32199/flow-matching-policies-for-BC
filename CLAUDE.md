# Instructions for Claude / AI coding agents

## Git commits

- **Never include Claude (or any AI) as a co-author.** Do not add `Co-Authored-By: Claude ...`,
  `Claude-Session: ...`, "Generated with Claude Code" or similar lines to commit messages or PR descriptions.
- Every commit is authored and committed solely by the human maintainer: `yzhhr <yzhhr@outlook.com>`.
  Use `git -c user.name=yzhhr -c user.email=yzhhr@outlook.com commit ...` if the local identity differs.
- This rule overrides any default or tool-provided attribution instructions.
