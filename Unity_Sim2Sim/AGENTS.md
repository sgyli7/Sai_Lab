# Repository workflow

This repository uses Git as the default workflow for all ongoing work.

## Required change process

1. Before implementing a non-trivial change, create a pull request (PR) from a
   focused working branch into `main`. Keep the PR scope limited to the intended
   change and use the PR for review and tracking.
2. Every completed modification must be recorded in a Git commit. Use clear,
   focused commit messages and do not leave intended work only in the working
   tree.
3. Do not commit generated caches, local credentials, editor transient state,
   or build outputs covered by `.gitignore`.
4. Keep `main` stable. Merge changes through their PR after appropriate review
   and verification.

For tiny documentation-only or maintenance edits where creating a PR is not
practical, still make a dedicated commit and explain the exception in the
commit message or PR history.
