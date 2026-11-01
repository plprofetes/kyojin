# Upstream PR from a clean branch (PM files stay in the fork)

- Status: todo
- Depends: S5-01, S2-03
- Risk: LOW technically; the decisions are the user's: which docs are public, target repo.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-03 · priority order 3 in the sprint

**Goal.** A pull request to the upstream repository containing the Windows code, scripts, tests and user docs, built from a separate branch so the fork's `main` keeps the PM files (journal, sprints, review notes, tracker).

**Steps.**
1. Decide the public doc set (proposal: `doc/windows/install.md`, a short limits/deferred page, `assessment.md` if the sources and licence section 7c should be public) and fix cross-links so the set has no links to PM files (`tasks.py check` verifies links for the whole tree; run it on the PR branch too). Optionally move the PM files under `doc/windows/pm/` so the exclusion is one path.
2. Build the branch from the upstream default branch with only the PR paths:
   `git fetch upstream` (the upstream remote URL is the user's to confirm), `git switch -c pr/windows upstream/main`, then apply the diff without PM paths: `git diff $(git merge-base upstream/main main) main -- . ':!doc/windows/pm' ':!scripts/windows/tasks.py' ':!scripts/windows/test_tasks.py' ':!scripts/windows/CLAUDE.md' | git apply --3way --index`, commit with a normal message, push, open the PR from `pr/windows` (not from `main`).
3. PR body from the repo template (draft in the session scratchpad: What this changes / How you checked it / Notes); state what was not run (Linux build, full `tests/` on a GPU machine).
4. After upstream merges: merge `upstream/main` back into the fork's `main`; identical changes merge cleanly.

**Done when.**
- PR opened as a draft or ready; PM files unchanged in the fork.

**If it fails.** If upstream asks for a split, one PR per concern (core Windows build; serve fixes; install docs), following CONTRIBUTING.md ("one change per PR").

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [CONTRIBUTING.md](../../../../CONTRIBUTING.md) · [PR template](../../../../.github/pull_request_template.md) · [S2-03](../s2-extension-builds/03-linux-no-regression.md) · next: [S6-04](04-flash-next-85gib.md)

Depends on: [S5-01](../s5-harden/01-clean-machine-run.md), [S2-03](../s2-extension-builds/03-linux-no-regression.md)
