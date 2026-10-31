# CI tier 3: written local GPU release gate

- Status: todo
- Depends: S5-01
- Risk: LOW: documentation of what was done by hand on 2026-10-30/31.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-07 · priority order 7 in the sprint

**Goal.** A short checklist, `doc/windows/release-gate.md`, of what to run on the Strix Halo machine before a tag or a release asset, with the expected result of each step and where to record the numbers. No automation and no self-hosted runner (GitHub advises against them on public repositories; fork pull requests could run code on the machine).

**Steps.**
1. Steps with expected results, from the journal: `setup --recreate` (PASS, pins reproduced), `build` (or the CI wheel installed), `check` PASS=4, A/B kernel parity (`EXL3_PLE_HIP/DQ_HIP/GR_HIP/GDN_FUSE/GDN_HF=0` vs defaults, `choices[0]` identical at `temperature: 0` with logprobs), one served request, streaming, 4 distinct 4K-token prompts (memory table before and after), MTP on/off identical on a 27B pack, `/health` shows the SDK `hipcc.exe`.
2. Record: date, commit, driver version, torch/ROCm/triton versions, timings, run-folder names, in a journal entry (or release notes).
3. Say what was not run (Linux build, other GPUs).

**Done when.**
- The file exists, was followed once end to end by someone other than the author, and the deviations were folded back.

**If it fails.** If a step needs hardware nobody has, mark it "not run" in the release notes instead of dropping it from the list.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [journal](../../journal.md) · [install.md](../../install.md) · [S4-05](../s4-serve-qwen/05-stability.md)

Depends on: [S5-01](../s5-harden/01-clean-machine-run.md)
