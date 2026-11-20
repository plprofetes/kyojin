# AMD bug reports: split into four (not filed)

Date: 2026-11-20. One investigation, four reports, each with one question a maintainer can answer. File in this order and replace the `<LINK_n>` placeholders after each is filed. Each report is self-contained (environment, repro, expected/actual); evidence and history: [shared-vram-research.md](../shared-vram-research.md).

| # | File | Repo | One-line scope | Depends on |
|---|---|---|---|---|
| 1 | [1-residency-wall-therock.md](1-residency-wall-therock.md) | ROCm/TheRock | Windows APU: GPU work fails (719) or never completes once about 61 GiB are allocated; no limit is exposed | none |
| 2 | [2-vmm-double-charge-rocm-systems.md](2-vmm-double-charge-rocm-systems.md) | ROCm/rocm-systems | `hipMemMap` fails at about 31 GiB: VMM memory appears to be charged twice | links 1 |
| 3 | [3-always-resident-off-page-fault-rocm-systems.md](3-always-resident-off-page-fault-rocm-systems.md) | ROCm/rocm-systems | `PAL_ALWAYS_RESIDENT=0` removes the wall but kernels with pointers in struct args page-fault | links 1 |
| 4 | [4-windows-wheels-stale-therock.md](4-windows-wheels-stale-therock.md) | ROCm/TheRock | No public Windows gfx1151 wheel contains rocm-systems #10974 / #13193 (wrong `hipMemGetInfo`) | none |

Not filed on purpose (already tracked upstream or out of scope): `hipMemGetInfo` on APUs ignoring other processes (rocm-systems #16013 declares APUs out of scope; a commenter there already reported the Radeon 8060S); the 719 reports with dGPU + iGPU (#15959, TheRock #9339).

Attachments: `scripts/windows/probes/wddm_segments.cpp` (report 1) and `doc/windows/logs-shared-vram-20261120-pal-submit-trace.txt` (report 1, rename to `trace4.log`). The repro sources are inlined in the reports.

Before filing: confirm the environment block still matches (driver, runtime) and rerun the repro of the report you file.
