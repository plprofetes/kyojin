# Measure GPU-addressable memory on Windows

- Status: todo
- Depends: S1-01
- Risk: MEDIUM probability, HIGH impact on product fit, very cheap to learn. Windows WDDM exposes unified memory differently from Linux GTT; the answer sets the model-size ceiling.

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-04 · priority order 4 in the sprint

**Goal.** Know how many GiB torch can actually allocate on the GPU, with the current BIOS/Adrenalin graphics-memory setting.

**Steps.**
1. Read `torch_gpu` details: total_gb, mem_free_gb.
2. Simplest: a few lines the user pastes into `.venv-win\Scripts\python`; adding a probe to `kyojin_win.py` needs consent. Allocation sweep (allocate 1 GiB tensors until failure, cap at physical RAM, free them); log the maximum.
3. Record the BIOS/Adrenalin 'Variable Graphics Memory' setting and total RAM; repeat after changing the setting if it is adjustable.
4. Update [assessment §4.3](../../assessment.md) and the journal with the numbers.

**Done when.**
- A number (GiB) for max single allocation and max total, with the settings used.

**If it fails.** If the ceiling is small (for example under 32 GiB), only small models are viable on Windows; record it as the Sprint 6 constraint, the POC continues.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [torch probe](../../../../scripts/windows/kyojin_win.py) · [expert tiering code that assumes Linux memory](../../../../exllamav3/util/memory.py) · next: [S1-05](05-model-pack-on-disk.md)

Depends on: [S1-01](01-preflight-green.md)
