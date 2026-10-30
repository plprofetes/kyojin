# Measure GPU-addressable memory on Windows

- Status: todo
- Depends: S1-01
- Risk: MEDIUM probability, HIGH impact on product fit, very cheap to learn. Windows WDDM exposes unified memory differently from Linux GTT; the answer sets the model-size ceiling.

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-04 · priority order 4 in the sprint

**Goal.** Know how many GiB torch can actually allocate on the GPU, with the current BIOS/Adrenalin graphics-memory setting.

**Steps.**
No script. Paste into a Python prompt (`.venv-win\Scripts\python`); nothing is created.

1. Visible memory and a 1 GiB allocation sweep (stops at 200 GiB or the first failure; close other programs first):
   ```python
   import torch
   f, t = torch.cuda.mem_get_info(); print("free/total GiB", f / 2**30, t / 2**30)
   xs = []
   try:
       while len(xs) < 200: xs.append(torch.empty(2**30, dtype=torch.uint8, device="cuda"))
   except Exception as e: print(type(e).__name__, e)
   print("allocated GiB", len(xs)); del xs
   ```
2. Note total RAM, the BIOS/Adrenalin "Variable Graphics Memory" setting and the driver version. If the setting can be changed, repeat after changing it.
3. Put the numbers in the journal and update [assessment §4.3](../../assessment.md).

**Done when.**
- A number (GiB) for max single allocation and max total, with the settings used.

**If it fails.** If the ceiling is small (for example under 32 GiB), only small models are viable on Windows; record it as the Sprint 6 constraint, the POC continues.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [existing torch probe output](../../../../scripts/windows/kyojin_win.py) · [expert tiering code that assumes Linux memory](../../../../exllamav3/util/memory.py) · next: [S1-05](05-model-pack-on-disk.md)

Depends on: [S1-01](01-preflight-green.md)
