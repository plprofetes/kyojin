# Measure GPU-addressable memory on Windows

- Status: done
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
- 2026-10-31, server stopped, nothing else on the GPU (user), torch 2.15.0a0+rocm10.1.0a20260822, hip 7.16.26332: `mem_get_info` free/total 107.72 / 107.87 GiB. Allocating `torch.ones(2**30, uint8)` chunks (memory touched): **OOM after 75 GiB**, with the driver reporting 32.70 GiB free (`free: 35113152512, total: 115822460928` bytes in the allocator warning). After `del` and `empty_cache`, **single allocations of 24, 32, 48, 64 and 80 GiB all succeeded** (80 GiB was the largest tried) **[V: user's output]**. This first run is **superseded** by the second below: something else probably held about 25 GiB during it (the allocator warning showed 32.70 GiB free at the OOM, 107.72 free at the start) **[?]**; it was not investigated. Not recorded yet: BIOS/Adrenalin graphics-memory setting, driver version, Task Manager dedicated/shared totals, and whether the chunk size changes the ceiling.
- 2026-10-31, second sweep, idle GPU, same torch: chunks of 256 MiB reach **101.75 GiB** before OOM, 1 GiB chunks **101.0 GiB**, 4 GiB chunks **100.0 GiB**; single allocations **88 GiB ok, 96 GiB FAIL** (OutOfMemoryError) **[V: user's output]**. So the usable GPU memory is about **100 GiB of the 107.87 GiB** torch reports (about 93 %), nearly independent of chunk size, and one block can be somewhere between 88 and 95 GiB. This is the S1-04 number. Not recorded: BIOS/Adrenalin graphics-memory setting, driver version, Task Manager dedicated/shared totals (earlier, with GLM running: dedicated 95.2/95.8 GB, shared 2.4/15.8 GB).


**Links.** [existing torch probe output](../../../../scripts/windows/kyojin_win.py) · [expert tiering code that assumes Linux memory](../../../../exllamav3/util/memory.py) · next: [S1-05](05-model-pack-on-disk.md)

Depends on: [S1-01](01-preflight-green.md)

- 2026-10-30 (check report 5be57594, another LLM running on the same GPU): **[V] torch/HIP on Windows sees 107.87 GiB of GPU memory, 107.72 GiB free**, device `AMD Radeon(TM) 8060S Graphics`, `gfx1151`, 20 multiprocessors. Windows itself sees 31.6 GB of system RAM. 107.87 GiB is larger than the ~96 GB BIOS carve-out alone, so it appears to include shared system memory **[I]**. A plain `torch.randn(...)` on the device worked. Whether the free figure is trustworthy while another process uses the GPU is unknown **[?]**; the allocation sweep (kept to a single 16 GiB allocation while the other workload runs) is still to do. Python for the venv: uv-managed CPython 3.12.16.

- 2026-10-30, `check` run 20261030-131823, other LLM still holding the GPU: **[V] `torch.cuda.mem_get_info()` is not reliable on Windows**: it reported 107.55 GiB free while `torch.empty(1 GiB)` failed with `OutOfMemoryError` (512 KiB allocations worked). The user confirms there is no free VRAM (the other process holds it). So the earlier `HIPBLAS_STATUS_ALLOC_FAILED` is consistent with real VRAM exhaustion after all; my interim correction that the data contradicted that theory was premature (it trusted `mem_get_info`). The probe now measures single allocations of 1/16, 1/4, 1, 4, 16 GiB and warns when 1 GiB fails. The true limit with the GPU idle is still to measure.

- 2026-10-30, Task Manager while GLM-4.5-Air Q4 runs (user): **Dedicated GPU memory 95.2 / 95.8 GB, Shared GPU memory 2.4 / 15.8 GB.** **[V]** Reading: the BIOS carve-out is 95.8 (GiB in Task Manager units) and the other process has nearly all of it (about 0.6 left), which is exactly why a 1 GiB `hipMalloc` fails while 512 KiB works. torch's 107.87 GiB total is about dedicated + shared (95.8 + 15.8 = 111.6, not equal; difference unexplained), and its 'free' figure ignores other processes. **[I]** HIP allocations come from the dedicated pool, so the usable ceiling for this POC is about the dedicated size (about 95 GiB), not the 107.87 GiB torch reports; shared memory is not usable for `hipMalloc` here. To be confirmed with the GPU idle (alloc sweep) and by watching Dedicated/Shared in Task Manager during an allocation.
- Practical consequence: the other runner (llama.cpp-style) preallocates almost the whole dedicated pool. To work alongside it, restart it with a smaller context/KV so at least about 20 GB of dedicated memory stays free; otherwise GPU tests wait.
