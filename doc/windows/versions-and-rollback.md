# Versions, results per state, and rollback (Windows, Strix Halo)

Recorded 2026-11-20. Tags: **[V]** measured/read on this machine, **[?]** not recorded at the time. Related: [shared-vram-research.md](shared-vram-research.md), [amd-bug-report-draft.md](amd-bug-report-draft.md).

## Machine (unchanged across states)

| Item | Value |
|---|---|
| System | ASUS ProArt PX13 HN7306EAC, Ryzen AI Max+ 395, 127.15 GiB RAM **[V]** |
| BIOS | v3.14 (`HN7306EAC.314`, 2026-09-21) **[V]** |
| Windows | 11 Pro 26H2, build 26300.9974, `DxgKrnlVersion` 69639 (0x11007) **[V]** (same build before and after the driver update; OS build on 2026-11-18 and earlier **[?]**) |
| Python stack | venv `.venv-win`, `rocm 10.1.0a20260822`, torch `2.15.0a0+rocm10.1.0a20260822`, venv `amdhip64_7.dll` 10.0.3581.0 **[V]** |

## State table

| State | GPU memory mode | Adrenalin / driver store version | System32 `amdhip64_7.dll` | Result |
|---|---|---|---|---|
| **S0: known good for big models** (journal 2026-11-05 to 11-18: 122B 61 GiB, Flash-Next 4.05bpw 64 GiB, 19.8 to 33 tok/s) | fixed 96 GB dedicated, Windows sees 31.6 GB | Adrenalin 26.10.1 = `32.0.32015.7029` (`oem91.inf`, `u0205306.inf`, 2026-10-18) **[V]** that package; installed date vs. those runs **[?]** | 10.0.3683.0 **[V]** | models load, `mem_get_info` total 107.87 GiB **[V, journal]** |
| **S1: dynamic, before update** (2026-11-20 morning) | dynamic/shared, 102 GB shared in Task Manager | 26.10.1 = `32.0.32015.7029` **[V]** | 10.0.3683.0 **[V]** | Flash-Next load fails: loader check at 37.9 GiB (`mem_get_info` total 76.79 GiB, 2:1 accounting); with the check bypassed or with a rebuilt runtime, error 719 at 35 to 61 GiB; compute-touched memory wall at 61 GiB **[V]** |
| **S2: dynamic, after update** (2026-11-20, after reboot) | dynamic/shared | Adrenalin **26.11.2** = `32.0.32029.2010` (`oem70.inf`, `u0206168.inf`, 2026-11-18, release `26.20.29.02-261116a`) **[V]** + chipset update (below) | 10.0.3686.0 **[V]** | see "S2 results" |

Adrenalin 26.10.1 ↔ store version `32.0.32015.7029` and 26.11.2 ↔ driver `26.20.29.02` match AMD's release notes (RN-RAD-WIN-26-10-1; 26-11-2 notes list "Adds support for ROCm 10.2") **[V: search results, AMD page not fully readable]**.

## S2 results (after update), same tests as S1 **[V]**

- Real Flash-Next load (`serve.py`, stock runtime): **fails exactly as in S1**: `[loaddiag] OOM at ...layers.28 ... allocated 37.931 GiB`, then `RuntimeError: Insufficient VRAM in split for model and cache`. `mem_get_info` still total 76.79 GiB with 2:1 accounting (15.00 GiB "consumed" for 8 x 1 GiB; 7 and 11.9 GiB not returned after free).
- PyTorch sweeps on the stock runtime: untouched `torch.empty` 1 GiB chunks: reaches **101 GiB** then a clean `OutOfMemoryError`; with `fill_` (kernel) first failure after **61 to 62 GiB** (error 719); with `zero_` failure after 27 GiB in one run (unexplained, not repeated).
- Raw ctypes loop (`hipMalloc` 1 GiB + `hipMemset` + `hipDeviceSynchronize`, no torch), after the update: stock venv DLL (3581) and System32 DLL (3686): **reach 101 GiB, then `hipMalloc` error 2 (out of memory), no error 719**; the operator saw OS RAM usage climb to 120 GB while Task Manager's shared GPU memory fell to 0 during those runs (so Task Manager's shared counter does not show these allocations; they are real, RAM-backed). My rebuilt DLL (HIP 7.18.26465): still fails after 58 GiB with `PAL failed to submit CMD! result:-5` **[V]**. Not measured before the update with the stock/System32 DLLs, so whether 26.11.2 changed this is **[?]**.
- Net: dynamic mode is **not** working for loading Flash-Next on the stock wheel in S2. The new lead is that the `hipMemset` path of the stock runtime goes to 101 GiB while compute kernels (and my rebuilt runtime) stop near 60 GiB; not yet explained.

## Chipset and platform drivers read after the update **[V]**

Installed (current): AMD GPIO Controller 2.2.0.137 (2026-03-22), AMD PMF 26.10.15.0 (2026-07-24), AMD PSP 11.0 5.46.0.0, AMD SMBus 2.0.0.29 (2026-08-07). Older packages still in the driver store (rollback candidates, from `pnputil /enum-drivers`): GPIO 2.2.0.136, PMF 26.10.14.0, PSP 5.44.0.0, SMBus 2.0.0.26. Which of these changed in this update, and what the versions were before it, was not recorded **[?]**.

## Rollback

- The previous Adrenalin package is still in the driver store: published name `oem91.inf` (original `u0205306.inf`), folder `C:\Windows\System32\DriverStore\FileRepository\u0205306.inf_amd64_ed4bdb51fc506ec7`. To go back (elevated PowerShell; not run): `pnputil /add-driver C:\Windows\System32\DriverStore\FileRepository\u0205306.inf_amd64_ed4bdb51fc506ec7\u0205306.inf /install`, then reboot and check `(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000').RadeonSoftwareVersion` (the key index may differ) reads 26.10.1. Alternatively reinstall 26.10.1 from AMD's download page. Adrenalin app components (OpenCL/Vulkan/WinML software components such as `amdwin-u0205306.inf`) are separate packages and are also kept.
- Fixed 96 GB dedicated mode (the S0 configuration) is a firmware/Adrenalin setting; it needs a reboot.
- Nothing was rolled back or changed by the assistant; the venv was not modified (rebuilt runtime is a separate DLL in `C:\Users\plpro\.kyojin\hip-build\`, loaded only by probes).

## S2 follow-up: what limits kernels, and the lever (2026-11-20, after the update) **[V]**

- With the **stock runtime** and torch: allocating N GiB with `torch.empty` (untouched), then running one tiny kernel + sync: N=40 and 55 ok; **N=65 and 80: error 719**. So the wall is about total **allocated** bytes at the time a kernel is submitted (all allocations are referenced and must be resident because HIP runs with `alwaysResident_`), not about touched bytes. Raw `hipMemset` does not hit it because it does not go through that kernel-submit path on the stock runtime (not isolated further).
- **`PAL_ALWAYS_RESIDENT=0` removes the wall for torch kernels on the stock runtime**: N=80 and N=95 GiB allocated, tiny kernel and `fill_` on the last buffer both ok. (Before the update, the same flag did not help the rebuilt runtime in the raw memset loop.)
- Simple copy patterns work with the flag off: pageable/pinned H2D, pinned D2H, 8 threads x 8 streams of async H2D, and a C-level mimic of the stloader copy (`hipHostMalloc` + `hipMallocs` + non-blocking stream + `hipMemcpyAsync`) on both runtimes.
- But the **real loader does not work with the flag off**: stock runtime + `PAL_ALWAYS_RESIDENT=0` + `EXL3_AUTOSPLIT_MARGIN_MB=-90000` hangs forever inside the native `ext.stloader_read` (`exllamav3/loader/safetensors.py:843`, main thread, loading `hyperconnections`, module 2; 10 minutes without progress, `faulthandler` dump in the log); the rebuilt runtime + flag off fails at the first module with `stloader: device sync failed: unspecified launch failure`. Cause inside `stloader_hip.cpp` (8 threads, per-thread non-blocking streams, `hipEventRecord`/`hipEventSynchronize` on slots, `hipDeviceSynchronize`) not isolated.
- **Correction (later the same day): `PAL_ALWAYS_RESIDENT=0` is not a usable lever.** With it off, kernels that get their pointers inside a by-value struct (every torch elementwise kernel) page-fault on buffers HIP does not track (PAL -87, error 719); reproduced in pure C++ (`hip_fresh_alloc.cpp`, struct vs direct pointer args). The tests above that passed only touched buffers that happened to be resident. The loader failure is this fault, not a stloader problem.
- Original text, kept for the record: there is a user-space lever (`PAL_ALWAYS_RESIDENT=0`) that appears to remove the 61 GiB wall for compute, and a concrete next obstacle (the native loader engine under that flag). Next step would be to isolate the stloader hang (test events and `hipDeviceSynchronize` with the flag off, or load tensors through the mmap path instead), then retest the full Flash-Next load with the flag on top of the rebuilt runtime (correct accounting) or the stock one with the check bypassed.

## Pure C++ reproduction (no torch) **[V]**

- `scripts/windows/probes/hip_wall.cpp` (built with the SDK `hipcc`, stock runtime): `hip_wall.exe N` allocates N GiB untouched, then runs one tiny kernel. Flag on: N=40 ok, N=65 `FAIL hipMalloc small: 719`, N=80 fail. `PAL_ALWAYS_RESIDENT=0`: N=40, 65, 80 ok.
- `scripts/windows/probes/hip_stloader_mimic.cpp` (same build): mimic of the native loader engine; 200 batches x 16 runs ok, flag on and off, with and without in-stream kernels. The loader hang with the flag off is therefore **not** reproduced by the engine's HIP call pattern; direct `ext.stloader_read` and `SafetensorsCollection.get_tensor` (320 tensors) also work with the flag off.

## Pure C++ reproduction of the flag-off page fault **[V]**

`scripts/windows/probes/hip_fresh_alloc.cpp`: `hip_fresh_alloc.exe 8 10 struct` with `PAL_ALWAYS_RESIDENT=0` fails with 719 at rep 1; `... 8 10` (direct pointer args) and the flag-on runs pass. The module bisect that led here: `scripts/windows/probes` do not include the Python hooks; they log each `Module.load` with a device sync (see journal).
