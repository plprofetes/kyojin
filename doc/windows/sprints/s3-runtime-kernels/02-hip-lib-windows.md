# util/hip_lib.py: find the HIP runtime DLL on Windows

- Status: todo
- Depends: S2-02
- Risk: MEDIUM. Hard-codes `libamdhip64.so` and reads `/proc/self/maps`; every JIT `.hsaco` kernel loads through `ctypes`.

Sprint: [Sprint 3: runtime kernels run](README.md) · ID S3-02 · priority order 2 in the sprint

**Goal.** `load_hip_runtime()` returns a working handle on Windows.

**Steps.**
0. A reviewed patch is saved: [hip_lib_windows.patch](../../review/patches/hip_lib_windows.patch) (Windows branch returning the `amdhip64*.dll` torch already loaded, 4 unit tests). Apply it with `git apply`, then verify on Windows.
1. Find the real DLL name and location from `doctor` (`rocm_sdk` bin_listing; expected `amdhip64*.dll` [?]).
2. Add a Windows branch (search torch lib dir and the SDK bin dir, skip `/proc`); keep the Linux path untouched.
3. Add a unit test with injected directory listings (the function already takes `lib_dir` and `read_maps`).

**Done when.**
- `load_hip_runtime()` works in the venv; Linux tests unchanged.

**If it fails.** If torch bundles a HIP DLL under another name, record the pattern in the journal.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [hip_lib.py](../../../../exllamav3/util/hip_lib.py) · next: [S3-03](03-jit-hip-kernels.md)

Depends on: [S2-02](../s2-extension-builds/02-import-and-dll-resolution.md)
