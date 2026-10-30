# Import the extension: DLL search path

- Status: todo
- Depends: S2-01
- Risk: MEDIUM. Python 3.8+ on Windows does not use PATH to resolve dependent DLLs of an extension module; ROCm DLLs may need `os.add_dll_directory`. Exact failing DLL unknown [?].

Sprint: [Sprint 2: the full extension builds and imports](README.md) · ID S2-02 · priority order 2 in the sprint

**Goal.** `import exllamav3_ext` and `import exllamav3` succeed in the venv without hand-set environment.

**Steps.**
1. `kyojin_win.py check` (extension probe) and read the traceback.
2. If a DLL is not found, identify it from the log; decide where the directory is added (package `__init__`, guarded for Windows).
3. Verify the same works from a plain `.venv-win\Scripts\python`, not only through the script.

**Done when.**
- `extension` probe PASS from a plain shell.

**If it fails.** If the failure is a DLL version mismatch (`0xC0000139`), rebuild with `build --clean` and compare `pip-freeze.txt`.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [extension probe](../../../../scripts/windows/kyojin_win.py) · [exllamav3 __init__](../../../../exllamav3/__init__.py) · next: [S2-03](03-linux-no-regression.md)

Depends on: [S2-01](01-full-build-triage.md)
