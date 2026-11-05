# Installer and release bundle: download, install, serve

- Status: todo
- Depends: S6-01
- Risk: MEDIUM: Visual Studio Build Tools are still needed at run time (Triton, hipcc JIT); the VS-less behaviour is untested; needs the user's consent for the installer.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-02 · priority order 2 in the sprint

**Goal.** A Windows user downloads `kyojin-windows-<ver>.zip`, runs one install command and reaches `READY` without compiling the extension.

**Steps.**
1. Ask for consent. Decide the shape: `kyojin_win.py setup --wheel <url|path>` (extends an existing subcommand) plus a thin `install.cmd`/PowerShell bootstrap that installs uv if missing.
2. Bundle: the wheel, `constraints-win.txt`, `kyojin_win.py`, the `tools/` folder (the wheel has no `serve.py`), SHA-256 of the wheel. ROCm/torch stay downloaded from AMD's index at install time (not bundled: the 2 GiB release-asset limit; redistribution terms not checked).
3. VS Build Tools: detect with `vswhere` (as Triton does); if missing, print or run the `winget` command with the C++ workload and a Windows SDK; this needs admin.
4. Test on a profile or machine where VS is not installed (Windows Sandbox has no GPU: install-only test) to learn the exact run-time requirement; record it. Also try Triton with `CC` pointing at its bundled TinyCC.
5. Optional: build a Windows Triton kernel seed (`tools/kernel_seed/build.py`) to cut the cold Triton compile (about 50 s for the GDN test).

**Done when.**
- A documented path from download to `READY`; the VS requirement measured, not assumed.

**If it fails.** If VS cannot be avoided, the installer checks for it and says what to install. If the wheel does not import without the SDK, the bundle installs the SDK too (it already does for JIT).

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [deferred.md bundle row](../../deferred.md) · [journal 2026-10-31 research](../../journal.md) · [install guide](../../install.md) · next: [S6-03](03-upstream-pr.md)

Depends on: [S6-01](01-ci-windows-rocm-wheel.md)
