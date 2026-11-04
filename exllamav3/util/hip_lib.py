# Find torch's bundled HIP runtime. Stable wheels ship libamdhip64.so; nightly ROCm wheels ship only versioned
# names (libamdhip64.so.7, libamdhip64.so.7.1.2 ...). Plain name first, then the versioned ones, highest first.

import ctypes, glob, os, re, sys

_NAME = "libamdhip64.so"
_logged = set()


def _version_key(path: str):
    return tuple(int(x) for x in re.findall(r"\d+", os.path.basename(path)[len(_NAME):]))


def _loaded_hip_runtime(read_maps) -> str | None:
    """Path of the libamdhip64 this process has already mapped (torch preloads it), or None."""
    try:
        lines = read_maps().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) == 6 and os.path.basename(parts[5]).startswith(_NAME) and os.path.isabs(parts[5]):
            return parts[5]
    return None


def _read_proc_maps() -> str:
    with open("/proc/self/maps") as f:
        return f.read()


def find_hip_runtime(lib_dir: str | None = None, read_maps = _read_proc_maps) -> str | None:
    """Path of libamdhip64 inside lib_dir (default: <torch>/lib), or None when there is none.

    With the default lib_dir, a torch wheel that keeps the library elsewhere (AMD nightly: _rocm_sdk_core/lib,
    preloaded at import) falls back to the copy this process has already mapped (read_maps: /proc/self/maps text).
    """
    default_dir = lib_dir is None
    if default_dir:
        import torch
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    plain = os.path.join(lib_dir, _NAME)
    if os.path.isfile(plain):
        return plain
    versioned = [p for p in glob.glob(os.path.join(lib_dir, _NAME + ".*")) if os.path.isfile(p)
                 and re.fullmatch(r"(\.\d+)+", os.path.basename(p)[len(_NAME):])]
    if not versioned:
        loaded = _loaded_hip_runtime(read_maps) if default_dir else None
        if loaded and loaded not in _logged:
            _logged.add(loaded)
            print(f"hip_lib: {_NAME} not found in {lib_dir}, using the copy torch already loaded: {loaded}",
                  file = sys.stderr, flush = True)
        return loaded
    best = max(versioned, key = _version_key)
    if best not in _logged:
        _logged.add(best)
        print(f"hip_lib: {_NAME} not found in {lib_dir}, using {os.path.basename(best)}", file = sys.stderr, flush = True)
    return best


# Windows: the runtime is amdhip64_<major>.dll in _rocm_sdk_core\bin (rocm_sdk _dist_info: "amdhip64*.dll"), loaded by
# `import torch`. A JIT kernel must use that same module instance: loading another copy of the DLL creates a second
# HIP context, in which torch's pointers and streams are invalid (the bug triton-windows fixed in 7feb4df8).
_WIN_NAME = re.compile(r"amdhip64(_\d+)?\.dll", re.I)


def _win_candidate_dirs() -> list[str]:
    dirs = []
    try:
        import torch
        dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:  # noqa: BLE001
        pass
    try:
        import rocm_sdk
        dirs += [str(p.parent) for p in rocm_sdk.find_libraries("amdhip64")]
    except Exception:  # noqa: BLE001
        pass
    return dirs


def _win_module_path(basename: str) -> str | None:
    """Full path of `basename` if this process has it loaded (GetModuleHandleW), else None. Never loads anything."""
    k32 = ctypes.WinDLL("kernel32", use_last_error = True)
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    h = k32.GetModuleHandleW(basename)
    if not h:
        return None
    buf = ctypes.create_unicode_buffer(32768)
    n = k32.GetModuleFileNameW(h, buf, len(buf))
    return buf.value[:n] if n else None


def find_hip_runtime_windows(dirs: list[str] | None = None, list_dir = os.listdir, module_path = None) -> str | None:
    """Path of the amdhip64*.dll this process already loaded (searched by the DLL names found in `dirs`, default: the
    torch lib directory and rocm_sdk's), else the highest-versioned one on disk, else None."""
    module_path = module_path or _win_module_path
    names = set()
    for d in (dirs if dirs is not None else _win_candidate_dirs()):
        try:
            names |= {(d, f) for f in list_dir(d) if _WIN_NAME.fullmatch(f)}
        except OSError:
            continue
    for _, f in sorted(names, key = lambda x: _version_key_win(x[1]), reverse = True):
        loaded = module_path(f)
        if loaded:
            return loaded
    if not names:
        return None
    d, f = max(names, key = lambda x: _version_key_win(x[1]))
    path = os.path.join(d, f)
    if path not in _logged:
        _logged.add(path)
        print(f"hip_lib: no amdhip64 DLL loaded in this process yet (import torch first); using {path}", file = sys.stderr, flush = True)
    return path


def _version_key_win(name: str):
    return tuple(int(x) for x in re.findall(r"\d+", name[len("amdhip64"):]))


def load_hip_runtime(lib_dir: str | None = None) -> ctypes.CDLL:
    if sys.platform == "win32":
        path = find_hip_runtime_windows([lib_dir] if lib_dir else None)
        if path is None:
            raise OSError(f"no amdhip64*.dll in {lib_dir or 'the torch lib directory or the rocm_sdk core bin directory'}")
        return ctypes.CDLL(path)   # same full path as the loaded module -> the same instance, no second HIP context
    path = find_hip_runtime(lib_dir)
    if path is None:
        raise OSError(f"no {_NAME} or {_NAME}.<version> in {lib_dir or 'the torch lib directory'}")
    return ctypes.CDLL(path)
