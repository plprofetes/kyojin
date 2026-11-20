# Report 3 (ROCm/rocm-systems)

=== BEGIN ===

**Title:** `[clr/pal][Windows] PAL_ALWAYS_RESIDENT=0 leads to GPU page faults (PAL -87) for kernels whose pointers are passed inside a by-value struct (PyTorch)`

## Summary

`PAL_ALWAYS_RESIDENT=0` (`palsettings.cpp:411`) is the only switch that removes the ~61 GiB residency wall on Windows APUs (see <LINK_1>), but it is not usable. HIP then tracks residency per command from the kernel's declared pointer parameters only (`palvirtual.cpp:3817-3873`, `T_POINTER` parameters; `addCmdMemRef` is a no-op while `alwaysResident_` is true, `palvirtual.cpp:334-337`). Kernels that receive their buffer addresses inside a by-value struct, as PyTorch's TensorIterator kernels do (`std::array<char*,N>`), reference memory HIP does not know about, and the GPU takes a page fault: `PAL failed to submit CMD! result:-87` (`ErrorGpuPageFaultDetected`), then every call returns 719. Request: document the flag as unsafe, and consider a safe mode (conservative references for struct arguments, or an opt-in tracking API).

## Environment

```yaml
machine: ASUS ProArt PX13 HN7306EAC, BIOS v3.14 (HN7306EAC.314)
cpu_gpu: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S   # gfx1151
ram_visible_to_os_gib: 127.15                    # dynamic ("shared") GPU memory mode; dedicated carve-out ~0.33 GiB
os: Windows 11 Pro 26H2, build 26300.9974, DxgKrnlVersion 69639
amd_driver: Adrenalin 26.11.2, driver store 32.0.32029.2010, release 26.20.29.02-261116a (also seen on 26.10.1 / 32.0.32015.7029)
hip_runtime_stock: amdhip64_7.dll 10.0.3581.0 (wheel rocm 10.1.0a20260822, HIP 7.16.26332); driver System32 copy 10.0.3686.0
hip_runtime_built: HIP 7.18.26465 built from rocm-systems develop @ 3ac14f9893b96062f1767421c93700466bb13fdb (PAL backend, kpack on)
```

## Reproduction (pure HIP C++)

`hipcc hip_fresh_alloc.cpp -o hip_fresh_alloc.exe --offload-arch=gfx1151`; `set PAL_ALWAYS_RESIDENT=0`; `hip_fresh_alloc.exe 8 10 struct` (pointers in a struct) versus `hip_fresh_alloc.exe 8 10` (direct pointer arguments). Wait ~15 s after a failure.

```cpp
// Pure HIP C++: with PAL_ALWAYS_RESIDENT=0, does a kernel that reads/writes freshly hipMalloc'd buffers fault?
// Mirrors the torch pattern  a = torch.empty(N fp16); a.zero_(); b = a.float(); sync  (src MiB, dst 2x MiB), repeated.
// Build: hipcc hip_fresh_alloc.cpp -o hip_fresh_alloc.exe --offload-arch=gfx1151
// Run:   hip_fresh_alloc.exe <src_MiB> <reps> [free|struct|free,struct]     e.g. hip_fresh_alloc.exe 8 10      (set PAL_ALWAYS_RESIDENT=0)
// 'free' frees the buffers every iteration; 'struct' passes the pointers inside a by-value struct like torch kernels do.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CHECK(call, what) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    printf("FAIL %s at rep %d: %d (%s)\n", what, rep, (int)e_, hipGetErrorString(e_)); return 1; } } while (0)

__global__ void h2f(const unsigned short* a, float* b, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (float)a[i];
}

// torch passes tensor pointers inside a by-value struct (std::array<char*, N>), not as direct pointer arguments
struct PtrPack { char* p[2]; };
__global__ void h2f_struct(size_t n, PtrPack pk)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ((float*)pk.p[1])[i] = (float)((const unsigned short*)pk.p[0])[i];
}

int main(int argc, char** argv)
{
    double mib = argc > 1 ? atof(argv[1]) : 8;
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    bool do_free = argc > 3 && strstr(argv[3], "free");
    bool use_struct = argc > 3 && strstr(argv[3], "struct");   // pointers passed inside a struct, as torch does
    size_t src_bytes = (size_t)(mib * 1048576.0), n = src_bytes / 2;
    int rep = -1;
    CHECK(hipSetDevice(0), "hipSetDevice");
    for (rep = 0; rep < reps; ++rep) {
        unsigned short* a; float* b;
        CHECK(hipMalloc((void**)&a, src_bytes), "hipMalloc src");
        CHECK(hipMalloc((void**)&b, n * 4), "hipMalloc dst");
        CHECK(hipMemset(a, 0, src_bytes), "hipMemset src");                 // torch: a.zero_()
        if (use_struct) { PtrPack pk; pk.p[0] = (char*)a; pk.p[1] = (char*)b;
            hipLaunchKernelGGL(h2f_struct, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, 0, n, pk); }
        else hipLaunchKernelGGL(h2f, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, 0, a, b, n);
        CHECK(hipGetLastError(), "kernel launch");
        CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize");
        if (do_free) { hipFree(a); hipFree(b); }
    }
    printf("OK: %d reps of %.1f MiB fp16 -> fp32%s\n", reps, mib, do_free ? " (freed each rep)" : " (kept)");
    return 0;
}
```

## Results

```yaml
PAL_ALWAYS_RESIDENT=0:
  hip_fresh_alloc 8 10 struct:  "FAIL hipDeviceSynchronize at rep 1: 719"      # same for 16 MiB
  hip_fresh_alloc 8 10:         "OK: 10 reps ... direct pointer args"
PAL_ALWAYS_RESIDENT=1 (default): both OK
single struct-pointer kernel on fresh buffers: OK   # so the failing condition is not fully understood; it needs a second round or torch's allocation pattern
pytorch (stock runtime): x = torch.empty(N fp16); x.zero_(); y = x.float(); torch.cuda.synchronize(), repeated: 4-16 MiB tensors fail after 1-2 iterations; <= 2 MiB pass
log: "PAL failed to submit CMD! result:-87", "PAL failed to finalize a command buffer! result: -28"
real workload: an EXL3 model load (many torch ops) dies at the first torch conversion kernel with only 0.15 GiB allocated
```

## Questions

1. Is `PAL_ALWAYS_RESIDENT=0` supported on Windows, or only a debugging switch?
2. Could per-command tracking include pointers embedded in struct arguments (conservatively), or could a safe mode keep recently used allocations referenced?
3. What is the exact residency rule that lets a single struct-pointer kernel pass but fails repeated rounds?

=== END ===
