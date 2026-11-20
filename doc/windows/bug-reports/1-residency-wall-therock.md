# Report 1 (ROCm/TheRock)

=== BEGIN ===

**Title:** `[Windows][Strix Halo gfx1151] GPU work fails (719) or never completes once ~61 GiB of device memory are allocated; HIP/PAL expose no such limit`

## Summary

On Windows with dynamic (shared) GPU memory, a HIP process that has allocated about 61 GiB or more can no longer run GPU work: the next submit either returns `hipErrorLaunchFailure` (719) or never completes. With the default runtime settings every allocation is permanently resident (`alwaysResident_`), so every submit needs all allocations resident; the OS refuses at about 61 GiB (0.48 x the 127.15 GiB of OS-visible RAM) although 41 GiB of RAM are free, the DXGI budget is 101.3 GiB and no WDDM segment reaches its CommitLimit. Nothing in HIP or PAL reports the limit. Models above about 60 GiB cannot be loaded on a 128 GB machine in this mode. Related: #8922 (different ceilings, closed).

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

## Minimal reproduction (pure HIP C++, no torch)

`hipcc hip_wall.cpp -o hip_wall.exe --offload-arch=gfx1151`, then `hip_wall.exe <N>` (N GiB allocated, untouched, then one tiny kernel). After a failure wait ~15 s before the next run.

```cpp
// Pure HIP C++ (no torch, no Python): does a tiny kernel launch fail once many GiB are allocated but untouched?
// Build (VS 2022 Build Tools env + ROCm SDK bin on PATH):
//   hipcc hip_wall.cpp -o hip_wall.exe --offload-arch=gfx1151
// Run:  hip_wall.exe <N_GiB> [memset]       e.g. hip_wall.exe 80      and      set PAL_ALWAYS_RESIDENT=0
// Prints the first failing step. After a failure the GPU context is lost; wait ~15 s before the next run.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call, what) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    printf("FAIL %s: %d (%s)\n", what, (int)e_, hipGetErrorString(e_)); return 1; } } while (0)

__global__ void tiny(int* p) { if (threadIdx.x == 0) p[0] = 42; }

int main(int argc, char** argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 40;
    bool do_memset = argc > 2 && !strcmp(argv[2], "memset");
    const size_t G = 1ull << 30;
    CHECK(hipSetDevice(0), "hipSetDevice");
    size_t fr = 0, tot = 0;
    CHECK(hipMemGetInfo(&fr, &tot), "hipMemGetInfo");
    printf("hipMemGetInfo free %.2f total %.2f GiB\n", fr / (double)G, tot / (double)G);

    std::vector<void*> bufs;
    for (int i = 0; i < n; ++i) {
        void* p = nullptr;
        CHECK(hipMalloc(&p, G), "hipMalloc");
        bufs.push_back(p);
        if (do_memset) {
            CHECK(hipMemset(p, 1, G), "hipMemset");
            CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after memset");
        }
    }
    printf("allocated %d GiB%s\n", n, do_memset ? " (each memset+synced)" : " (untouched)");

    int* small = nullptr;
    CHECK(hipMalloc((void**)&small, 4096), "hipMalloc small");
    hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, 0, small);
    CHECK(hipGetLastError(), "kernel launch");
    CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after kernel");
    int h = 0;
    CHECK(hipMemcpy(&h, small, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy D2H");
    printf("OK: tiny kernel ran with %d GiB allocated, value %d\n", n, h);
    return 0;
}
```

## Observations (all measured on this machine)

```yaml
default_settings:
  hip_wall 40: ok
  hip_wall 65 and 80: FAIL hipMalloc small: 719   # also seen as a hang (no result in 75 s) for memset / copies / kernels after 80 GiB allocated
  allocate+memset+sync in 1 GiB steps (pure C++, ctypes, torch fill_): first failure after 49-62 GiB, error 719
  same wall for hipMalloc, hipExtMallocWithFlags (fine-grained, uncached), hipMallocManaged: 60-61 GiB; hipHostMalloc: 45 GiB
  chunk size (256 MiB .. 16 GiB) does not set it
  pinned host memory counts 1:1 (16 GiB pinned -> device wall 44 GiB; 32 GiB pinned -> 29 GiB); pageable host memory counts about half
log_at_failure: ["PAL failed to submit CMD! result:-5", "PAL failed to finalize a command buffer! result: -28", "PAL reports the device unusable! result:-5"]
  # -5 = Pal::Result::ErrorOutOfGpuMemory (palUtil.h), returned by iQueue_->Submit() at palvirtual.cpp:460
os_side:
  free_ram_at_failure_gib: 41
  event_logs: no dxgkrnl / TDR / WER entry; adapter TDR count 0
  D3DKMTQueryStatistics (wddm_segments.cpp, read-only): AMD adapter segments {#0 CommitLimit 0.33 GiB; #1 7.98 GiB aperture; #2 119.17 GiB aperture; #3 aperture, no CommitLimit}
    all HIP allocations sit in #3; BytesResident 30.23 GiB with 30 GiB held; 61.33 GiB at the failure; 0.04 GiB after the context is lost
  DXGI QueryVideoMemoryInfo local budget: 101.30 GiB (unchanged while another process held 8 or 32 GiB); SharedSystemMemory 101.72 GiB
exposed_by_hip_or_pal: nothing near 61 GiB (hipDeviceGetAttribute 0..399 and 10000..10099, hipDeviceGetLimit 0..11, hipMemGetInfo total 98.22 GiB on the built runtime; PAL GART heaps 101.72 GiB, maxPhysicalMemSize 102.22 GiB, budget 101.30 GiB)
runtime_flag: PAL_ALWAYS_RESIDENT=0 avoids the failure in these tests but is not usable (see <LINK_3>)
fixed_carveout: on the same machine with a fixed 96 GB carve-out (Windows saw 31.6 GB RAM) the same allocations worked up to ~100 GiB and 61-64 GiB models loaded
```

## Expected

Either the pool PAL advertises (about 101 GiB) is usable, or the runtime reports the real residency limit (`hipMemGetInfo`, `totalGlobalMem`) and fails `hipMalloc` with `hipErrorOutOfMemory` instead of leaving a hung or lost context at the next submit.

## Questions

1. Which component enforces the ~61 GiB resident limit (VidMm policy, AMD KMD, a 26H2 change), and is it tunable?
2. Can the runtime query and report it, and fail the allocation instead of the later submit?
3. Is permanent residency (`alwaysResident_`, `GpuMemoryRefCantTrim`) required for allocations in the aperture heaps?

Related: #8922 (same chip and OS family, ceilings 32/64/100 GiB, closed); rocm-systems #13351 (open: `MakeResident` failure path corrupts accounting, 719 on APUs). Separate reports: <LINK_2> (VMM path), <LINK_3> (`PAL_ALWAYS_RESIDENT=0`), <LINK_4> (stale Windows wheels).

=== END ===
