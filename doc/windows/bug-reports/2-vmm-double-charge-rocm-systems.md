# Report 2 (ROCm/rocm-systems)

=== BEGIN ===

**Title:** `[clr/pal][Windows][gfx1151] hipMemMap fails with 719 after ~31 GiB (half the hipMalloc limit): VMM memory seems charged twice below clr`

## Summary

On Windows (PAL), the virtual memory management path (`hipMemAddressReserve` + `hipMemCreate` + `hipMemMap` + `hipMemSetAccess`) fails with `hipErrorLaunchFailure` (719) in `hipMemMap` after about **31 GiB** are mapped, independent of chunk size and of touching the memory. `hipMalloc` on the same machine fails at about 61 GiB (see <LINK_1>), so VMM reaches half of that. With `PAL_ALWAYS_RESIDENT=0` the same program maps 80 GiB. Source reading (rocm-systems develop @ 3ac14f9893b9): clr adds one `AddGpuMemoryReferences(..., GpuMemoryRefCantTrim)` per physical allocation (`GpuMemoryReference::MakeResident`, `palresource.cpp:35-41`, reached from `hipMemCreate` through the normal `Create` path) and none in `hipMemMap` (`Device::virtualMap`, `paldevice.cpp:2615-2675`, only `RemapVirtualMemoryPages`), so the second charge, if there is one, is below clr (PAL or the kernel-mode driver). The wall does not depend on the size of the reserved VA range (N=80 and N=100 fail at the same chunk).

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

`hipcc hip_vmm.cpp -o hip_vmm.exe --offload-arch=gfx1151`; `hip_vmm.exe 80 touch` (map and touch 1 GiB at a time), `hip_vmm.exe 80 late` (map 80 GiB, then one memset), `hip_vmm.exe 80 kernel`; the optional third argument is the chunk size in MiB. One process per run; wait ~15 s after a failure.

```cpp
// Pure HIP C++: does the virtual-memory-management path (hipMemCreate / hipMemMap) avoid the ~61 GiB residency wall?
// Build: hipcc hip_vmm.cpp -o hip_vmm.exe --offload-arch=gfx1151
// Run:   hip_vmm.exe <N_GiB> <mode> [chunk_MiB=1024]
//   mode touch : after mapping each chunk, hipMemset it and synchronize (allocate-and-touch ladder)
//   mode late  : map all N GiB first, then one hipMemset of a 64 MiB window at the end (the case that hangs with hipMalloc)
//   mode kernel: like late, but the final operation is a tiny kernel on the first chunk
// One process per run (a failure loses the context; wait ~15 s). Compare PAL_ALWAYS_RESIDENT=1 (default) and 0.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(call, what) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    printf("FAIL %s: %d (%s)\n", what, (int)e_, hipGetErrorString(e_)); fflush(stdout); return 1; } } while (0)

__global__ void tiny(int* p) { if (threadIdx.x == 0) p[0] = 42; }

int main(int argc, char** argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 40;
    const char* mode = argc > 2 ? argv[2] : "touch";
    size_t chunk = (argc > 3 ? (size_t)atoll(argv[3]) : 1024) << 20;
    CHECK(hipSetDevice(0), "hipSetDevice");
    int vmm = -1; hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0);
    printf("VirtualMemoryManagementSupported = %d\n", vmm);

    hipMemAllocationProp prop; memset(&prop, 0, sizeof(prop));
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    size_t gran = 0;
    CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum), "hipMemGetAllocationGranularity");
    printf("granularity %zu bytes; chunk %zu MiB\n", gran, chunk >> 20);
    if (chunk % gran) { printf("chunk not a multiple of granularity\n"); return 2; }

    size_t total = (size_t)n << 30;
    size_t nchunks = total / chunk;
    void* base = nullptr;
    CHECK(hipMemAddressReserve(&base, total, 0, nullptr, 0), "hipMemAddressReserve");
    printf("reserved %d GiB of address space at %p\n", n, base);

    hipMemAccessDesc acc; memset(&acc, 0, sizeof(acc));
    acc.location = prop.location; acc.flags = hipMemAccessFlagsProtReadWrite;

    std::vector<hipMemGenericAllocationHandle_t> handles;
    bool touch = !strcmp(mode, "touch");
    for (size_t i = 0; i < nchunks; ++i) {
        hipMemGenericAllocationHandle_t h;
        char what[64];
        snprintf(what, sizeof(what), "hipMemCreate chunk %zu", i);
        CHECK(hipMemCreate(&h, chunk, &prop, 0), what);
        char* p = (char*)base + i * chunk;
        snprintf(what, sizeof(what), "hipMemMap chunk %zu", i);
        CHECK(hipMemMap(p, chunk, 0, h, 0), what);
        snprintf(what, sizeof(what), "hipMemSetAccess chunk %zu", i);
        CHECK(hipMemSetAccess(p, chunk, &acc, 1), what);
        handles.push_back(h);
        if (touch) {
            snprintf(what, sizeof(what), "hipMemset chunk %zu", i);
            CHECK(hipMemset(p, 1, chunk), what);
            CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize");
            if ((i + 1) % 8 == 0) { printf("  %zu GiB mapped+touched\n", ((i + 1) * chunk) >> 30); fflush(stdout); }
        }
    }
    printf("mapped %d GiB via VMM (%s)\n", n, mode); fflush(stdout);
    if (!touch) {
        if (!strcmp(mode, "kernel")) {
            hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, 0, (int*)base);
            CHECK(hipGetLastError(), "kernel launch");
        } else {
            CHECK(hipMemset(base, 1, 64ull << 20), "hipMemset 64 MiB window");
        }
        CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize (final)");
    }
    printf("OK vmm %s with %d GiB\n", mode, n);
    return 0;
}
```

## Results

```yaml
VirtualMemoryManagementSupported: 1     # PAL: paldevice.cpp:637
granularity: 65536 bytes
default_settings:
  1 GiB chunks, touch / late / kernel / 100 GiB ladder: "FAIL hipMemMap chunk 31: 719"   # about 31 GiB mapped
  256 MiB chunks: "FAIL hipMemMap chunk 124: 719"                                        # 31 GiB again
  4 GiB chunks: 719 at the first touch/sync
PAL_ALWAYS_RESIDENT=0:
  1 GiB chunks, late: ok (80 GiB mapped, 64 MiB memset)   # unsafe mode, see <LINK_3>
hipMalloc_for_comparison: ~61 GiB wall (hip_wall.cpp in <LINK_1>)
```

## Question

Is a physical allocation that is mapped into a reserved VA range charged against the residency limit twice (once as the physical allocation, once as the mapping) in PAL or the kernel-mode driver? Can it be charged once so VMM reaches the same limit as `hipMalloc`?

=== END ===
