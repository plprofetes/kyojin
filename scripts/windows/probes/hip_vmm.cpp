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
