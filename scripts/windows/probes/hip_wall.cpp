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
