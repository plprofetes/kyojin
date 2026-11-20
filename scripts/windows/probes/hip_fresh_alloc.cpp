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
