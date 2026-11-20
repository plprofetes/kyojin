// Pure HIP C++: which GPU operations still work after N GiB are allocated (untouched)?
// Build: hipcc hip_ops_after_alloc.cpp -o hip_ops_after_alloc.exe --offload-arch=gfx1151
// Run:   hip_ops_after_alloc.exe <N_GiB> <op> [first]
//   op: memset | memsetasync | d2d | h2d | d2h | kernel_direct | kernel_struct | malloc_only | struct_after_submit | direct_after_submit
//   target buffer: a fresh 64 MiB buffer allocated after the N GiB, or with 'first' the first 1 GiB buffer.
// Run once per process (a failure loses the GPU context; wait ~15 s). Compare PAL_ALWAYS_RESIDENT=1 (default) and 0.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(call, what) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    printf("FAIL %s: %d (%s)\n", what, (int)e_, hipGetErrorString(e_)); return 1; } } while (0)

struct PtrPack { char* p[2]; };
__global__ void k_direct(float* dst, const float* src, size_t n) { size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[i] + 1.0f; }
__global__ void k_struct(size_t n, PtrPack pk) { size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) ((float*)pk.p[1])[i] = ((const float*)pk.p[0])[i] + 1.0f; }

int main(int argc, char** argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 80;
    const char* op = argc > 2 ? argv[2] : "memset";
    bool use_first = argc > 3 && !strcmp(argv[3], "first");
    const size_t G = 1ull << 30, S = 64ull << 20;
    CHECK(hipSetDevice(0), "hipSetDevice");
    std::vector<void*> bufs;
    for (int i = 0; i < n; ++i) { void* p; CHECK(hipMalloc(&p, G), "hipMalloc big"); bufs.push_back(p); }
    printf("allocated %d GiB untouched\n", n);
    if (!strcmp(op, "malloc_only")) { void* p; CHECK(hipMalloc(&p, S), "hipMalloc 64MiB"); CHECK(hipDeviceSynchronize(), "sync"); printf("OK malloc_only\n"); return 0; }
    if (!strcmp(op, "struct_after_submit") || !strcmp(op, "direct_after_submit")) {
        // first GPU submit happens BEFORE the target buffers exist; they are allocated afterwards
        void* w; CHECK(hipMalloc(&w, S), "hipMalloc warmup"); CHECK(hipMemset(w, 0, S), "warmup memset"); CHECK(hipDeviceSynchronize(), "warmup sync");
        char *a, *b; CHECK(hipMalloc((void**)&a, S), "hipMalloc a"); CHECK(hipMalloc((void**)&b, S), "hipMalloc b");
        size_t nn = S / 4;
        if (!strcmp(op, "struct_after_submit")) { PtrPack pk; pk.p[0] = a; pk.p[1] = b; hipLaunchKernelGGL(k_struct, dim3((unsigned)((nn + 255) / 256)), dim3(256), 0, 0, nn, pk); }
        else hipLaunchKernelGGL(k_direct, dim3((unsigned)((nn + 255) / 256)), dim3(256), 0, 0, (float*)b, (const float*)a, nn);
        CHECK(hipGetLastError(), "launch"); CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize");
        printf("OK %s with %d GiB allocated\n", op, n); return 0;
    }
    char* t; size_t tsz;
    if (use_first) { t = (char*)bufs[0]; tsz = S; }
    else { CHECK(hipMalloc((void**)&t, S), "hipMalloc target"); tsz = S; }
    char* t2; CHECK(hipMalloc((void**)&t2, S), "hipMalloc second");
    std::vector<char> host(S, 3);
    size_t nf = tsz / 4;
    if      (!strcmp(op, "memset"))      CHECK(hipMemset(t, 1, tsz), "hipMemset");
    else if (!strcmp(op, "memsetasync")) CHECK(hipMemsetAsync(t, 1, tsz, 0), "hipMemsetAsync");
    else if (!strcmp(op, "d2d"))         CHECK(hipMemcpy(t, t2, tsz, hipMemcpyDeviceToDevice), "hipMemcpy D2D");
    else if (!strcmp(op, "h2d"))         CHECK(hipMemcpy(t, host.data(), tsz, hipMemcpyHostToDevice), "hipMemcpy H2D");
    else if (!strcmp(op, "d2h"))         CHECK(hipMemcpy(host.data(), t, tsz, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    else if (!strcmp(op, "kernel_direct")) hipLaunchKernelGGL(k_direct, dim3((unsigned)((nf + 255) / 256)), dim3(256), 0, 0, (float*)t, (const float*)t2, nf);
    else if (!strcmp(op, "kernel_struct")) { PtrPack pk; pk.p[0] = t2; pk.p[1] = t; hipLaunchKernelGGL(k_struct, dim3((unsigned)((nf + 255) / 256)), dim3(256), 0, 0, nf, pk); }
    else { printf("unknown op\n"); return 2; }
    CHECK(hipGetLastError(), "launch/last error");
    CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize");
    printf("OK %s%s with %d GiB allocated\n", op, use_first ? " (first big buffer)" : "", n);
    return 0;
}
