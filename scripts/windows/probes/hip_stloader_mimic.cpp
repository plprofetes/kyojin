// Pure HIP C++ mimic of the exllamav3 native loader engine (exllamav3_ext/stloader_hip.cpp, pool_worker_main +
// stloader_cuda_engine): 8 persistent worker threads, one pinned pool (hipHostMalloc), a ring of 4 slots per
// worker, one non-blocking stream per worker, hipMemcpyAsync H2D per slot, hipEventRecord / hipEventSynchronize
// per slot, hipStreamSynchronize drain per batch, hipDeviceSynchronize by the caller. "File reads" are memsets.
// Build: hipcc hip_stloader_mimic.cpp -o hip_stloader_mimic.exe --offload-arch=gfx1151
// Run:   hip_stloader_mimic.exe [batches=200] [runs_per_batch=16] [kernel]      with and without  set PAL_ALWAYS_RESIDENT=0
// Prints progress per 20 batches; a hang shows as no further output (use a timeout).
#include <hip/hip_runtime.h>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define THREADS 8
#define SLOTS 4
#define SLOT_SIZE (4u * 1024 * 1024)

static uint8_t* pinned_pool = nullptr;
static uint8_t* dest = nullptr;           // device destination buffer
static size_t dest_size = 0;
static std::mutex mtx;
static std::condition_variable cv_work, cv_done;
static uint64_t generation = 0;
static int workers_done = 0;
static std::atomic<int> next_run{0};
static std::atomic<bool> failed{false};
static int runs_in_batch = 0;
static bool with_kernel = false;   // 3rd arg 'kernel': also launch an in-place bf16->fp16-like kernel per copy (stloader does this for bf16 tensors)

__global__ void inplace_convert(unsigned short* p, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (unsigned short)(p[i] + 1);
}

static void fail(const char* what, hipError_t e)
{
    if (!failed.exchange(true)) printf("FAIL %s: %d (%s)\n", what, (int)e, hipGetErrorString(e));
}

static void worker(int idx)
{
    hipEvent_t events[SLOTS] = {};
    bool pending[SLOTS] = {};
    hipStream_t stream = nullptr;
    int slot = 0;
    uint64_t seen = 0;
    hipSetDevice(0);
    while (true)
    {
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_work.wait(lk, [&] { return generation != seen; });
            seen = generation;
        }
        uint8_t* ring = pinned_pool + (size_t)idx * SLOTS * SLOT_SIZE;
        int run;
        while (!failed.load() && (run = next_run.fetch_add(1)) < runs_in_batch)
        {
            uint8_t* buf = ring + (size_t)slot * SLOT_SIZE;
            if (pending[slot]) {
                hipError_t e = hipEventSynchronize(events[slot]);
                if (e != hipSuccess) { fail("hipEventSynchronize", e); break; }
                pending[slot] = false;
            }
            memset(buf, run & 0xff, SLOT_SIZE);                        // stands in for the file read
            if (!stream) {
                hipError_t e = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
                if (e != hipSuccess) { fail("hipStreamCreateWithFlags", e); break; }
            }
            hipError_t e = hipMemcpyAsync(dest + (size_t)(run % (dest_size / SLOT_SIZE)) * SLOT_SIZE, buf, SLOT_SIZE,
                                          hipMemcpyHostToDevice, stream);
            if (e != hipSuccess) { fail("hipMemcpyAsync", e); break; }
            if (with_kernel) {
                size_t n = SLOT_SIZE / 2;
                unsigned short* d = (unsigned short*)(dest + (size_t)(run % (dest_size / SLOT_SIZE)) * SLOT_SIZE);
                hipLaunchKernelGGL(inplace_convert, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, stream, d, n);
                e = hipGetLastError();
                if (e != hipSuccess) { fail("kernel launch", e); break; }
            }
            if (!events[slot]) {
                e = hipEventCreateWithFlags(&events[slot], hipEventDisableTiming);
                if (e != hipSuccess) { fail("hipEventCreateWithFlags", e); break; }
            }
            e = hipEventRecord(events[slot], stream);
            if (e != hipSuccess) { fail("hipEventRecord", e); break; }
            pending[slot] = true;
            slot = (slot + 1) % SLOTS;
        }
        if (stream) {
            hipError_t e = hipStreamSynchronize(stream);
            if (e != hipSuccess) fail("hipStreamSynchronize", e);
        }
        for (int i = 0; i < SLOTS; ++i) pending[i] = false;
        { std::lock_guard<std::mutex> lk(mtx); workers_done++; }
        cv_done.notify_all();
    }
}

int main(int argc, char** argv)
{
    int batches = argc > 1 ? atoi(argv[1]) : 200;
    runs_in_batch = argc > 2 ? atoi(argv[2]) : 16;
    with_kernel = argc > 3 && !strcmp(argv[3], "kernel");
    dest_size = 256ull << 20;
    if (hipSetDevice(0) != hipSuccess) { printf("hipSetDevice failed\n"); return 1; }
    hipError_t e = hipHostMalloc((void**)&pinned_pool, (size_t)THREADS * SLOTS * SLOT_SIZE);
    if (e != hipSuccess) { printf("hipHostMalloc failed %d\n", (int)e); return 1; }
    e = hipMalloc((void**)&dest, dest_size);
    if (e != hipSuccess) { printf("hipMalloc failed %d\n", (int)e); return 1; }

    std::vector<std::thread> pool;
    for (int i = 0; i < THREADS; ++i) { pool.emplace_back(worker, i); pool.back().detach(); }

    for (int b = 0; b < batches; ++b)
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            next_run = 0; workers_done = 0; generation++;
        }
        cv_work.notify_all();
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_done.wait(lk, [&] { return workers_done == THREADS; });
        }
        if (failed) { printf("batch %d: failed\n", b); return 1; }
        e = hipDeviceSynchronize();                                     // as in stloader_cuda_engine
        if (e != hipSuccess) { printf("FAIL hipDeviceSynchronize after batch %d: %d (%s)\n", b, (int)e, hipGetErrorString(e)); return 1; }
        if (b % 20 == 0) { printf("batch %d ok\n", b); fflush(stdout); }
    }
    printf("OK: %d batches x %d runs\n", batches, runs_in_batch);
    return 0;
}
