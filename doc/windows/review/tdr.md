# TDR (GPU timeout detection and recovery) on Windows for the gfx1151 POC

Sources: MicrosoftDocs/windows-driver-docs @ b1f6aa5c1275777b7b4be30c4bbff86dc614e0b0 (branch staging),
windows-driver-docs-pr/display/tdr-registry-keys.md (ms.date 11/05/2025) and
display/timeout-detection-and-recovery.md. learn.microsoft.com itself is blocked from this sandbox; the raw GitHub
copy was read instead.

## Registry values [V, tdr-registry-keys.md]

All live under `HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers`, type **REG_DWORD**.
Changes take effect after a reboot [I; standard for these keys, not stated on the page read].

| Value | Meaning | Default | Suggested for the POC |
|---|---|---|---|
| TdrDelay | Seconds the GPU may delay the scheduler's preempt request; "effectively the timeout threshold" | 2 | 60 while debugging, then lower (e.g. 10) once a first run shows the longest kernel |
| TdrDdiDelay | Seconds threads may stay in the driver before bug check VIDEO_TDR_FAILURE (0x116) | 5 | 60 (keep it at or above TdrDelay) |
| TdrLevel | 0 off, 1 bug check, 3 recover (default) | 3 | leave at 3. Do not use 0 on the only display adapter (Strix Halo drives the desktop) |
| TdrLimitTime / TdrLimitCount | 5 TDRs (0x117) within 60 s bug-check the machine | 60 / 5 | leave |
| TdrDebugMode | 2 = recover without debugger (default) | 2 | leave |

Microsoft's page says these keys are for "testing or debugging ... during driver development" and that end users
should not change them. Treat this as a debugging step and record it in the journal.

Commands (elevated cmd.exe), then reboot:
```bat
reg add "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
reg add "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 60 /f
reg query "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay
```
Undo: `reg delete "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /f` (same for TdrDdiDelay), then reboot.

## Detecting a TDR

- [V, timeout-detection-and-recovery.md] On recovery Windows shows "Display driver stopped responding and has
  recovered" and logs it in Event Viewer. The screen flickers. Five hangs in 60 s cause a bug check 0x117.
  Engine timeouts are 0x141; a process blocked from the GPU is 0x142.
- [I] Event Viewer > Windows Logs > System, Source **Display**, **Event ID 4101** ("Display driver <name> stopped
  responding and has successfully recovered"). Query it from the command line:
  ```bat
  wevtutil qe System /q:"*[System[Provider[@Name='Display'] and (EventID=4101)]]" /c:5 /rd:true /f:text
  ```
- [I] Live kernel reports (dumps) appear under `C:\Windows\LiveKernelReports\WATCHDOG\` (0x141/0x117).
- In the process [?]: HIP calls after the reset fail. Expect a torch `RuntimeError: HIP error: ...` (wording
  unverified; possibly "unspecified launch failure" or a device-lost error). The process must be restarted.
  kyojin_win should, on any HIP error, run the wevtutil query above and put its output in FAILURE.md.
- [?] Whether AMD's Windows driver preempts HIP compute mid-kernel (so long kernels are safe) is not documented
  anywhere I could reach. Assume it does not until a run shows otherwise. The triton-windows readme lists no TDR issue.

## Which Kyojin GPU work could run longer than 2 s in one submission on gfx1151

Ranked by risk for the POC (Qwen3.5-9B, `-c 8192`, `--draft-policy off`):

1. **Persistent wait kernels of the CPU-MoE handoff**
   - Code: cpu/moe_handoff_amdgpu.cu:1-20 and 115-135. These kernels spin on a host flag with
     `__builtin_amdgcn_s_sleep`, with a calibrated budget of about 30 s ("the ~30 s wait timeout").
   - Any wait longer than TdrDelay is a TDR.
   - Only used when MoE experts are offloaded to the CPU (moe_cpu_host.py), not by a dense 9B model.
   - **Certain TDR risk for Qwen3.8 CPU-MoE on Windows** (Sprint 6); it needs TdrDelay > 30 or a redesign.
2. **Long-context prefill attention**
   - One launch per full-attention layer per 4096-row chunk: EXL3_PREFILL_CHUNK=4096 from SERVE_ENV, serve.py:512-600,
     and generator.py:296-302.
   - FLOPs grow with context. Rough cost for a 4096-row chunk at 64K context: about 4096 x 65536 x hd(256) x heads(16) x 4
     = 4.4e12 FLOP, so about 0.3 s at 15 TFLOPS. At 256K it is about 1.2 s, close to 2 s [I, estimate].
   - Not an issue at `-c 8192`. Mitigation: `set EXL3_PREFILL_CHUNK=1024`.
   - Kernel: modules/attention_fn/triton_paged.py (Triton prefill). CK paths are Linux-only.
3. **First-launch dense-GEMM screening**
   - Code: model/dense_warmup.py:51-100 and hgemm.cu dtune (EXL3_WARM_DENSE, EXL3_DENSE_GEMM_TUNE).
   - Many short rocBLAS GEMMs (a 4096x4096x12288 GEMM is about 16 ms at 25 TFLOPS) [I]. Minutes in total, but
     each submission is short, so low risk.
   - `set EXL3_WARM_DENSE=0` skips it for a first smoke run.
4. **Triton first-call autotune**
   - Code: vendor/fla/* `@triton.autotune`, kernel_seed absent for Triton 3.8.
   - Many short timed runs. Low risk per launch, long wall time.
5. **Large host-to-device copies during load**
   - Code: stloader.cpp:255-265 (`cudaMemcpyAsync` per worker chunk) and 344 (pinned staging).
   - Chunked, MB-sized. Low.
6. **Grid-wide barriers: not on the HIP path**
   - quant/exl3_gemv_kernel.cuh:17-19,27-29,277-295: the cooperative `grid.sync()` is CUDA-only (`#if !defined(USE_ROCM)`).
     "HIP launches the Hadamard stages separately" [V].
   - hc_mix.cu:785 is explicitly "no-grid-sync".
   - No cooperative launch exists in the ROCm TU set outside parallel/ (excluded) [V].

For the first run: `-c 8192`, EXL3_PREFILL_CHUNK=1024, EXL3_WARM_DENSE=0, TdrDelay 60. If Event 4101 appears, the
log line just before it names the kernel (EXL3_LOG_JIT=1 also logs Triton compiles).
