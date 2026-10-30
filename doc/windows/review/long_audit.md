# `long` under LLP64 (Windows): audit for the Qwen3.5-9B dense path

Repo c4a9f87 (sources identical to cca1e17). Method: include closure of the 72 ROCm TUs
(scripts/incl.py over build_config.get_sources, 191 files), plus the `.hip` JIT sources. Grep for `\blong\b`
excluding `long long`. "Qwen path" = the include closure of exl3_gemv, exl3_gemv_int8, exl3_dec, reconstruct,
exl3_devctx, hgemm, hgemm_f16acc, gdn, fused_elt_rocm, rope, add, activation_rocm, generator/*, stloader*,
quant/hadamard, graph, bindings and the other host .cpp (129 files, list in /tmp/claude-0/review/qwenpath.txt).

Premise [I]: clang compiling HIP for an x86_64-pc-windows-msvc host uses the host data model for both sides, so
`long` is 32-bit in device code as well as host code (LLP64). On Linux it is 64-bit. `int64_t` and `long long` are
64-bit on both.

## Result

**No `long` arithmetic is on the Qwen3.5-9B dense decode/prefill path.** The 12 hits in the Qwen-path closure come from:
- comments (exl3_gemv_int8_kernel.cuh:1096, hgemm_f16acc.cu:475, gdn.cu:152);
- `quant/exl3_moe_valu.cuh:97` and `quant/exl3_moe_valu_idx.h:15-30`, pulled in by `quant/exl3_gemv.cu:24`.
  That is the VALU grouped-MoE decode path (EXL3_MOE_VALU, "K2, K3 and K4 experts"), which a dense model never runs.

The JIT `.hip` sources reached on the Qwen3.5 path (vendor/fla/hip/gdn_fused_h.hip, gdn_hf.hip) contain no `long`.
The only `.hip` files with `long` are DS41 (dsv41_hip.hip 14, dsv41_idx.hip 9, ds41_pf.hip 4).

**No fix is needed for the POC.**

## Ranked watch list (for later MoE / Qwen3.8 work; none needed now)

| Rank | Site | Expression | Overflows when | Assessment |
|---|---|---|---|---|
| 1 | quant/exl3_moe_fused_idx.h:100 | `expert_base(long expert_u32, int e) = (long) e * expert_u32` (u32 index into one layer's expert table) | the expert table of one layer exceeds 2^31 u32 = 8 GiB | Low: Qwen3.8 K6 (256 experts, about 1.1M u32 each) is about 2.8e8 u32 (1.1 GiB). It is the first to check for larger MoE packs. |
| 2 | quant/exl3_moe_fused_idx.h:77-78 | `EXPERT_U32`, `SHARED_U32` (constexpr per shape) | one expert's weights exceed 8 GiB | Not reachable. |
| 3 | quant/exl3_moe_fused9_idx.h:12-22, exl3_moe_fused9.cuh:17-56 | `touch_*` byte counts for prefetch "touch" of one expert/shared/router block | a single block exceeds 2 GiB **in bytes** (these count bytes, not u32) | Low: per-expert and per-router blocks are MB-sized; `router_bytes = NEXP*D*2` is about 1.3 MB for 256x2560. |
| 4 | quant/exl3_moe_fused.cuh:230,293,344; exl3_moe_fused7_mv.cuh:55,126,177; exl3_moe_valu.cuh:97 | `long ss` (k-slice stride), `a_idx = r*(k/2)+8s` | k*n per expert exceeds 2^31 | Not reachable. |
| 5 | kda_inter.cu:37-39,146 | `bos = (long) i_b * T`, strides `H*KD` | B*T*H*KD exceeds 2^31 | KDA (Kimi linear) only, not Qwen. T=4096 chunk, H=32, KD=128 gives 1.7e7. Safe. |
| 6 | modules/dsv41_fn/dsv41_hip.py:72, ds41_pf.py:58 | `ctypes.c_long(a[1])` kernel arguments | any value at or above 2^31; on Windows c_long is 32-bit | DS41 is out of scope. Use `c_int64` if DS41 ever comes to Windows. |
| 7 | host counters: cpu/moe_handoff_amdgpu.cu:382, cpu/moe_mul1.cpp:2426, ngram_gather_win.cpp:53-56 (`std::atomic<long>` timers/counters) | profiling event counters (ngram_gather_win.cpp:168,239,267 increment by 1, only when gather profiling is on) | above 2^31 events | Not reachable. Stats only, Qwen3.8 PLE path only. |

Not hazards: `unsigned long long` (64-bit everywhere), `int64_t`/`size_t` uses, and `c10::optional<at::Tensor>` comments
in libtorch/dsv4_compressor.h (excluded from ROCm).

Same-on-both-OS `int` overflows (32-bit `int` offsets) exist independently of LLP64 and are not in scope here.
