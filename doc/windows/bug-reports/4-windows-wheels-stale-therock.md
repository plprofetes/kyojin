# Report 4 (ROCm/TheRock)

=== BEGIN ===

**Title:** `[Windows][gfx1151] Public nightly wheels/tarballs stop at 10.1.0a20260822 and lack rocm-systems #10974 and #13193 (hipMemGetInfo total and 2:1 accounting)`

## Summary

The newest Windows `torch`, `rocm` and `therock-dist-windows-gfx1151` artifacts on `https://rocm.nightlies.amd.com/whl-multi-arch/` and `tarball-multi-arch/` are `10.1.0a20260822` (HIP 7.16.26332, `amdhip64_7.dll` 10.0.3581.0). They predate rocm-systems #10974 (merged 2026-09-04) and #13193 (merged 2026-11-02), so on a Strix Halo APU in dynamic GPU memory mode `hipMemGetInfo` is wrong and applications that trust it fail early. A runtime built from rocm-systems `develop` shows the fixes work. Request: publish current Windows gfx1151 nightlies (and the 10.2 line), or say where they are published.

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

## Evidence

```yaml
index_listing: newest Windows cp312 torch on whl-multi-arch is 2.15.0a0+rocm10.1.0a20260822; rocm 10.1.0a20260822; therock-dist-windows-gfx1151-10.1.0a20260822   # whl-staging-multi-arch, v2, v2-staging hold older 7.x builds; PyPI rocm has only 0.1.0
stock_runtime (wheel 20260822), 127.15 GiB machine, dynamic GPU memory:
  hipMemGetInfo total: 76.79 GiB
  consumed per allocation: first (largest) allocation 1:1, every later one 2:1   # 1x8 GiB -> 8.15; 2x4 GiB -> 12.15; 8x1 GiB -> 15.15; 32x256 MiB -> 15.90; 512x16 MiB -> 15.97 GiB
  after free + empty_cache: 7.0 GiB (8x1 GiB case) and 11.9 GiB (128x64 MiB case) still counted as used
  effect: a loader that checks free memory (EXL3 autosplit) refuses at ~38 GiB allocated although the machine can hold more
built_runtime (rocm-systems develop @ 3ac14f9893b9, includes #10974 and #13193):
  hipMemGetInfo total: 98.22 GiB; 1:1 for every allocation pattern; free returns to the starting value after free
upstream:
  rocm-systems #10974 (2026-09-04): report the full unified pool, max allocation 256 GiB
  rocm-systems #13193 (2026-11-02): stop double-counting APU allocations in hipMemGetInfo (ROCM-32243)
  Adrenalin 26.11.2 release notes: "Adds support for ROCm 10.2"
note: these fixes do not change the separate ~61 GiB residency wall (<LINK_1>)
```

## Question

Where can Windows gfx1151 users get a build that includes #10974 and #13193 (nightly index, 10.2 release, or tarball)?

=== END ===
