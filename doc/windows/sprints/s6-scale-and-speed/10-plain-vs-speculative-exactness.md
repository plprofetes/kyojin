# Plain decode and speculative decode are not bitwise equal on every prompt: find the first differing module

- Status: todo
- Depends: S4-05
- Risk: MEDIUM: the property "plain == speculative" is documented for `--draft-policy`; the module that breaks it is not known, and tiny numeric differences only show at near-tie tokens.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-10 · priority order 10 in the sprint

**Marking:** **[V]** measured on the target machine, **[I]** inferred, **[?]** unknown.

## Summary

On a 4K prompt (`tools/sweep.py` prompt, greedy, 256 new tokens, Qwen3.8-27B 4.0 bpw), the reply depends on the draft depth in the **default** configuration **[V, 2026-11-02]**: `--draft-policy mix` with `QWSPEC_FIXD` 1 gives sha1 `5f21272fc4`, 3 gives `f2b7fbf8b0`, 7 gives `9ab3086c7f` = plain decode. The texts diverge at character 403 ("fought for freedom" / "fought for their freedom"), a near-tie token. On eight other 2K prompts (200 tokens) the default mix and plain replies were identical. The kbs table (`EXL3_DEC_KBS_TAB`, S6-08) moved the draft composition and showed the same variant with the default depth; it is not the cause (dense MLP GEMVs are bitwise row-consistent for R = 1..32 on real weights **[V]**).

So some module in the verify forward (attention verify rows, GatedDeltaNet rows or state handling, the MTP drafter's state, norms, or the lm_head rows) produces a row that differs in the last bits from the plain step for the same context, at least for 2 and 4 verify rows. Not located.

## Steps

1. Reproduce: the 4K prompt of `tools/sweep.py` (`prompt(txt, 4096, 0, tokenizer)`), default build, `--draft-policy mix`, `QWSPEC_FIXD` 1 / 3 / 7 against `--draft-policy off`. Scratch drivers used: `ident.py`, `ident2.py` (journal 2026-11-02).
2. Compare one plain step with one verify forward of 2 rows for the same context, module by module (hook each module's output, bitwise per row). `tools/qwen/batch_inv.py` stage `ops` does this for a job alone against the same job next to others; it needs a plain-vs-verify variant.
3. Name the first differing module and decide: make it row-exact, or document the property as "equal up to last-bit differences" and say so in the docs that claim bit-identity.

## Done when

- The first differing module is named with a bitwise reproduction, and either fixed (the 4K prompt gives the same reply for every draft depth) or the claim is corrected in the docs.

## If it fails

- If the difference comes from a kernel that cannot be made row-exact cheaply, record it and weaken the documented property instead.

## Links

- [S6-08](08-decode-gemv-upstream-exl3-gemm.md), [journal](../../journal.md) (2026-11-02 entries), [qwen38-27b-decode-vs-upstream-20261102](../../../benchmarks/qwen38-27b-decode-vs-upstream-20261102.md)
