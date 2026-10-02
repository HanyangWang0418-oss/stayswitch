# Claude Code harness: compaction-timing results (2026-10-02, mid-model arms final)

Setup: Claude Code 2.1.286 inside SWE-bench Verified containers (Harbor), Anthropic `/v1/messages` → CliffCompaction
proxy → StaySwitch proxy → Qwen3.6-35B-A3B on Tinker (64K window; cached reads 0.2×, no write premium). 40 tasks
(`configs/swe_tasks_all.txt`), seeds 1–3 = 120 trajectories per arm. Cliff text caps 300 (paper setting), keep_recent 3,
Claude Code's own auto-compaction off in every Cliff/EOQ arm (`DISABLE_AUTO_COMPACT=1`). Costs: the CliffCompaction paper's
perfect-cache rule (`scripts/perfect_cache_cost.py`). Tables: `report_seeds.py`, `pass_at_k.py`, `price_regimes.py`, `action_mix.py`.

## Arms
| arm | trigger | what it is |
|---|---|---|
| native | Claude Code auto-compact at 25% of its assumed 200K (~50K real) | the harness's own LLM summarisation |
| cliff45 / cliff40 / cliff32 | fixed, in Cliff's chars/4 estimate | the paper's method at its 45K matched point and below |
| eoq1 | L* = L0 + sqrt(2gC/c_r); g = whole-history mean growth, L0 guessed | our rule, v1 (median trigger ~42K, p10 30K, p90 52K) |
| eoq2 | g = cycle-mean growth, L0 measured | v2: the lower g lowers the trigger to ~33K (a mistake, kept for the record) |

## Main table (40 tasks × 3 seeds)
| arm | solved % (pooled ± se) | calls/task | $/task | mean ctx | compact./task | vs native: Δsolved % | Δ$/task | Δ$/call |
|---|---|---|---|---|---|---|---|---|
| native | 43.6 ± 4.6 | 42.4 | 0.261 | 26.8K | 0.8 | — | — | — |
| cliff45 | 45.8 ± 4.5 | 40.8 | 0.240 | 25.0K | 1.2 | +3.3 ± 3.9 | −0.021 ± 0.020 | −4.3% |
| cliff40 | 47.9 ± 4.6 | 42.0 | 0.257 | 24.6K | 1.6 | +5.0 ± 3.3 | −0.004 ± 0.022 | −0.6% |
| cliff32 | 40.2 ± 4.5 | 43.9 | 0.282 | 21.9K | 3.0 | −3.3 ± 3.7 | +0.021 ± 0.027 | +4.2% |
| **eoq1** | **49.2 ± 4.6** | 47.7 | 0.270 | 24.7K | 1.8 | **+5.8 ± 4.0** | +0.009 ± 0.024 | −8.2% |
| eoq2 | 41.0 ± 4.5 | 45.3 | 0.257 | 24.1K | 2.0 | −2.6 ± 3.8 | −0.004 ± 0.027 | −5.9% |

Test-time scaling from the seeds (3 rollouts, oracle pass@3): native 57.5%, cliff45 55.0, cliff40 55.0, cliff32 52.5,
eoq1 62.5, eoq2 52.5; oracle solves per dollar of 3 rollouts: native 0.73, cliff45 0.76, eoq1 0.77.

Price regimes (cost per task relative to native, same trajectories re-priced): cliff45 −7% to −9% under all four
sheets; cliff32 +16–19% where reads are cheap (0.1×) and −1% where reads are dear (0.5×); eoq1 +1% to +6%. The EOQ
trigger the rule derives moves 34K (0.5× reads) → 41K (0.1× reads).

Action mix (calls per task): compaction arms spend more on exploration and re-reads and less on edits/tests — native 18.7
explore / 5.4 edit / 6.7 test / 13.4 re-reads; cliff32 24.2 / 3.7 / 5.1 / 18.1; eoq1 24.6 / 5.2 / 5.4 / 18.7.

## Reading
1. **A dose–response curve with an optimum.** 32K < native < 45K ≈ 40K ≈ EOQ(~42K). Below the optimum, cheaper calls are
   paid back in extra calls and lost solves; the harness's ~18K fixed prompt sets the floor (the paper's own §8 caveat).
2. **At the optimum, compaction buys solves rather than dollars**: +3 to +6 points at equal or lower cost per task; cost per
   solved task falls ~10–15%. Direction matches the paper's Claude Code row (Table 2) on a different benchmark, model and pool.
3. **EOQ lands in the optimum without tuning** — derived from the price sheet — and is statistically indistinguishable
   from the best fixed threshold (cliff40: +5.0 vs +5.8). Its per-trajectory variation shows up only as a modest diversity
   gain across rollouts (pass@3 +5 points, +5% solves per dollar), not in pass@1. The claim is "no tuning, transfers", not
   "beats a tuned threshold" — the terminus-2 matrix (running) tests the transfer half.
4. **EOQ v2 is worse** because its cycle-mean growth estimate (~780/call) sits well below the whole-history mean (~1250)
   that early file reads dominate, firing ~10K early. v3 keeps v1's growth prior, measures L0, and caps the estimate
   below the window (running on both harnesses).
5. Where the extra calls go: re-reading sources. Consistent with the paper's precision-over-recall account.

## Caveats
1. Cliff's chars/4 estimate runs ~30% below real tokens here; EOQ's 52K-est ceiling let one eoq1 trajectory (1 of 360)
   hit the 64K wall (counted as failed). v3 uses a 46K ceiling.
2. astropy-14539 returns `RewardFileNotFoundError` in some trials (agent breaks the verifier env): scored as failed in
   paired comparisons, excluded from the solved denominator. astropy-14598 (native) looped until Claude Code's
   `rapid_refill_breaker`. Agent outcomes, not infrastructure; nothing was rerun selectively.
3. The native trigger (~50K real) is slightly above the paper's ~45K matched point.
4. 120 pairs give ±3.5–4 points on paired differences; eoq1 vs native ≈ 1.5 se, cliff40 ≈ 1.5 se, cliff45 ≈ 0.8 se.
5. One provider, one model so far; price transfer is by re-pricing, not real runs. Running: terminus-2 matrix (native /
   Cliff 16K / Cliff 32K / EOQ v3, 40 tasks, 3 seeds), EOQ v3 and v3f (failure-triggered) on Claude Code, 397B
   replication (native / Cliff 45K / EOQ v3), Terminal-Bench 2 long-horizon (12 tasks, 2 seeds). Ledger $348 of $700.
