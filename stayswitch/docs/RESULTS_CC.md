# Claude Code harness: compaction-timing results (interim, 2026-10-02 06:40)

Setup: Claude Code 2.1.286 inside SWE-bench Verified containers (Harbor), Anthropic `/v1/messages` → CliffCompaction
proxy → StaySwitch proxy → Qwen3.6-35B-A3B on Tinker (64K window; cached reads 0.2×, no write premium). 40 tasks
(`configs/swe_tasks_all.txt`: 19 original + 21 same-repo extension), seeds 1–3. Cliff text caps 300 (paper setting),
keep_recent 3, Claude Code's own auto-compaction off in every Cliff/EOQ arm (`DISABLE_AUTO_COMPACT=1`).
Costs: the CliffCompaction paper's perfect-cache rule (`scripts/perfect_cache_cost.py`); `report_seeds.py` builds the tables.

## Arms
| arm | trigger | what it is |
|---|---|---|
| native | Claude Code auto-compact at 25% of its assumed 200K (~50K) | the harness's own LLM summarisation |
| cliff45 / cliff40 / cliff32 | fixed, Cliff est-tokens | the paper's method at their 45K matched point and below |
| eoq1 | L* = L0 + sqrt(2gC/c_r), g = whole-history mean growth, L0 guessed | our rule, v1 (median trigger ~42K) |
| eoq2 | same, g = cycle-mean growth, L0 measured | v2 (median ~33K); the lower g lowers the trigger |

## Pooled over 3 seeds (78 trajectories per arm: 40 tasks seed 1, 19 tasks seeds 2–3) — extension seeds 2–3 and cliff40 still running
| arm | solved % (pooled ± se) | calls/task | $/task | mean ctx | compactions/task | vs native: Δsolved % | Δ$/task |
|---|---|---|---|---|---|---|---|
| native | 42.1 ± 5.7 | 43.9 | 0.268 | 27.6K | 0.9 | — | — |
| cliff45 | 47.4 ± 5.7 | 41.4 | 0.252 | 25.3K | 1.2 | +6.4 ± 4.2 | −0.012 ± 0.025 |
| cliff32 | 40.8 ± 5.6 | 48.2 | 0.314 | 21.9K | 3.6 | −1.3 ± 4.3 | +0.037 ± 0.036 |
| **eoq1** | **51.3 ± 5.7** | 49.7 | 0.279 | 25.2K | 1.9 | **+10.3 ± 4.3** | +0.016 ± 0.032 |
| eoq2 | 40.0 ± 5.7 | 44.8 | 0.261 | 24.8K | 2.0 | −2.6 ± 4.2 | +0.005 ± 0.038 |

Reading:
1. **Compaction near the right point buys solves, not dollars.** eoq1 and cliff45 solve more at the same cost per task;
   cost per *solved* task falls ~17% (native 0.64 → cliff45 0.53, eoq1 0.54). This matches the paper's own Claude Code
   row (Table 2: Cliff 76.7% vs native 71.0% at ~45K, $0.16 vs $0.14) — replicated on a different benchmark, model and pool.
2. **The cost/call × calls/task decomposition is visible in every arm.** cliff32 and eoq2 cut cost per call (−6–8%) and
   lose it to extra calls and lost solves. 32K is below Claude Code's floor here (≈18K fixed prompt).
3. **EOQ v1 is the best arm so far and needed no tuning**: it derived ~42K from the price sheet. Whether its +3.9 points over
   cliff45 is real is what the **cliff40 control** (running) decides: if fixed 40K matches eoq1, the claim is "the rule lands
   on the right fixed point"; if not, the per-trajectory variation (p10 30K, p90 52K) matters.
4. **EOQ v2 (measured L0, cycle-mean growth) is worse.** The measured L0 equalled v1's guess (20.7K vs 20.9K); the cycle-mean
   growth (~780/call) is well below the whole-history mean (~1250/call) because early file reads dominate growth, so v2
   fired ~10K too early. v3 should keep v1's growth prior and add a ceiling in *real* tokens (see caveat 1).
5. Price regimes (`scripts/price_regimes.py`): the arm ordering is stable under Tinker/Anthropic-like/DeepSeek-like/OpenAI-like
   cache pricing; the EOQ trigger the rule derives moves 34K → 41K as reads get cheaper, in the direction the model predicts.

## Caveats (keep in the paper)
1. Cliff's chars/4 estimate runs ~30% below real tokens on this content. EOQ's 52K-est ceiling let one eoq1_r2 trajectory
   (django-13406) reach 68.9K real tokens and die on the 64K wall (counted as failed; 1 of 274 trajectories). Fixed arms
   peak at 58K real. The ceiling must be set in calibrated real tokens.
2. astropy-14539 returns `RewardFileNotFoundError` in 4 of 15 trials (agent breaks the verifier environment); scored as
   failed in paired comparisons, excluded from the solved denominator. astropy-14598 (native) looped until Claude Code's
   `rapid_refill_breaker` tripped. Both are agent outcomes, not infrastructure.
3. The native baseline's trigger (~50K real) is slightly above the paper's ~45K matched point; a calibrated 22% run was not done.
4. n = 40 tasks × 3 seeds gives ±4 points on paired differences; eoq1 vs native is ~2.4 se, cliff45 vs native ~1.5 se.
5. e_hat (extra calls per compaction) is too noisy per arm to report; use the paired Δcalls instead.

## Running / next
- cliff40 control (3 seeds, 40 tasks); extension tasks seeds 2–3 for all arms. Ledger $272 of $500 before these (~$75 more).
- Proposed, not launched: replication on the 397B model (native / cliff45 / eoq1, 40 tasks, 1 seed, ~$100).
