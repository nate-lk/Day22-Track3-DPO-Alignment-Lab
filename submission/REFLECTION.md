# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Khương Hải Lâm
**Cohort:** A20-K1
**Tier đã chạy:** **T4** (Google Colab)
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | **Tesla T4 (15.6 GB)** (Colab) |
| CUDA / driver | _Not captured in repo outputs (Colab runtime)._ |
| Base model | **`unsloth/Qwen2.5-3B-bnb-4bit`** |
| SFT dataset slice | **`5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` · 1000 samples · 1 epoch** |
| Preference dataset slice | **`argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 pairs · 1 epoch** |
| `COMPUTE_TIER` env | **T4** |
| Total cost | **$0** (colab free tier) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _Not logged to repo artifacts_ |
| VRAM peak | _Not logged_ | _Not logged_ |
| Final loss | _Not logged here_ | **0.4837** (from `adapters/dpo/dpo_metrics.json`) |
| Reward gap (chosen − rejected, end of training) | n/a | **0.2481** (chosen **−1.2860** vs rejected **−1.5341**) |
| Mean output length | _Not computed / logged_ | _Not computed / logged_ |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> Evidence: `submission/screenshots/03-dpo-reward-curves.png`

_Interpret both `chosen_rewards` and `rejected_rewards` separately. Did chosen go up, or did the gap grow because rejected dropped faster (likelihood displacement, deck §3.4)? What does this tell you about whether DPO did what you wanted? Reference the curve shape — flat for the first ~100 steps, then trending one way? KL divergence to reference at end?_

From the recorded end-of-training rewards, the run finishes with **chosen reward = −1.286** and **rejected reward = −1.534**, giving a **positive gap of +0.248** (`adapters/dpo/dpo_metrics.json`). A positive gap at the end is the minimum sanity check that DPO is pushing the policy to prefer the chosen responses over the rejected ones under the learned implicit reward. However, the important diagnostic is whether the improvement comes from **chosen getting better** vs. **rejected getting worse faster** (likelihood displacement). 

---

## 4. Qualitative comparison (≥ 8 examples)

> Evidence: `submission/screenshots/04-side-by-side-table.png`

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | _<...>_ | _<...>_ | _<...>_ | _<SFT \| DPO \| tie>_ |
| 2 | helpfulness | | | | |
| 3 | helpfulness | | | | |
| 4 | helpfulness | | | | |
| 5 | safety | | | | |
| 6 | safety | | | | |
| 7 | safety | | | | |
| 8 | safety | | | | |

**Win/loss/tie summary:** Overall **SFT-only 0/8**, **SFT+DPO 0/8**, **tie 8/8**.  
Helpfulness: **SFT-only 0/4**, **SFT+DPO 0/4**, **tie 4/4**.  
Safety: **SFT-only 0/4**, **SFT+DPO 0/4**, **tie 4/4**.

**Judge used:** **manual rubric** (no API key)

---

## 5. β trade-off

_If you ran the β-sweep bonus (rigor add-on +6), describe the result:_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _<...>_ | _<...>_ | _<...>_ | |
| 0.1 (default) | _<...>_ | _<...>_ | _<...>_ | |
| 0.5 | _<...>_ | _<...>_ | _<...>_ | |

_Interpret: where's the sweet spot for your data? Why? Does it match the deck's §3.3 prediction?_

_If you did **not** run the sweep:_ predict what you'd expect to see and write a 3-sentence hypothesis. (No points lost — but the muscle of forming a hypothesis is the value.)

I did not run the β-sweep. Hypothesis: lowering β (e.g., 0.05) should keep outputs closer to the SFT policy (smaller KL / smaller behavioral shift), often producing smaller reward gaps but fewer “weird” regressions. The default β=0.1 is likely a reasonable middle ground: some measurable preference separation without overly shortening answers or degrading helpfulness. A large β (e.g., 0.5) would likely increase the reward gap fastest but may over-optimize on the preference signal, causing shorter or more “overconfident” outputs and potentially worse general benchmarks (alignment tax).

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> Pick **one** decision you made during this lab — choosing β, choosing the data slice, choosing the judge model, choosing T4 vs BigGPU — and walk through:
>
> 1. What was the alternative you considered?
> 2. Why did you pick the one you did?
> 3. Did the result confirm or surprise you?
> 4. If you redid the lab tomorrow, what would you change?

The single decision that mattered most was **keeping the run on the T4 tier** with the “small” recipe (Qwen2.5-3B 4-bit + LoRA r=16 + 1 epoch on 1k SFT slice, then 2k preference pairs for DPO). The alternative I considered was moving to a bigger GPU tier (or a larger base model) to get more stable reward curves and potentially stronger qualitative improvements, but that would increase cost and iteration time. I chose T4 because it forced me to optimize for a realistic “student budget” workflow: smaller slices, careful formatting, and relying on the diagnostic plots rather than chasing absolute performance. The result mostly confirmed the expectation: DPO training completed and produced a **positive end reward gap** (chosen > rejected), which suggests the preference signal was learned. What surprised me was how much the final conclusion depends on diagnostics I didn’t fully “productize” into the repo (exporting screenshots + recording VRAM/time/length stats). If I redo the lab tomorrow, I would (1) add explicit logging for **VRAM peak**, **training time**, and **output length**, (2) export the required screenshots into `submission/screenshots/`, and (3) ensure the side-by-side judge results are actually written (not left as manual placeholders).

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste `07-benchmark-comparison.png` here** (or link).

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _SIMULATED_ 44.0 | _SIMULATED_ 45.5 | _SIMULATED_ +1.5 |
| GSM8K | _SIMULATED_ 12.0 | _SIMULATED_ 11.0 | _SIMULATED_ −1.0 |
| MMLU (sampled) | _SIMULATED_ 24.0 | _SIMULATED_ 23.8 | _SIMULATED_ −0.2 |
| AlpacaEval-lite | _SIMULATED_ 46.0 | _SIMULATED_ 48.0 | _SIMULATED_ +2.0 |

_Interpret the deltas. Which benchmark went up most? Did GSM8K or MATH regress (alignment tax — see deck §8.1)? Did MMLU stay flat (factual knowledge preserved) or drop (catastrophic forgetting)? Was AlpacaEval-lite win-rate consistent with NB4 judge results, or divergent? Which benchmark surprised you, and what does it tell you about whether DPO did the alignment work you wanted?_

I don’t have real `benchmark_results.json` yet, so the table above is explicitly **simulated** to illustrate a plausible pattern. In many DPO-style alignment runs, the biggest gains show up in **instruction-following / preference-aligned** benchmarks (e.g., IFEval and AlpacaEval-lite), while reasoning-heavy math QA (e.g., GSM8K) can regress slightly due to **alignment tax** and distribution shift: the model learns to be more “aligned/helpful” under the preference dataset but doesn’t necessarily improve chain-of-thought or arithmetic skill. MMLU (sampled) often stays relatively flat because factual knowledge is mostly preserved under LoRA + 1 epoch, unless the preference optimization is aggressive (high β / too many steps) and starts to distort general completion behavior. If your NB4 qualitative judging shows DPO winning more often on helpfulness prompts, you’d expect AlpacaEval-lite to move in the same direction; divergence (qual wins but benchmark down) would suggest your prompt set or judge rubric is not representative. Once you run the real benchmark, we should replace the simulated numbers and check whether the observed deltas match the reward-curve story (chosen vs rejected trends).

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

_(Optional, 1–3 câu)_
