# Exploratory follow-up E3: investigators at high reasoning

Status: exploratory. It was written after the main study, E1, E2 and replication R1, and before any E3 model call.
Configuration: `configs/explore_reasoning.json`.

## Question

Every investigator in the main study ran at the lowest reasoning setting it accepts. Will more reasoning stop a
retraction from working? Opus 5.5 always thinks, and at low effort it still lost at least 18 points to the
retraction, but no investigator has been tested above its floor.

## Design

1. **Cells and boards.** E3 uses the five retraction cells of the main study, with the boards written by Claude
   Haiku 5.5 and Gemini 3.8 Flash, in violation and clean versions.
2. **Variants.** The neutral post and the fixed retraction (spoof), with the main-study prompt, the full view and
   no header.
3. **Investigators.** Claude Haiku 5.5 and Claude Sonnet 5.5 at effort "high" with adaptive thinking, Gemini 3.8
   Flash at reasoning_effort "high", and Claude Opus 5.5 at effort "high" on Haiku's boards only, as in the main
   study. Each gets up to 16,000 output tokens. They are run as `<model>@high` (see `swarm/llm.py`), so the
   prompt and the investigator's stated identity are the same as at the floor.
4. **Floor comparison.** Each high-reasoning audit is paired with the same model's main-arm audit of the same
   board and variant at the floor.

## Hypotheses

1. **E3a.** Pooled over the four investigators at high reasoning, detection on retraction cells will be lower with
   the retraction than with the neutral post, by at least 5 points with a 95% seed-bootstrap interval below zero.
2. **E3b.** The retraction effect at high reasoning minus the retraction effect at the floor, paired on model,
   author and seed, will be reported with its 95% interval. It is two-sided, with no prediction.

## Analysis and reporting

1. Contrasts are paired on (model, author, seed, violating). Single-investigator tests are exact McNemar tests,
   intervals come from 2,000 seed-bootstrap draws, and there is no multiplicity correction.
2. Detection with the neutral post at high reasoning against the floor, false positives on the clean twins, and
   the lenient rule that counts insufficient_evidence as correct will also be reported, per investigator and
   pooled.
3. Refusals, unparsed answers and answers cut off at the token limit count as misses, and their numbers will be
   reported.
4. Spending is capped at $12 for the Claude models, through the Message Batches API, and $5 for Gemini. Requests
   run in this order: Haiku, Sonnet, Gemini, Opus. If a cap stops a run, the audits that did not run will be
   reported as not run, and the contrasts will use the boards that finished.
5. None of these results will be pooled into the main study's hypotheses.

## Deviation (November 7, about 13:20 ET)

Gemini 3.8 Flash at reasoning_effort "high" averaged about 12,200 output tokens per audit (corrected from 12,600,
which was the mean of the first 17 audits). 4 of its first 18
answers stopped at the 16,000-token limit, and each audit cost about $0.05, so the $5 cap would have stopped it near
100 of 160 audits with about a quarter of the answers cut off. The Gemini run was stopped after 18 audits and $0.90,
before any E3 verdict was looked at, and the user chose to report it as stopped rather than change its settings.
E3a is therefore pooled over the three Claude investigators, and the 18 Gemini audits are reported but not analyzed.

## Amendment (November 7, about 14:05 ET)

The user approved a larger Gemini budget, so Gemini 3.8 Flash at reasoning_effort "high" is rerun on all 160 boards
with up to 32,000 output tokens and a $12 cap. Nothing else changes. The 18 stopped audits at 16,000 tokens are kept
in `results/main/raw/` under a separate name and are not analyzed. Before this amendment, one aspect of the stopped
run had been seen: Gemini flagged all 6 neutral-post violation boards it answered. No verdict on a retraction board
had been looked at. E3a is again pooled over the four investigators as first registered, and the three Claude
investigators are also reported on their own, as they were before the rerun.

## Extension (November 7, about 14:20 ET)

Committed before any call of the extension. Qwen3-32B with thinking on, its only setting above the floor, and
gpt-oss-20b at reasoning_effort "high" audit the same 160 boards with up to 24,000 output tokens. The Modal
servers' context length rises from 16,384 to 32,768 tokens to make room, which does not change shorter requests.
Each model is compared with its own main-arm audits of the same boards at the floor. Both are reported per
investigator and pooled with each other, outside the E3a pool of the four registered investigators. They run from
the laptop through `scripts/run_openweight_e3.sh`, like the main study's open-weight half, and Modal bills GPU time
rather than tokens, so there is no token cap.
