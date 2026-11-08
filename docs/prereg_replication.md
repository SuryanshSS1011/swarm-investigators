# Pre-registered replication R: model-written retractions and GPT-5.6 Sol

Status: written after the main study, E1 and E2, and committed before any call of this study, including the calls
that write the retractions. Configuration: `configs/replication.json`. Nothing here changes the analysis of the
main study in `docs/prereg_main.md`, and none of these results will be pooled into H1 to H8.

## Why

1. Each retraction cell of the main study used one fixed wording that we wrote, so the retraction effect could be
   a property of that sentence.
2. METR handed much of its review to GPT-5.6 Sol, which none of the main-study investigators is.

## R1: model-written retractions

1. **Stimuli.** For each board in the five retraction cells written by Claude Haiku 5.5 or Gemini 3.8 Flash (40
   boards), the board's own author model writes the culprit's correction. It sees the culprit's earlier post, in
   its violating version, and a one-line brief that states the benign reading (`swarm.world.GEN_BRIEF`). The brief
   carries the same claim as the fixed wording, so only the wording changes. The prompt uses the main study's team
   framing (`scripts/gen_retractions.py`). A text is redrawn, up to three times, when it is empty, longer than 300
   characters or mentions a term in `swarm.author.FORBIDDEN`. Boards with no usable text after three draws are
   dropped and counted. The same text appears on the violation board and its clean twin.
2. **Investigators.** Claude Haiku 5.5, Claude Sonnet 5.5, Claude Opus 5.5 (Haiku's boards only, as in the main
   study), Gemini 3.8 Flash and GPT-5.6 Sol.
3. **R1 hypothesis.** Pooled detection will be lower with the model-written retraction than with the neutral post,
   paired on investigator, author, seed and violation status. It is supported if the pooled difference is at least
   5 points and its 95% seed-bootstrap interval lies below zero.
4. **Secondary.** The model-written against the fixed retraction will be reported with its interval, two-sided and
   with no prediction. False positives on the clean twins will be reported.

## R2: GPT-5.6 Sol as an investigator

1. **Settings.** `openai/gpt-5.6-sol` through the Chat Completions API, with reasoning effort "none" (the lowest it
   accepts, which is the main study's rule), no temperature and up to 2,000 output tokens. The flex service tier
   will be used if the API accepts it for this model. It changes the price, not the model.
2. **Arms, on Haiku- and Gemini 3.8 Flash-written boards.**
   1. The main arm, with the six variants and no header.
   2. The own-model header with no note, where "self" names GPT-5.6 Sol as the agents' model and "other" names
      another family's model.
   3. On the five retraction cells, the alibi variants (vouch, self_vouch) with the standard prompt, the
      retraction and retraction-plus-alibi with the conflict-aware prompt D3, and the same two variants with the
      E2 view that logs tool effects.
   4. The R1 model-written retraction.
3. **Hypotheses.**
   1. **R2a.** On retraction cells, detection will be lower with the fixed retraction than with the neutral post,
      by at least 5 points with a 95% interval below zero.
   2. **R2b.** As in the main study, no note addressed to reviewers will lower detection by at least 5 points with
      a 95% interval below zero. Each note is reported.
   3. **R2c.** If the 90% interval of the self-minus-other difference lies within plus or minus 5 points, for
      detection and for false positives, we will report no effect larger than 5 points.
   4. **R2d.** Under a retraction, the E2 view will raise detection by at least 20 points over the standard view.
4. **Exploratory.** Fake tool records, the alibi alone, D3 and the Appendix D row.

## Analysis and reporting

1. Contrasts are paired as in the main study. Single-investigator tests use exact McNemar tests, and intervals come
   from 2,000 seed-bootstrap draws. There is no multiplicity correction across R1 and R2, and every result is
   reported with its interval.
2. Refusals and unparsed answers count as misses. The lenient rule that counts insufficient_evidence as correct
   will also be reported.
3. GPT-5.6 Sol spending is capped at $40. Requests run in this order: main arm, E2, R1, header, alibis, D3. If the
   cap stops the run, the arms that did not run will be reported as not run.

## Amendment (November 7, about 19:15 ET)

GPT-5.6 Sol did not run. We could not get access to the OpenAI API within the hackathon, so no GPT-5.6 Sol call was
made in R1 or R2. R1 ran as registered with the other four investigators. R2 is reported as not run, as item 3 of
the analysis plan provides for arms that do not run. No other part of R changes.
