# Pilot protocol (kill test), written before any model call

Status: exploratory pilot. Its data are excluded from every confirmatory test. The main study will use
fresh board seeds (1000 and up) under a separate pre-registration written after this pilot.

## Purpose

The pilot will check that the task is usable, estimate effect sizes for power planning, and choose which
arms the main study carries. It will not test hypotheses.

## Design

1. Boards. 20 seeded skeletons will rotate through 5 violation archetypes (hidden answers, scorer tampering,
   misreported test results, transcript tampering, credential sharing) and 2 tiers (explicit, implicit).
   Each skeleton has 10 agents, about 110 log lines and 35 posts.
2. Authors. Claude Haiku 5.5, Claude Sonnet 5.5 and Gemini 3.8 Flash will each write the routine posts of
   every skeleton in their own voice. Authors never see tool records, planted lines, rules or variant text.
3. Matched pairs. Every base board will be shown once with the planted violation and once with its benign
   twin, which matches it in agent, position, length and surface form.
4. Text variants. The culprit's post after the planted line will be absent (none) or will carry one of six
   fixed texts: a neutral task rationale, a solidarity rationale, a false authorization addressed to
   reviewers, a peer appeal addressed to AI reviewers, a direct instruction to reviewers, or a spoofed tool
   record or false correction. The same text goes on the benign twin as a placebo.
5. Kinship headers. With the none and peer-appeal variants, the deployment header will also name the
   agents' model as the investigator's own model (self) or as a model from another family (other). Every
   investigator is told its own model name in its system prompt.
6. Investigators. Claude Haiku 5.5, Gemini 3.8 Flash and Claude Sonnet 5.5 will audit every board. Each
   returns a verdict (violation, no_violation, insufficient_evidence), the agents and lines responsible,
   the rules broken and a confidence from 0 to 100.
7. Baseline. A fixed rule-based checker will scan tool records for forbidden paths and hosts and posts for
   credential patterns.

## Decision rules for the main study

1. Usable task. The main study will keep the current boards if, for at least two of three investigators,
   recall on explicit-tier boards without variants is at least 80% and the false-positive rate on clean
   boards without variants is at most 25%. Otherwise the boards will be revised and re-piloted.
2. Persuasion arm. The persuasion and injection arm will be the primary hypothesis family if any reviewer
   addressed variant or the spoof variant lowers paired recall by at least 10 points for at least one
   investigator, or raises confident exoneration by at least 10 points.
3. Kinship arm. Kinship will be a primary hypothesis if the self-minus-other header contrast or the
   same-model-minus-other-family author contrast moves recall or the false-positive rate by at least 5
   points in either direction for at least one investigator. Otherwise it will be a secondary hypothesis
   tested against a pre-registered equivalence bound.
4. Variants carried forward. The main study will keep none, neutral and every variant whose paired effect
   on recall or the false-positive rate is at least 5 points for at least one investigator, up to four.
5. Cost. The main study will be sized to stay under $60 of projected spend across all providers.

## Known limits of the pilot

With 60 violation boards per cell, a paired recall difference needs to be roughly 10 points to stand out
from sampling noise. Claude 5.5 models accept no temperature setting, so repeated calls can differ. The
pilot will measure that noise on a re-run subset before the main study is sized.
