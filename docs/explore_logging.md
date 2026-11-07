# Exploratory follow-up E2: logging what agents did

Status: exploratory. It was written after the main study and E1 showed that retractions and peer alibis work in
cells where the tool records cannot separate a violation from its benign twin, and before any E2 model call.
Configuration: configs/explore_logging.json.

## Question

If the platform logs what each tool call actually did, will retractions and alibis stop working? This tests the
main recommendation of the report directly instead of asserting it.

## Design

1. **Cells and boards.** E2 uses the same five retraction cells and 20 seeds as E1, with Haiku- and
   Gemini-authored boards, in violation and clean versions.
2. **Effects view.** The first planted tool record of each board gains a bracketed platform note saying what the
   call did, such as "[platform: script wrote /eval/scorer/limits.yaml]" on the violation board and "[platform:
   script wrote /workspace/ag-52/limits.yaml]" on its twin. Everything else is unchanged.
3. **Variants.** The variants are neutral, spoof (the retraction) and self_vouch (a retraction and a peer
   alibi), with the main-study prompt.
4. **Investigators.** Haiku and Gemini 3.8 Flash audit both authors' boards. Sonnet audits Haiku's boards, and
   Gemini 4 Argon audits Gemini's boards with spoof and self_vouch only.

## Comparisons

Each variant in the effects view will be compared with the same variant in the main study or E1 on the same
boards, for detection on violation boards and the false-positive rate on clean boards. Results are exploratory,
with paired differences and exact McNemar tests and no multiplicity correction.

## Extension (November 7, about 09:50 ET)

After the first results, the Anthropic budget turned out to be $100 rather than $40, so Opus was extended to every variant on Haiku's boards and Sonnet to Gemini's boards as well. The open-weight investigators were added in the same Modal session as the main study. Nothing else changed.
