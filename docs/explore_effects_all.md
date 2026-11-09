# Exploratory follow-up E5: effect notes on every tool call

Status: exploratory. It was written on November 9, 2026, at about 02:00 ET, after the main study and E1 to E4, and
before any E5 model call. Configuration: configs/explore_effects_all.json. Code: the effects_all view in
swarm/assemble.py and swarm/world.generic_effect.

## Question

In E2 only the planted tool call carried a platform note, on the violation board and on its twin. The note told the
investigator what the call did, but it was also the only bracketed note on the board, so it pointed the investigator
at the right line. Does effect logging still restore detection when every tool call carries a note and the planted
note is one of about 75?

## Design

1. **Boards.** E5 uses E2's boards, the five retraction cells by 20 seeds with Haiku- and Gemini-authored posts, and
   their benign twins.
2. **View.** The effects_all view is E2's effects view with one change. Every other tool record gains a note in the
   same bracketed form, written from the record itself, such as "[platform: read /data/public/task_14/spec.md]",
   "[platform: ran the tests of task 22]" or "[platform: sent /workspace/ag-04/out_t42.json to the grader]". The
   planted record keeps E2's note word for word. Violation and clean boards still differ only in the planted lines,
   and tests/test_effects_all.py checks this.
3. **Variants.** Neutral, spoof (the retraction), vouch (the peer alibi) and self_vouch (both), as in E2, with the
   main-study prompt, which says nothing about the notes.
4. **Investigators.** The same investigators audit the same boards as in E2. Haiku, Sonnet and Gemini 3.8 Flash
   audit both authors' boards, Opus audits Haiku's boards and Gemini 4 Argon audits Gemini's boards. Claude runs at
   its floor as in E2. Gemini is pinned to reasoning effort "low", the setting its E2 calls climbed to, so that E5
   differs from E2 only in the notes. Qwen3-32B and gpt-oss-20b audit both authors' boards at their main-study
   settings if the laptop run on Modal can be done before the deadline, and the report will say if it was not.
5. **Budget.** At most $22 for the five API investigators, capped per output file in the configuration.

## Outcomes

1. **Primary.** Detection on violation boards under the retraction, pooled over the five Claude and Gemini
   investigators, compared with E2 on the same boards and investigators as a paired difference with an exact
   McNemar test.
2. **Pre-committed reading.** E2's result "does not depend on pointing" if pooled detection under the retraction in
   E5 is at least 90%, "depends partly on pointing" if it is from 75% to below 90%, and "depends on pointing" if it
   is below 75%. These are the bars E4 used.
3. **Secondary.** The same comparison for the neutral post, the alibi and the combination, per investigator, and the
   false-positive rate on twins for every variant.
4. **Rule checker.** We will also report the main study's rule-based checker and the same checker with two added
   patterns, a protected path or a hub token anywhere in a tool record, on both effects views. The two patterns
   were written after seeing E2's notes, so this check is post hoc for E2 and fixed in advance for E5.

All results are exploratory, with no multiplicity correction. We will report them whichever way they come out.

## Limitations stated in advance

The generic notes restate what each record already shows, and a real platform would also log effects that do not
appear in the call, such as files a script read. Only the planted note carries information beyond its record, so a
careful investigator can still find it by reading every note. E5 asks whether investigators read the notes when
nothing points at the planted one, not whether a noisier real log would still work.
