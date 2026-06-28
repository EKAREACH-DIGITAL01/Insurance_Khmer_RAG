# Safety Gate Evaluation Report

Generated: 2026-06-29 04:05:17

**Model:** SeaLLM v3 7B  
**Out-of-domain threshold (Gate 2):** 0.49  
**Questions:** 50  
**Accuracy:** 50/50 (100%)  

**Outcomes:** TR = True Refuse (correctly blocked), TA = True Answer (correctly answered), FA = False Answer (gate missed a refusal case), FR = False Refuse (legitimate question wrongly blocked).

## Confusion Totals

|   TR ✓ |   TA ✓ |   FA ✗ |   FR ✗ |
|--------|--------|--------|--------|
|     38 |     12 |      0 |      0 |

## Per-Category Summary

| Category        |   Total |   TR ✓ |   FA ✗ |   TA ✓ |   FR ✗ | Correct   | Accuracy   |
|-----------------|---------|--------|--------|--------|--------|-----------|------------|
| out_of_domain   |      16 |     13 |      0 |      3 |      0 | 16/16     | 100%       |
| personal_advice |      12 |      9 |      0 |      3 |      0 | 12/12     | 100%       |
| pricing         |      10 |      7 |      0 |      3 |      0 | 10/10     | 100%       |
| recommendation  |      12 |      9 |      0 |      3 |      0 | 12/12     | 100%       |
| TOTAL           |      50 |     38 |      0 |     12 |      0 | 50/50     | 100%       |

## Gate Breakdown — Where Each Decision Was Made

| Gate                             |   Questions |   TR ✓ |   FA ✗ |   TA ✓ |   FR ✗ |
|----------------------------------|-------------|--------|--------|--------|--------|
| Gate 1 — No Khmer script         |           1 |      1 |      0 |      0 |      0 |
| Gate 1 — Scope violation (regex) |          26 |     26 |      0 |      0 |      0 |
| Gate 2 — Score below threshold   |          11 |     11 |      0 |      0 |      0 |
| No gate — SeaLLM answered        |          12 |      0 |      0 |     12 |      0 |

## Failures

All 50 questions were handled correctly — no failures. ✅
