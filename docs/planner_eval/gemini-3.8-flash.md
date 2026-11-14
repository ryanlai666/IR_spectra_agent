# Planner accuracy (gemini-3.8-flash, 20 prompts)

- Configured model: gemini-3.8-flash (fallbacks: gemini-3.6-flash, gemini-3.5-flash-lite)
- Answered by: gemini-3.5-flash-lite 10, gemini-3.6-flash 10; 20 prompts saw API errors before an answer
- Exact plan match: 100%
- Valid plan after the validator/retry loop: 100%
- Valid without a validator rejection: 100% (0 rejections in total)
- Mean latency: 19.8 s

| field | accuracy | n |
|---|---:|---:|
| adsorbate | 100% | 20 |
| calculator | 100% | 19 |
| metal | 100% | 20 |
| structure | 100% | 4 |
| facet | 100% | 13 |
| site | 100% | 3 |
| model | 100% | 1 |
| dipole_method | 100% | 1 |

## Misses

None.
