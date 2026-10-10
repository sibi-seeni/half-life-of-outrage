# Changes from the course-project code

This repo restructures the CAP5771 / `arc-raiders-trend` notebooks into a reproducible package. These are the substantive changes, so the paper's methods match what the code actually does.

## Bugs fixed
| Old behaviour | New behaviour | Effect on results |
|---|---|---|
| Half-life fitted on **daily comment volume**, from the global volume peak (Game 2026-03-06, Business 2026-03-09, a week before the data ended) | Fitted on **excess negative share above a 30-day pre-event baseline**, from the peak within 3 days of a dated event. Fixed horizon, min-data rules, bootstrap CI and a model-free half-life. Volume decay is reported separately as attention half-life | The old t½ values (21.0 / 0.5 / 1.8 d) and the "Developer Intelligence Taxonomy" should not be reported |
| Labeler read `parent_context`, but the wrangled CSV column is `parent_text`, so **parent comments were never sent to the LLM** | `build_user_message` reads `parent_text` | Replies (~22% of rows) should be relabeled |
| Event dates differed between notebooks (business-model change 2024-08-20 vs 2025-06-06; GenAI 2024-05-15 vs 2025-11-12) and some were after the data ended | One `config/events.csv` with status (verified / approximate / unverified) and a source per event; unverified events are excluded by default | Business-model window moves to Aug 2024 |
| Confidence retention divided by all rows, including non-game rows with no confidence (reported 65.6% at ≥0.70) | Denominator = game-related rows with a confidence value | Real retention is ~98% |
| Discovery reset its backfill cursor to "now" after each run, so older windows were never searched incrementally | The cursor moves back 90 days per run, as the original comment intended | More pre-2025 videos on re-collection |
| `commentThreads` returns at most 5 replies per thread, so the rest were silently dropped | Threads with more replies are paged through `comments.list` | More replies on busy threads |
| Spike z-scores were computed over calendar days with no comments skipped | Days are zero-filled before the rolling z-score | Slightly different spike days pre-launch |
| `langdetect` unseeded (non-deterministic) | Seeded | Reproducible language filter |
| `@\w+` mention mask left fragments (`[USER]-cw9nj`) | `@[\w.-]+` | Cleaner inputs |
| Parent text looked up only among filtered comments | Looked up from the raw table | Replies keep their context even when the parent is short or non-English |

## Design notes surfaced during the port
- The final prompt (`absa_v2`) first gates on target (Game / Creator / Other) and only assigns aspect and sentiment to game comments. About one third of comments are not about the game; this is a finding, not missing data, and the paper should report it.
- The original gold set was stratified by the **LLM's own predicted category** and labeled by one annotator. `hlo.eval.gold` adds a random stratum (unbiased estimates) and a two-annotator overlap (human ceiling).
- Verbalized confidence is nearly constant (median 0.80, IQR 0.80–0.90). `results/label_summary.json` reports its distribution, which is the starting point of the second (NLP) paper.

## Security
The `arc-raiders-trend` `Llama_Labeler.ipynb` contained a hard-coded NaviGator API key in a code cell. Keys are now read only from `.env`. Revoke the old key; it remains in that repo's git history.
