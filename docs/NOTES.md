Of the 321,550 labels, 181,165 came from an earlier run with the same model and prompt. Relabeling a random 500 of them with the final pipeline gave κ = 0.96 (95% CI 0.94–0.98) on the full label. The negative share among game comments changed by 0.5 points.

## Collection Re-Run

Comment collection was rerun on Oct 21–22, 2026. During analysis I found that every collection run started from the first video again and used up the daily quota re-reading videos it already had. As a result, 3,633 of the 5,823 discovered videos never had their comments fetched, including all of Jan–Oct 2025 and the launch. The collector now marks each video once all its comment pages are fetched and skips it on later runs (commit 901409e).

After the rerun, comments have been fetched for 5,720 of the 5,823 videos (the rest have comments disabled). The database holds 959,647 comments from 4,908 videos, and `comments_clean.csv` grew from 321,550 to 694,800 rows. Most of the new rows are from the launch period (Oct 2025 – Mar 2026, about 543k rows).

The rebuilt dataset is being relabeled with the same model (llama-3.3-70b-instruct) and prompt (`absa_v2`). As of Oct 28, `labels_llm.csv` covers 397,925 of the 694,800 comments. `labels_roberta.csv` still covers only the old 321,550 rows and has to be regenerated.