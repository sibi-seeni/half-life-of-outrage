Of the 321,550 labels, 181,165 came from an earlier run with the same model and prompt. Relabeling a random 500 of them with the final pipeline gave κ = 0.96 (95% CI 0.94–0.98) on the full label. The negative share among game comments changed by 0.5 points.

## Collection Re-Run

Comment collection was rerun on Oct 21–22, 2026. During analysis I found that every collection run started from the first video again and used up the daily quota re-reading videos it already had. As a result, 3,633 of the 5,823 discovered videos never had their comments fetched, including all of Jan–Oct 2025 and the launch. The collector now marks each video once all its comment pages are fetched and skips it on later runs (commit 901409e).

After the rerun, comments have been fetched for 5,720 of the 5,823 videos (the rest have comments disabled). The database holds 959,647 comments from 4,908 videos, and `comments_clean.csv` grew from 321,550 to 694,800 rows. Most of the new rows are from the launch period (Oct 2025 – Mar 2026, about 543k rows).

The rebuilt dataset was relabeled with the same model (llama-3.3-70b-instruct) and prompt (`absa_v2`), with context. `labels_llm.csv` has a valid label for 694,798 of the 694,800 comments. The two without a label have no real text ("[USER] my comment", "Comment was made"), so the model asked for the comment instead of labeling it. 180,734 of the labels were reused from the earlier run (`prompt_sha256 = legacy`). The κ check above was done on the old 321,550-comment dataset. 471,095 comments are game-related: 46.2% Neg, 35.9% Pos, 16.2% Neu and 1.6% Ambivalent.

During relabeling, two API keys ran out of budget, one was revoked, and a new key at first had no model access. NaviGator reports a spent budget as 422 and missing model access as 400, but the labeler only dropped a key on 401/403. As a result about half of the requests failed, and comments were saved as errors (2,245 still failed at the worst point). The labeler now drops keys that can no longer be used (commit 70e6c71), and the failed comments were relabeled.

`labels_roberta.csv` was regenerated on HiPerGator (B200, about 6 min) for all 694,800 comments: 40.9% Neg, 32.6% Neu, 26.6% Pos. On the 470,059 game comments both models labeled, RoBERTa and the LLM agree on sentiment 65.6% of the time.