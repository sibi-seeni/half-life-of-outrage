# Annotation guidelines

You label each comment **without seeing any model output**. Use the video title, the description excerpt and the parent comment only to understand what the comment refers to.

## 1. `gold_game_related`: is the opinion about Arc Raiders itself?
- **Yes**: the target is the game: gameplay, design, visuals, bugs, pricing or business model, AI/TTS voices *in Arc Raiders*.
- **No**: the target is the creator or streamer (their skill, voice, editing), another game, AI in general, or anything unrelated.

If **No**, leave category and sentiment empty.

## 2. `gold_category` (only if Yes)
- **AI_Voice_Related**: AI or text-to-speech voices in Arc Raiders, voice actors, ethics of that choice.
- **Business_Model_Related**: price, free-to-play vs paid, monetisation, cosmetics or battle pass, the PvE→PvPvE change.
- **Game_Related**: everything else about the game.

If a comment touches several aspects, pick the one carrying the strongest opinion.

## 3. `gold_sentiment` (only if Yes)
- **Pos**: praise, excitement.
- **Neg**: criticism, disappointment, mockery.
- **Neu**: factual statements and questions with no evaluative stance.
- **Ambivalent**: clear praise *and* clear criticism.

Sarcasm takes its intended meaning ("great, another delay" → Neg).

## Process
1. Annotate independently, and don't discuss items until both of you have finished.
2. Record each annotator's time spent and any comments that need outside knowledge (a `notes` column is provided).
3. After agreement is computed, the two annotators adjudicate the overlap disagreements. The adjudicated label becomes the gold label, and the pre-adjudication labels are kept for reporting.
