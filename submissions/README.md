# Zindi Submission Candidates

Current public baseline from `zindi_submission_conditional_hybrid_plus_0_507600.csv`:

- Public score: `0.602576`
- Local validation: `0.507600578`

Next public probe candidates:

- `zindi_probe_reranker_top200_enguga_local_0_513599.csv`
  - Local validation: `0.513598851`
  - Change: keeps the public baseline, but uses the top-200 learned candidate reranker only for `Eng_Uga`.
  - Test rows changed: `741`, all `Eng_Uga`.
  - Recommended next submission because it improves the safer `Eng_Uga`-only reranker probe without changing more rows.

- `zindi_probe_reranker_top200_enguga_engeth_local_0_514662.csv`
  - Local validation: `0.514662088`
  - Change: top-200 learned candidate reranker for `Eng_Uga` and `Eng_Eth`.
  - Test rows changed: `800` (`741` `Eng_Uga`, `59` `Eng_Eth`).
  - Submit after the top-200 `Eng_Uga`-only probe only if the public score improves.

- `zindi_probe_reranker_enguga_local_0_512600.csv`
  - Local validation: `0.512599672`
  - Change: keeps the public baseline, but uses the learned candidate reranker only for `Eng_Uga`.
  - Test rows changed: `741`, all `Eng_Uga`.
  - Superseded by the top-200 `Eng_Uga`-only probe.

- `zindi_probe_reranker_enguga_engeth_local_0_514348.csv`
  - Local validation: `0.514348454`
  - Change: learned candidate reranker for `Eng_Uga` and `Eng_Eth`.
  - Test rows changed: `800` (`741` `Eng_Uga`, `59` `Eng_Eth`).
  - Superseded by the top-200 `Eng_Uga+Eng_Eth` probe.

Rejected public probes:

- `zindi_submission_nllb_amh_hybrid_rules_0_508261.csv`
  - Local validation: `0.508260973`
  - Public score: `0.600599`
  - Change: keeps NLLB Amharic query-translation candidate, then applies dynamic hybrid rules from `outputs/hybrid_rule_sweep_nllb_aka_enggha/results.csv`.
  - Rules: `Aka_Gha:pred_input:80+80:pred_words>120`, `Eng_Gha:input_pred:50+50:input_words>30`.
  - Result: worse than the current public baseline; do not continue NLLB Amharic plus these hybrid-rule changes.

- `zindi_submission_nllb_amh_query_0_508147.csv`
  - Local validation: `0.508147020`
  - Change: replaces only `Amh_Eth` rows with query-translated English retrieval answers translated back with NLLB.
  - Status: do not submit; superseded locally and the larger NLLB hybrid public probe was worse.

- `zindi_probe_akan_broader_eng80_local_0_507316.csv`
  - Public score: `0.601786`
  - Local validation: `0.507315548`
  - Result: worse than the current public baseline; do not continue this broader-Akan direction.

Do not submit `zindi_probe_akan_broader_eng60_local_0_507306.csv` unless a later experiment gives a specific reason.

`aya:8b` translation smoke tests were rejected. NLLB query translation for Amharic
improved local validation but failed to transfer to public scoring.

Keep the original current best available as:

- `zindi_submission_conditional_hybrid_plus_0_507600.csv`
