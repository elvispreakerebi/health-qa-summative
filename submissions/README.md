# Zindi Submission Candidates

Current public baseline from `zindi_submission_conditional_hybrid_plus_0_507600.csv`:

- Public score: `0.602576`
- Local validation: `0.507600578`

New local-best candidate:

- `zindi_submission_nllb_amh_hybrid_rules_0_508261.csv`
  - Local validation: `0.508260973`
  - Change: keeps NLLB Amharic query-translation candidate, then applies dynamic hybrid rules from `outputs/hybrid_rule_sweep_nllb_aka_enggha/results.csv`.
  - Rules: `Aka_Gha:pred_input:80+80:pred_words>120`, `Eng_Gha:input_pred:50+50:input_words>30`.
  - Status: best local candidate; public score not submitted yet.

- `zindi_submission_nllb_amh_query_0_508147.csv`
  - Local validation: `0.508147020`
  - Change: replaces only `Amh_Eth` rows with query-translated English retrieval answers translated back with NLLB.
  - Status: superseded locally by `zindi_submission_nllb_amh_hybrid_rules_0_508261.csv`.

Public probe result:

- `zindi_probe_akan_broader_eng80_local_0_507316.csv`
  - Public score: `0.601786`
  - Local validation: `0.507315548`
  - Result: worse than the current public baseline; do not continue this broader-Akan direction.

Do not submit `zindi_probe_akan_broader_eng60_local_0_507306.csv` unless a later experiment gives a specific reason.

`aya:8b` translation smoke tests were rejected. NLLB query translation for Amharic
is the first translation-backed run to beat the current local validation baseline,
but the gain is small and affects only 61 public test rows.

Keep the original current best available as:

- `zindi_submission_conditional_hybrid_plus_0_507600.csv`
