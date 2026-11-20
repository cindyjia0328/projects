# Customer churn prediction

Monthly account snapshots are used to predict whether an active paying account churns within 30 days. Two XGBoost models are fit separately: one for first-month accounts, and one for accounts in month 2 or later.

The notebooks are Databricks notebooks. Each modeling notebook starts with `%run ./0_ETL_clean_change_hist`, so keep them in the same folder as the ETL notebook and the five `ravenstack_*.csv` files.

## Run order

1. `0_ETL_clean_change_hist`
2. `M1_model_tuning`
3. `M1_model_final`
4. `M2plus_model_tuning`
5. `M2plus_model_final`

The tuning notebooks choose the parameters. The final notebooks refit those chosen settings and add the explanation plots. Copy the printed `CHOSEN_PARAMETERS` from a tuning notebook into the matching final notebook before running it.

## Notebooks

### `0_ETL_clean_change_hist`

Builds the modeling table from accounts, subscriptions, feature usage, support tickets, and churn events.

- One snapshot on the first day of each month.
- An account is included when a paid, non-trial subscription covers that date and started within the past 31 days. The latest qualifying subscription is kept.
- `flag_churn` is 1 when a churn event falls in the 30 days after the snapshot.
- Account and subscription fields are added, with categoricals one-hot encoded. Tenure is in months.
- Usage and support features are built for the latest 30 days, the prior 30 days, the change between those windows, and the whole history through the snapshot.

The output used downstream is `df_ft_accounts_model_final`.

### `M1_model_tuning`

Tunes the first-month model.

- Keeps rows with `tenure_account == 1`.
- Drops redundant or sparse columns (ARR, the raw churn flag, per-feature 30-day usage, selected whole-history feature columns, and satisfaction-score columns).
- Train: Jan 2023–Jul 2024. Validation: Aug–Sep 2024. Test: Oct–Nov 2024.
- Searches XGBoost depth, minimum child weight, learning rate, number of trees, and `scale_pos_weight`.
- Selects on validation PR-AUC.
- Include a month-by-month view.

### `M1_model_final`

Same population, features, and dates as `M1_model_tuning`. It skips the search and fits the chosen parameters, then adds feature importance, SHAP, and partial dependence on the training rows.

Chosen settings: `scale_pos_weight` 6.85, depth 5, minimum child weight 5, learning rate 0.05, 200 trees.

### `M2plus_model_tuning`

Same process as the first-month tuning notebook, for accounts with `tenure_account > 1`.

### `M2plus_model_final`

Refit of the chosen month-2+ model, plus feature importance, SHAP, and partial dependence.

Chosen settings: `scale_pos_weight` 1, depth 5, minimum child weight 5, learning rate 0.05, 100 trees.

## Shared modeling choices

- Target: `flag_churn`.
- Metric: PR-AUC. Precision and recall are also reported at a fixed score cutoff of 0.10 and for the top 10% of scores.
- Fixed XGBoost settings: subsample 0.8, column sample 0.8, L2 penalty 5, L1 penalty 0.1.
- SHAP and partial dependence describe the fitted model. They are not causal effects.
