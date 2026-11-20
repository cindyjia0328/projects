# Databricks notebook source
# MAGIC %md
# MAGIC # Tune XGBoost Mall model
# MAGIC
# MAGIC This notebook preserves the exact feature list, row filter, and chronological split from `2_train_xgboost_Mall`, then tunes imbalance handling and conservative tree complexity using validation data only.
# MAGIC
# MAGIC Controls are provided for:
# MAGIC
# MAGIC - Feature inputs
# MAGIC - Row filters applied before splitting
# MAGIC - Train, validation, and test snapshot ranges

# COMMAND ----------

# MAGIC %run ./0_ETL_clean_change_hist

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Feature lists
# MAGIC
# MAGIC Edit only the `FEATURES` assignment when changing model inputs.

# COMMAND ----------

# DBTITLE 1,drop cols
# feature/support cat for 30d_prev

CHANGE_SUFFIX = "_change_30d_vs_prev"

USAGE_METRICS = [
    "usage_duration_secs",
    "is_beta_feature",
    "usage_count",
    "error_count",
    "usage_pct",
]

SUPPORT_METRICS = [
    "avg_first_response_time_minutes",
    "avg_resolution_time_hours",
    "avg_satisfaction_score",
    "avg_escalation_flag",
    "num_ticket",
    "ticket_pct",
]


def is_category_change(column, metrics, usage=False):
    if not column.endswith(CHANGE_SUFFIX):
        return False
    stem = column[: -len(CHANGE_SUFFIX)]
    for metric in metrics:
        prefix = f"{metric}_"
        if stem.startswith(prefix):
            category = stem[len(prefix):]
            if usage and not category.startswith("feature_"):
                return False
            return True
    return False


change_category_features = sorted(
    column
    for column in df_ft_accounts_model_final.columns
    if is_category_change(column, USAGE_METRICS, usage=True)
    or is_category_change(column, SUPPORT_METRICS)
)

print(len(change_category_features))
# change_category_features

# whole_hist: usage_duration_secs, is_beta_feature, usage_count, error_count
WHOLE_HIST_SUFFIX = "_whole_hist"

WHOLE_HIST_USAGE_METRICS = [
    "usage_duration_secs",
    "is_beta_feature",
    "usage_count",
    "error_count"
]

whole_hist_usage_features = sorted(
    column
    for column in df_ft_accounts_model_final.columns
    if column.endswith(WHOLE_HIST_SUFFIX)
    and any(
        column.startswith(f"{metric}_feature_")
        for metric in WHOLE_HIST_USAGE_METRICS
    )
)

print(len(whole_hist_usage_features))
# whole_hist_usage_features

# satisfication
avg_satisfaction_score_features = sorted(
    column
    for column in df_ft_accounts_model_final.columns
    if "satisfaction_score_" in column
)

print(len(avg_satisfaction_score_features))
# avg_satisfaction_score_features

# 30d feature usage raw
usage_cat_feature_30d = sorted(
    column
    for column in df_ft_accounts_model_final.columns
    if column.endswith("_30d")
    and not column.endswith("_prior_30d")
    and (
        column.startswith("usage_count_feature_")
        or column.startswith("is_beta_feature_feature_")
        or column.startswith("error_count_feature_")
        or column.startswith("usage_duration_secs_feature_")
        or column.startswith("usage_pct_feature_")
    )
)

print(len(usage_cat_feature_30d))
# usage_cat_feature_30d

# COMMAND ----------

KEY_COLUMNS = [
    "snapshot_dt",
    "account_id",
]

LABEL = "flag_churn"

feature_drop = change_category_features + whole_hist_usage_features + avg_satisfaction_score_features + usage_cat_feature_30d + ['arr_amount', 
'support_avg_resolution_time_hours_whole_hist', 'support_avg_first_response_time_minutes_whole_hist', 'churn_flag']

# ============================================================
# EDIT THIS FEATURE LIST
# ============================================================

FEATURES = [
    i for i in list(df_ft_accounts_model_final.columns)
    if i not in KEY_COLUMNS+[LABEL]
    and i not in feature_drop
]

len(FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Optional row filter
# MAGIC
# MAGIC Edit `MODEL_FILTER` below. The filter is applied before train, validation, and test are created.

# COMMAND ----------

model_source = (
    df_ft_accounts_model_final.copy()
)

model_source["snapshot_dt"] = pd.to_datetime(
    model_source["snapshot_dt"]
).dt.normalize()

# ============================================================
# EDIT THIS FILTER
# ============================================================

# Default: retain every row
MODEL_FILTER = pd.Series(
    True,
    index=model_source.index,
)


# tenure > 1:
MODEL_FILTER &= (
    model_source["tenure_account"] > 1
)


model_input = (
    model_source.loc[MODEL_FILTER]
    .copy()
)

print(
    "Rows before filter:",
    len(model_source),
)

print(
    "Rows after filter:",
    len(model_input),
)

print(
    "Accounts after filter:",
    model_input["account_id"].nunique(),
)


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Train, validation, and test dates
# MAGIC
# MAGIC Date boundaries are inclusive.

# COMMAND ----------

# ============================================================
# EDIT THESE DATE RANGES
# ============================================================

TRAIN_START = pd.Timestamp("2023-01-01")
TRAIN_END = pd.Timestamp("2024-07-01")

VALIDATION_START = pd.Timestamp("2024-08-01")
VALIDATION_END = pd.Timestamp("2024-09-01")

TEST_START = pd.Timestamp("2024-10-01")
TEST_END = pd.Timestamp("2024-11-01")


# COMMAND ----------

if not (
    TRAIN_START
    <= TRAIN_END
    < VALIDATION_START
    <= VALIDATION_END
    < TEST_START
    <= TEST_END
):
    raise ValueError(
        "Train, validation, and test dates "
        "must be chronological and non-overlapping"
    )

train = model_input.loc[
    model_input["snapshot_dt"].between(
        TRAIN_START,
        TRAIN_END,
    )
].copy()

validation = model_input.loc[
    model_input["snapshot_dt"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
].copy()

test = model_input.loc[
    model_input["snapshot_dt"].between(
        TEST_START,
        TEST_END,
    )
].copy()

splits = {
    "train": train,
    "validation": validation,
    "test": test,
}

for split_name, split_frame in splits.items():
    if split_frame.empty:
        raise ValueError(
            f"{split_name} is empty after filtering"
        )

    if split_frame[LABEL].nunique() < 2:
        raise ValueError(
            f"{split_name} contains only one target class"
        )

split_summary = pd.DataFrame(
    [
        {
            "split": split_name,
            "snapshot_start": (
                split_frame[
                    "snapshot_dt"
                ].min()
            ),
            "snapshot_end": (
                split_frame[
                    "snapshot_dt"
                ].max()
            ),
            "snapshot_months": (
                split_frame[
                    "snapshot_dt"
                ].nunique()
            ),
            "rows": len(split_frame),
            "accounts": (
                split_frame[
                    "account_id"
                ].nunique()
            ),
            "churn_events": int(
                split_frame[LABEL].sum()
            ),
            "churn_rate": (
                split_frame[LABEL].mean()
            ),
        }
        for split_name, split_frame
        in splits.items()
    ]
)

display(split_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Imbalance and tuning controls
# MAGIC
# MAGIC `scale_pos_weight` candidates include:
# MAGIC
# MAGIC - `1`: no class weighting
# MAGIC - Square root of the train imbalance ratio: moderate weighting
# MAGIC - Full train imbalance ratio: standard inverse-frequency weighting
# MAGIC
# MAGIC No SMOTE is used. Weighting is selected only when it improves later validation ranking.

# COMMAND ----------

from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
)

negative_train = int(
    y_train.eq(0).sum()
) if "y_train" in globals() else int(
    train[LABEL].eq(0).sum()
)

positive_train = int(
    y_train.eq(1).sum()
) if "y_train" in globals() else int(
    train[LABEL].eq(1).sum()
)

TRAIN_IMBALANCE_RATIO = (
    negative_train / positive_train
)

SCALE_POS_WEIGHT_GRID = [
    1.0,
    # float(
    #     np.sqrt(
    #         TRAIN_IMBALANCE_RATIO
    #     )
    # ),
    float(TRAIN_IMBALANCE_RATIO),
]

MAX_DEPTH_GRID = [5, 10, 20]
MIN_CHILD_WEIGHT_GRID = [5, 15, 30]
LEARNING_RATE_GRID = [0.05, 0.1, 0.2]
N_ESTIMATORS_GRID = [100, 200, 300]

TOP_SHARE = 0.10
RANDOM_SEED = 7

imbalance_summary = pd.DataFrame(
    {
        "metric": [
            "negative train rows",
            "positive train rows",
            "train churn rate",
            "negative/positive ratio",
        ],
        "value": [
            negative_train,
            positive_train,
            positive_train
            / (
                positive_train
                + negative_train
            ),
            TRAIN_IMBALANCE_RATIO,
        ],
    }
)

display(imbalance_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Training inputs

# COMMAND ----------

X_train = train[FEATURES]
y_train = train[LABEL].astype(int)

X_validation = validation[FEATURES]
y_validation = validation[LABEL].astype(int)

X_test = test[FEATURES]
y_test = test[LABEL].astype(int)


def top_share_metrics(
    labels,
    scores,
    top_share=TOP_SHARE,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )
    scores = np.asarray(
        scores,
        dtype=float,
    )

    top_n = max(
        1,
        int(
            np.ceil(
                len(labels) * top_share
            )
        ),
    )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    top_labels = labels[
        order[:top_n]
    ]

    return {
        "precision_at_top_share": (
            top_labels.mean()
        ),
        "recall_at_top_share": (
            top_labels.sum()
            / labels.sum()
            if labels.sum() > 0
            else 0
        ),
    }


def monthly_ranking_metrics(
    frame,
    scores,
):
    work = frame[
        ["snapshot_dt", LABEL]
    ].copy()

    work["score"] = np.asarray(
        scores,
        dtype=float,
    )

    rows = []

    for snapshot_dt, month in (
        work.groupby(
            "snapshot_dt",
            sort=True,
        )
    ):
        labels = month[LABEL].astype(int)

        if labels.nunique() < 2:
            continue

        top_metrics = top_share_metrics(
            labels,
            month["score"],
        )

        rows.append(
            {
                "snapshot_dt": snapshot_dt,
                "rows": len(month),
                "churn_events": int(
                    labels.sum()
                ),
                "churn_rate": labels.mean(),
                "pr_auc": (
                    average_precision_score(
                        labels,
                        month["score"],
                    )
                ),
                "roc_auc": roc_auc_score(
                    labels,
                    month["score"],
                ),
                **top_metrics,
            }
        )

    return pd.DataFrame(rows)


# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Validation-only hyperparameter search
# MAGIC
# MAGIC Selection prioritizes average monthly validation PR-AUC, then the worst validation month, pooled PR-AUC, and pooled ROC-AUC. Top-10% metrics are reported but are not used to select the model.

# COMMAND ----------

# Paste the five values from the earlier "Chosen parameters" output.
CHOSEN_PARAMETERS = {
    "scale_pos_weight": 1,
    "max_depth": 5,
    "min_child_weight": 5,
    "learning_rate": 0.05,
    "n_estimators": 100,
}

model_xgboost_tuned = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=CHOSEN_PARAMETERS["scale_pos_weight"],
    max_depth=CHOSEN_PARAMETERS["max_depth"],
    min_child_weight=CHOSEN_PARAMETERS["min_child_weight"],
    learning_rate=CHOSEN_PARAMETERS["learning_rate"],
    n_estimators=CHOSEN_PARAMETERS["n_estimators"],
    subsample=0.80,
    colsample_bytree=0.80,
    reg_lambda=5.0,
    reg_alpha=0.10,
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=4,
    verbosity=0,
)

model_xgboost_tuned.fit(X_train, y_train)

print("Chosen parameters:")
print(CHOSEN_PARAMETERS)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Fixed score threshold
# MAGIC
# MAGIC Default and tuned XGBoost use the same fixed probability threshold of `0.50`. Top-10% metrics are reporting-only and do not affect model or threshold selection.

# COMMAND ----------

SCORE_THRESHOLD = 0.10

validation_scores_tuned = (
    model_xgboost_tuned.predict_proba(
        X_validation
    )[:, 1]
)

print(
    f"Fixed score threshold: "
    f"{SCORE_THRESHOLD:.2f}"
)


# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Final evaluation
# MAGIC
# MAGIC The test period is scored only after model selection is complete. Every XGBoost model uses the same `0.50` cutoff.

# COMMAND ----------

def full_performance(
    model_name,
    split_name,
    labels,
    scores,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )
    scores = np.asarray(
        scores,
        dtype=float,
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    top_metrics = top_share_metrics(
        labels,
        scores,
    )

    return {
        "model": model_name,
        "split": split_name,
        "rows": len(labels),
        "churn_events": int(labels.sum()),
        "churn_rate": labels.mean(),
        "mean_score": scores.mean(),
        "pr_auc": average_precision_score(
            labels,
            scores,
        ),
        "roc_auc": roc_auc_score(
            labels,
            scores,
        ),
        "brier": brier_score_loss(
            labels,
            scores,
        ),
        "log_loss": log_loss(
            labels,
            scores,
            labels=[0, 1],
        ),
        "threshold": threshold,
        "flagged_rate": (
            predictions.mean()
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        **top_metrics,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


tuned_score_lookup = {
    "train": (
        model_xgboost_tuned
        .predict_proba(X_train)[:, 1]
    ),
    "validation": (
        validation_scores_tuned
    ),
    "test": (
        model_xgboost_tuned
        .predict_proba(X_test)[:, 1]
    ),
}

tuned_metrics = pd.DataFrame(
    [
        full_performance(
            model_name="xgboost_tuned",
            split_name=split_name,
            labels=split_frame[LABEL],
            scores=(
                tuned_score_lookup[
                    split_name
                ]
            ),
            threshold=SCORE_THRESHOLD,
        )
        for split_name, split_frame
        in splits.items()
    ]
)

display(tuned_metrics)


# COMMAND ----------

# MAGIC %md
# MAGIC # 11. Month-level validation and test stability

# COMMAND ----------

monthly_comparison_blocks = []

for split_name in [
    "validation",
    "test",
]:
    split_frame = splits[split_name]

    model_scores = {
        "xgboost_tuned":
            tuned_score_lookup[
                split_name
            ],
        "xgboost_default":
            default_score_lookup[
                split_name
            ],
    }

    tenure_scores = split_frame[
        "tenure_account"
    ].astype(float)

    if not TENURE_HIGHER_IS_RISKIER:
        tenure_scores = -tenure_scores

    model_scores[
        "tenure_rank"
    ] = tenure_scores.to_numpy()

    for model_name, scores in (
        model_scores.items()
    ):
        monthly = monthly_ranking_metrics(
            split_frame,
            scores,
        )
        monthly["model"] = model_name
        monthly["split"] = split_name

        monthly_comparison_blocks.append(
            monthly
        )

monthly_ranking_comparison = pd.concat(
    monthly_comparison_blocks,
    ignore_index=True,
).sort_values(
    [
        "split",
        "snapshot_dt",
        "model",
    ]
)

display(monthly_ranking_comparison)


# COMMAND ----------

# MAGIC %md
# MAGIC # 12. Tuned feature importance
# MAGIC
# MAGIC Importance is model attribution, not causal evidence.

# COMMAND ----------

tuned_feature_importance = (
    pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": (
                model_xgboost_tuned
                .feature_importances_
            ),
        }
    )
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

tuned_feature_importance.insert(
    0,
    "rank",
    range(
        1,
        len(
            tuned_feature_importance
        ) + 1,
    ),
)

display(tuned_feature_importance)

top_importance = (
    tuned_feature_importance
    .head(30)
    .sort_values("importance")
)

fig, ax = plt.subplots(
    figsize=(11, 10)
)

ax.barh(
    top_importance["feature"],
    top_importance["importance"],
    color="#4C78A8",
)

ax.set_xlabel(
    "Tuned XGBoost feature importance"
)
ax.set_title(
    "Top 30 Tuned XGBoost Features"
)

fig.tight_layout()
plt.show()
plt.close(fig)


# COMMAND ----------

# MAGIC %md
# MAGIC # 13. Insights

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

import shap
from sklearn.inspection import PartialDependenceDisplay

# "test", "validation", or "train"
EXPLAIN_SPLIT = "train"
TOP_N_SHAP = 50
TOP_N_PDP = 50

explain_frames = {
    "train": X_train,
    "validation": X_validation,
    "test": X_test,
}
X_explain = explain_frames[EXPLAIN_SPLIT]

explainer = shap.TreeExplainer(model_xgboost_tuned)
shap_values = explainer.shap_values(X_explain)

if isinstance(shap_values, list):
    shap_values = shap_values[-1]

shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]

global_shap = (
    pd.DataFrame({
        "feature": list(X_explain.columns),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    })
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
global_shap.insert(0, "rank", np.arange(1, len(global_shap) + 1))
display(global_shap.head(TOP_N_SHAP))

top_shap = (
    global_shap
    .head(TOP_N_SHAP)
    .sort_values("mean_abs_shap")
)

fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(top_shap["feature"], top_shap["mean_abs_shap"], color="#4C78A8")
ax.set_xlabel(f"Mean absolute SHAP on {EXPLAIN_SPLIT}")
ax.set_title("Global SHAP")
fig.tight_layout()
plt.show()

base_value = np.asarray(explainer.expected_value).reshape(-1)[-1]
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(len(X_explain), base_value),
    data=X_explain.to_numpy(),
    feature_names=list(X_explain.columns),
)
shap.plots.beeswarm(shap_explanation, max_display=TOP_N_SHAP, show=False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDP

# COMMAND ----------

label_by_split = {
    "train": y_train,
    "validation": y_validation,
    "test": y_test,
}

pdp_features = global_shap["feature"].head(TOP_N_PDP).tolist()
y_explain = label_by_split[EXPLAIN_SPLIT]
baseline_score = model_xgboost_tuned.predict_proba(X_explain)[:, 1].mean()
observed_churn_rate = float(y_explain.mean())


def pdp_grid(values):
    values = pd.Series(values).dropna()
    unique_values = np.sort(values.unique())
    if len(unique_values) <= 5:
        return unique_values
    return np.quantile(values, np.linspace(0.05, 0.95, 20))


pdp_rows = []
pdp_curves = {}

for feature in pdp_features:
    grid = pdp_grid(X_explain[feature])
    scored = X_explain.copy()
    lifts = []

    for value in grid:
        scored[feature] = value
        avg_score = model_xgboost_tuned.predict_proba(scored)[:, 1].mean()
        lift = avg_score / baseline_score
        lifts.append(lift)
        pdp_rows.append({
            "feature": feature,
            "value": value,
            "avg_model_score": avg_score,
            "lift_vs_average_account": lift,
            "pct_vs_average_account": lift - 1,
        })

    pdp_curves[feature] = (grid, np.asarray(lifts))

pdp_table = pd.DataFrame(pdp_rows)
display(pdp_table)

n_cols = 3
n_rows = int(np.ceil(len(pdp_features) / n_cols))
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(5 * n_cols, 3.5 * n_rows),
    squeeze=False,
)

for ax, feature in zip(axes.ravel(), pdp_features):
    grid, lifts = pdp_curves[feature]
    if len(grid) <= 5:
        ax.bar(np.arange(len(grid)), lifts, color="#4C78A8")
        ax.set_xticks(np.arange(len(grid)))
        ax.set_xticklabels([f"{value:g}" for value in grid])
    else:
        ax.plot(grid, lifts, color="#4C78A8", linewidth=2)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_title(feature, fontsize=9)
    ax.set_ylabel("Lift vs average account")
    ax.set_xlabel(feature)

for ax in axes.ravel()[len(pdp_features):]:
    ax.axis("off")

fig.suptitle(
    f"{EXPLAIN_SPLIT}: 1.0 = average account"
    f"\nObserved churn rate {observed_churn_rate:.1%}; "
    f"average model score {baseline_score:.1%}",
    y=1.04,
)
fig.tight_layout()
plt.show()