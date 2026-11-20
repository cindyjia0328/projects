# Databricks notebook source
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # Import data

# COMMAND ----------

df_accounts = pd.read_csv("ravenstack_accounts.csv")
df_churn_events = pd.read_csv("ravenstack_churn_events.csv")
df_feature_usage = pd.read_csv("ravenstack_feature_usage.csv")
df_subscriptions = pd.read_csv("ravenstack_subscriptions.csv")
df_support_tickets = pd.read_csv("ravenstack_support_tickets.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC # Account universe & churn flag

# COMMAND ----------

# Clean dates once
df_subscriptions["start_date"] = (
    pd.to_datetime(df_subscriptions["start_date"]).dt.normalize()
)
df_subscriptions["end_date"] = (
    pd.to_datetime(df_subscriptions["end_date"]).dt.normalize()
)
df_churn_events["churn_date"] = pd.to_datetime(
    df_churn_events["churn_date"]
).dt.normalize()

target_window = 30

# Every first day from January 2023 through December 2024
snapshot_dates = pd.date_range(
    start=df_subscriptions["start_date"].min().to_period("M").to_timestamp(),
    end=df_churn_events["churn_date"].max().to_period("M").to_timestamp(),
    freq="MS",
)

all_snapshots = []

for snp_dt in snapshot_dates:

    # 1. User universe — exact existing logic
    df_user_input = (
        df_subscriptions[
            (df_subscriptions["start_date"] <= snp_dt)
            & (
                (snp_dt - df_subscriptions["start_date"])
                <= pd.Timedelta(days=31)
            )
            & (
                (df_subscriptions["end_date"] > snp_dt)
                | df_subscriptions["end_date"].isnull()
            )
            & (df_subscriptions["is_trial"] == False)
        ]
        .sort_values(["account_id", "start_date"])
        .copy()
    )

    # Keep latest subscription row per account
    df_user_input["row_number"] = (
        df_user_input.groupby("account_id")["start_date"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    df_user_input_fin = (
        df_user_input[df_user_input["row_number"] == 1].copy()
    )
    df_user_input_fin["snapshot_dt"] = snp_dt

    # 2. Churn flag — exact existing logic
    df_target = df_churn_events[
        (snp_dt < df_churn_events["churn_date"])
        & (
            df_churn_events["churn_date"]
            <= snp_dt + pd.Timedelta(days=target_window)
        )
    ].copy()

    # Keep earliest churn event per account
    df_target["row_number"] = (
        df_target.groupby("account_id")["churn_date"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    df_target_fin = df_target[
        df_target["row_number"] == 1
    ].copy()
    df_target_fin["snapshot_dt"] = snp_dt

    # Join universe and target
    snapshot_result = df_user_input_fin.merge(
        df_target_fin,
        how="left",
        on=["account_id", "snapshot_dt"],
        suffixes=("_sub", "_churn"),
    )

    snapshot_result["flag_churn"] = (
        snapshot_result["churn_date"].notna().astype(int)
    )

    all_snapshots.append(snapshot_result)

# Final account-snapshot table
df_user_input_wtarget_all = pd.concat(
    all_snapshots,
    ignore_index=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC # features from df_accounts

# COMMAND ----------

df_ft_df_accounts = df_user_input_wtarget_all.merge(df_accounts[['account_id','industry','country','signup_date','referral_source']],
                                                    how='inner',on='account_id')
df_ft_df_accounts['signup_date'] = pd.to_datetime(df_ft_df_accounts['signup_date']).dt.normalize()
df_ft_df_accounts['tenure_account'] = (
    (df_ft_df_accounts['snapshot_dt'].dt.year - df_ft_df_accounts['signup_date'].dt.year) * 12 +
    (df_ft_df_accounts['snapshot_dt'].dt.month - df_ft_df_accounts['signup_date'].dt.month)
) # tenure-months
# (df_ft_df_accounts['snapshot_dt'] - df_ft_df_accounts['signup_date']).dt.days # tenure-days

# COMMAND ----------

# MAGIC %md
# MAGIC # features from df_subscriptions

# COMMAND ----------

sub_acct_ft_cat = [
    "plan_tier",
    "billing_frequency",
    "auto_renew_flag",
    "industry",
    "country",
	"referral_source"
]

# Preserve missing values as their own category
df_ft_df_accounts_ohe = df_ft_df_accounts.copy()

df_ft_df_accounts_ohe[sub_acct_ft_cat] = (
    df_ft_df_accounts_ohe[sub_acct_ft_cat]
    .astype("string")
    .fillna("missing")
)

# One-hot encode and automatically drop original categorical columns
df_ft_df_accounts_ohe = pd.get_dummies(
    df_ft_df_accounts_ohe,
    columns=sub_acct_ft_cat,
    prefix=sub_acct_ft_cat,
    prefix_sep="_",
    drop_first=False,
    dtype="int64",
)

sub_acct_ft_drop = [
    'subscription_id','start_date','end_date',
    'is_trial','upgrade_flag', 'downgrade_flag',
    'row_number_sub', 'churn_event_id',
    'churn_date','reason_code', 'refund_amount_usd',
    'preceding_upgrade_flag', 'preceding_downgrade_flag', 'is_reactivation',
    'feedback_text', 'row_number_churn',
    'signup_date'
]
df_ft_df_accounts_ohe = df_ft_df_accounts_ohe.drop(columns=sub_acct_ft_drop)


# COMMAND ----------

acct_info_cols = ['snapshot_dt','account_id','flag_churn']
df_ft_df_accounts_ohe = df_ft_df_accounts_ohe[acct_info_cols + [c for c in df_ft_df_accounts_ohe.columns if c not in acct_info_cols]]

# COMMAND ----------

# MAGIC %md
# MAGIC # features from df_feature_usage & df_support_tickets

# COMMAND ----------

# ============================================================
# FEATURE USAGE + SUPPORT FEATURES
#
# Input:
#   df_ft_df_accounts_ohe
#
# Outputs:
#   df_ft_feature_usage_category
#   df_ft_feature_usage_aggregate
#   df_ft_feature_usage_all
#   df_ft_accounts_ohe_with_usage
#
#   df_ft_support_category
#   df_ft_support_aggregate
#   df_ft_support_all
#   df_ft_accounts_ohe_with_support
#
#   df_ft_accounts_model_final
#
# Added to the original 30-day features:
#   prior 30 days: (snapshot - 60 days, snapshot - 30 days]
#   change: latest 30 days minus that prior 30 days
#   whole-history activity features, alongside the binary flags
#
# df_ft_df_accounts_ohe is NOT changed.
# ============================================================

import re
import numpy as np
import pandas as pd


# ============================================================
# 0. Parameters
# ============================================================

KEYS = [
    "account_id",
    "snapshot_dt",
]

# name, lookback_days, days before snapshot where the window ends
# 30d keeps the original window. The other two are added.
FEATURE_WINDOWS = [
    ("30d", 30, 0),
    ("prior_30d", 30, 30),
    ("whole_hist", None, 0),
]


# ============================================================
# 1. Helper functions
# ============================================================

def safe_feature_name(value):
    """Convert category values into safe column names."""
    return re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def to_binary(series):
    """Convert Boolean-like values into zero/one."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(int)
    )


def calendar_period_count(
    start_date,
    end_date,
    frequency,
    include_start=False,
):
    """Count calendar periods touched by the exposure window."""
    if pd.isna(start_date) or pd.isna(end_date):
        return 1

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    first_date = (
        start_date
        if include_start
        else start_date + pd.Timedelta(days=1)
    )

    if first_date > end_date:
        return 1

    dates = pd.date_range(
        first_date,
        end_date,
        freq="D",
    )

    return max(
        int(dates.to_period(frequency).nunique()),
        1,
    )


def make_exposure_table(
    activity_base,
    lookback_days,
    end_offset_days=0,
):
    """Create exposure denominators for one window."""
    exposure = activity_base.copy()

    exposure["window_end"] = (
        exposure["snapshot_dt"]
        - pd.Timedelta(days=end_offset_days)
    )

    if lookback_days is None:
        # Whole history includes signup date
        exposure["window_start"] = exposure["signup_date"]
        include_start = True
    else:
        # Window excludes its lower boundary
        exposure["window_start"] = (
            exposure["window_end"]
            - pd.Timedelta(days=lookback_days)
        )
        include_start = False

    # Handle missing signup dates
    exposure["window_start"] = (
        exposure["window_start"]
        .fillna(exposure["window_end"])
    )

    # Protect against a start after the window end
    exposure["window_start"] = pd.concat(
        [
            exposure["window_start"],
            exposure["window_end"],
        ],
        axis=1,
    ).min(axis=1)

    exposure["exposure_days"] = (
        exposure["window_end"]
        - exposure["window_start"]
    ).dt.days

    if include_start:
        exposure["exposure_days"] = (
            exposure["exposure_days"] + 1
        )

    exposure["exposure_days"] = (
        exposure["exposure_days"].clip(lower=1)
    )

    exposure["calendar_weeks"] = [
        calendar_period_count(
            start,
            end,
            "W-SUN",
            include_start=include_start,
        )
        for start, end in zip(
            exposure["window_start"],
            exposure["window_end"],
        )
    ]

    exposure["calendar_months"] = [
        calendar_period_count(
            start,
            end,
            "M",
            include_start=include_start,
        )
        for start, end in zip(
            exposure["window_start"],
            exposure["window_end"],
        )
    ]

    return exposure


def apply_window_filter(
    frame,
    activity_date_column,
    lookback_days,
):
    """Apply the requested historical window."""
    window_end = (
        frame["window_end"]
        if "window_end" in frame.columns
        else frame["snapshot_dt"]
    )

    valid = (
        frame[activity_date_column].notna()
        & (
            frame[activity_date_column]
            <= window_end
        )
    )

    if lookback_days is None:
        # Whole history includes the account signup date
        valid = valid & (
            frame[activity_date_column]
            >= frame["window_start"]
        )
    else:
        # Previous 30 days: lower boundary excluded
        valid = valid & (
            frame[activity_date_column]
            > frame["window_start"]
        )

    return frame.loc[valid].copy()


def combine_window_blocks(
    base_keys,
    feature_blocks,
):
    """Combine feature blocks from multiple windows."""
    output = base_keys.drop_duplicates().copy()

    for feature_block in feature_blocks:
        overlapping_columns = (
            set(output.columns)
            .intersection(feature_block.columns)
            .difference(KEYS)
        )

        if overlapping_columns:
            raise ValueError(
                "Duplicate feature columns: "
                + ", ".join(sorted(overlapping_columns))
            )

        output = output.merge(
            feature_block,
            how="left",
            on=KEYS,
            validate="one_to_one",
        )

    feature_columns = [
        column
        for column in output.columns
        if column not in KEYS
    ]

    output[feature_columns] = (
        output[feature_columns].fillna(0)
    )

    return output


def join_feature_table(
    base_df,
    feature_df,
):
    """Join a one-row-per-account-snapshot feature table."""
    overlapping_columns = (
        set(base_df.columns)
        .intersection(feature_df.columns)
        .difference(KEYS)
    )

    if overlapping_columns:
        raise ValueError(
            "Feature columns already exist in base table: "
            + ", ".join(sorted(overlapping_columns))
        )

    output = base_df.merge(
        feature_df,
        how="left",
        on=KEYS,
        validate="one_to_one",
    )

    new_columns = [
        column
        for column in feature_df.columns
        if column not in KEYS
    ]

    output[new_columns] = (
        output[new_columns].fillna(0)
    )

    return output


def store_window_frame(
    base_blocks,
    extra_blocks,
    window_name,
    frame,
):
    """Keep the original 30-day blocks separate from added windows."""
    if window_name == "30d":
        base_blocks.append(frame)
    else:
        extra_blocks[window_name].append(frame)


def make_change_table(
    current,
    previous,
    current_suffix="_30d",
    previous_suffix="_prior_30d",
    change_suffix="_change_30d_vs_prev",
):
    """Subtract the prior window from the current window."""
    current_features = {
        column[: -len(current_suffix)]: column
        for column in current.columns
        if (
            column not in KEYS
            and column.endswith(current_suffix)
        )
    }
    previous_features = {
        column[: -len(previous_suffix)]: column
        for column in previous.columns
        if (
            column not in KEYS
            and column.endswith(previous_suffix)
        )
    }

    stems = sorted(
        set(current_features)
        | set(previous_features)
    )

    output = current[KEYS].copy()
    current_lookup = current.set_index(KEYS)
    previous_lookup = previous.set_index(KEYS)
    output_index = pd.MultiIndex.from_frame(
        output[KEYS]
    )
    change_columns = {}

    for stem in stems:
        if stem in current_features:
            current_values = (
                current_lookup[
                    current_features[stem]
                ]
                .reindex(output_index)
                .fillna(0)
            )
        else:
            current_values = 0

        if stem in previous_features:
            previous_values = (
                previous_lookup[
                    previous_features[stem]
                ]
                .reindex(output_index)
                .fillna(0)
            )
        else:
            previous_values = 0

        change_columns[
            f"{stem}{change_suffix}"
        ] = (
            np.asarray(current_values, dtype=float)
            - np.asarray(previous_values, dtype=float)
        )

    return pd.concat(
        [
            output,
            pd.DataFrame(
                change_columns,
                index=output.index,
            ),
        ],
        axis=1,
    )


# ============================================================
# 2. Base account-snapshot table
# ============================================================

if df_ft_df_accounts_ohe.duplicated(KEYS).any():
    raise ValueError(
        "df_ft_df_accounts_ohe contains duplicate "
        "account-snapshot rows."
    )

df_ft_activity_base = (
    df_ft_df_accounts_ohe[KEYS]
    .drop_duplicates()
    .merge(
        df_accounts[
            [
                "account_id",
                "signup_date",
            ]
        ].drop_duplicates("account_id"),
        how="left",
        on="account_id",
        validate="many_to_one",
    )
)

df_ft_activity_base["snapshot_dt"] = pd.to_datetime(
    df_ft_activity_base["snapshot_dt"]
).dt.normalize()

df_ft_activity_base["signup_date"] = pd.to_datetime(
    df_ft_activity_base["signup_date"]
).dt.normalize()

base_keys = (
    df_ft_activity_base[KEYS]
    .drop_duplicates()
    .copy()
)


# ============================================================
# 3. FEATURE-USAGE FEATURES
# ============================================================

# ------------------------------------------------------------
# 3.1 Prepare usage source
# ------------------------------------------------------------

df_usage_source = df_feature_usage.copy()

df_usage_source["usage_date"] = pd.to_datetime(
    df_usage_source["usage_date"]
).dt.normalize()

df_usage_source["is_beta_feature"] = to_binary(
    df_usage_source["is_beta_feature"]
)

for column in [
    "usage_count",
    "usage_duration_secs",
    "error_count",
]:
    df_usage_source[column] = pd.to_numeric(
        df_usage_source[column],
        errors="coerce",
    ).fillna(0)


# ------------------------------------------------------------
# 3.2 Connect account snapshots to subscriptions
# ------------------------------------------------------------

df_subscription_lookup = (
    df_subscriptions[
        [
            "subscription_id",
            "account_id",
            "start_date",
        ]
    ]
    .drop_duplicates()
    .copy()
)

df_subscription_lookup["start_date"] = pd.to_datetime(
    df_subscription_lookup["start_date"]
).dt.normalize()

df_usage_snapshot_events = base_keys.merge(
    df_subscription_lookup,
    how="left",
    on="account_id",
)

# Subscription must exist by the snapshot
df_usage_snapshot_events = (
    df_usage_snapshot_events[
        df_usage_snapshot_events["start_date"]
        <= df_usage_snapshot_events["snapshot_dt"]
    ]
    .copy()
)

df_usage_snapshot_events = (
    df_usage_snapshot_events.merge(
        df_usage_source,
        how="left",
        on="subscription_id",
    )
)


# ------------------------------------------------------------
# 3.3 Generate usage features
# ------------------------------------------------------------

usage_category_blocks = []
usage_aggregate_blocks = []
usage_category_blocks_added = {
    "prior_30d": [],
    "whole_hist": [],
}
usage_aggregate_blocks_added = {
    "prior_30d": [],
    "whole_hist": [],
}

for (
    window_name,
    lookback_days,
    end_offset_days,
) in FEATURE_WINDOWS:

    df_usage_exposure = make_exposure_table(
        df_ft_activity_base,
        lookback_days,
        end_offset_days=end_offset_days,
    )

    df_usage_window_input = (
        df_usage_snapshot_events.merge(
            df_usage_exposure[
                KEYS
                + [
                    "window_start",
                    "window_end",
                    "exposure_days",
                    "calendar_weeks",
                    "calendar_months",
                ]
            ],
            how="left",
            on=KEYS,
            validate="many_to_one",
        )
    )

    df_usage_window = apply_window_filter(
        df_usage_window_input,
        activity_date_column="usage_date",
        lookback_days=lookback_days,
    )

    df_usage_window["calendar_week"] = (
        df_usage_window["usage_date"]
        .dt.to_period("W-SUN")
        .astype(str)
    )

    # ========================================================
    # 3.3.1 Usage by feature category
    # ========================================================

    df_usage_category_long = (
        df_usage_window
        .groupby(
            KEYS + ["feature_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            usage_count=(
                "usage_count",
                "sum",
            ),
            usage_duration_secs=(
                "usage_duration_secs",
                "sum",
            ),
            error_count=(
                "error_count",
                "sum",
            ),
            is_beta_feature=(
                "is_beta_feature",
                "sum",
            ),
        )
    )

    df_usage_category_total = (
        df_usage_category_long
        .groupby(
            KEYS,
            as_index=False,
        )
        .agg(
            usage_count_total=(
                "usage_count",
                "sum",
            ),
        )
    )

    df_usage_category_long = (
        df_usage_category_long.merge(
            df_usage_category_total,
            how="left",
            on=KEYS,
            validate="many_to_one",
        )
    )

    # Feature-category usage / total usage
    df_usage_category_long["usage_pct"] = np.where(
        df_usage_category_long[
            "usage_count_total"
        ] > 0,
        (
            df_usage_category_long["usage_count"]
            / df_usage_category_long[
                "usage_count_total"
            ]
        ),
        0,
    )

    df_usage_category_long = (
        df_usage_category_long.drop(
            columns=["usage_count_total"]
        )
    )

    df_usage_category_wide = (
        df_usage_category_long.pivot_table(
            index=KEYS,
            columns="feature_name",
            values=[
                "usage_count",
                "usage_duration_secs",
                "error_count",
                "is_beta_feature",
                "usage_pct",
            ],
            aggfunc="sum",
            fill_value=0,
        )
    )

    df_usage_category_wide.columns = [
        (
            f"{metric}_"
            f"{safe_feature_name(feature_name)}_"
            f"{window_name}"
        )
        for metric, feature_name
        in df_usage_category_wide.columns
    ]

    df_usage_category_wide = (
        df_usage_category_wide.reset_index()
    )

    store_window_frame(
        usage_category_blocks,
        usage_category_blocks_added,
        window_name,
        df_usage_category_wide,
    )

    # ========================================================
    # 3.3.2 Aggregate usage amount/frequency/latency
    # ========================================================

    df_usage_aggregate_events = (
        df_usage_window
        .groupby(
            KEYS,
            as_index=False,
        )
        .agg(
            usage_count_total=(
                "usage_count",
                "sum",
            ),
            usage_duration_secs_total=(
                "usage_duration_secs",
                "sum",
            ),
            usage_error_count_total=(
                "error_count",
                "sum",
            ),
            usage_beta_count_total=(
                "is_beta_feature",
                "sum",
            ),
            usage_active_days=(
                "usage_date",
                "nunique",
            ),
            usage_active_weeks=(
                "calendar_week",
                "nunique",
            ),
            usage_last_activity_date=(
                "usage_date",
                "max",
            ),
        )
    )

    df_usage_aggregate = (
        df_usage_exposure.merge(
            df_usage_aggregate_events,
            how="left",
            on=KEYS,
            validate="one_to_one",
        )
    )

    usage_count_columns = [
        "usage_count_total",
        "usage_duration_secs_total",
        "usage_error_count_total",
        "usage_beta_count_total",
        "usage_active_days",
        "usage_active_weeks",
    ]

    df_usage_aggregate[usage_count_columns] = (
        df_usage_aggregate[usage_count_columns]
        .fillna(0)
    )

    # Number of activities per observed day
    df_usage_aggregate[
        "usage_activities_per_day"
    ] = (
        df_usage_aggregate["usage_count_total"]
        / df_usage_aggregate["exposure_days"]
    )

    # Number of active days per calendar week
    df_usage_aggregate[
        "usage_active_days_per_calendar_week"
    ] = (
        df_usage_aggregate["usage_active_days"]
        / df_usage_aggregate["calendar_weeks"]
    )

    # Number of active weeks per calendar month
    df_usage_aggregate[
        "usage_active_weeks_per_calendar_month"
    ] = (
        df_usage_aggregate["usage_active_weeks"]
        / df_usage_aggregate["calendar_months"]
    )

    # Days since most recent usage
    df_usage_aggregate[
        "usage_days_since_last_activity"
    ] = (
        df_usage_aggregate["window_end"]
        - df_usage_aggregate[
            "usage_last_activity_date"
        ]
    ).dt.days

    # No usage: latency equals the full exposure window
    df_usage_aggregate[
        "usage_days_since_last_activity"
    ] = (
        df_usage_aggregate[
            "usage_days_since_last_activity"
        ]
        .fillna(
            df_usage_aggregate["exposure_days"]
        )
    )

    # These columns were only needed as denominators
    df_usage_aggregate = (
        df_usage_aggregate.drop(
            columns=[
                "signup_date",
                "window_start",
                "window_end",
                "exposure_days",
                "calendar_weeks",
                "calendar_months",
                "usage_last_activity_date",
            ]
        )
    )

    df_usage_aggregate = (
        df_usage_aggregate.rename(
            columns={
                column: f"{column}_{window_name}"
                for column
                in df_usage_aggregate.columns
                if column not in KEYS
            }
        )
    )

    store_window_frame(
        usage_aggregate_blocks,
        usage_aggregate_blocks_added,
        window_name,
        df_usage_aggregate,
    )


# Whole history contributes only one binary usage indicator.
df_usage_whole_hist_exposure = make_exposure_table(
    df_ft_activity_base,
    lookback_days=None,
)

df_usage_whole_hist_input = (
    df_usage_snapshot_events.merge(
        df_usage_whole_hist_exposure[
            KEYS + ["window_start"]
        ],
        how="left",
        on=KEYS,
        validate="many_to_one",
    )
)

df_usage_whole_hist_events = apply_window_filter(
    df_usage_whole_hist_input,
    activity_date_column="usage_date",
    lookback_days=None,
)

df_usage_whole_hist_indicator = (
    df_usage_whole_hist_events[KEYS]
    .drop_duplicates()
    .assign(has_usage_whole_hist=1)
)

df_ft_feature_usage_whole_hist = (
    base_keys.merge(
        df_usage_whole_hist_indicator,
        how="left",
        on=KEYS,
        validate="one_to_one",
    )
)

df_ft_feature_usage_whole_hist[
    "has_usage_whole_hist"
] = (
    df_ft_feature_usage_whole_hist[
        "has_usage_whole_hist"
    ]
    .fillna(0)
    .astype(int)
)


# ------------------------------------------------------------
# 3.4 Usage output tables
# ------------------------------------------------------------

df_ft_feature_usage_category = combine_window_blocks(
    base_keys,
    usage_category_blocks,
)

df_ft_feature_usage_aggregate = combine_window_blocks(
    base_keys,
    usage_aggregate_blocks,
)

df_ft_feature_usage_all = join_feature_table(
    df_ft_feature_usage_category,
    df_ft_feature_usage_aggregate,
)

df_ft_feature_usage_prior_30d = join_feature_table(
    combine_window_blocks(
        base_keys,
        usage_category_blocks_added["prior_30d"],
    ),
    combine_window_blocks(
        base_keys,
        usage_aggregate_blocks_added["prior_30d"],
    ),
)

df_ft_feature_usage_whole_hist_activity = join_feature_table(
    combine_window_blocks(
        base_keys,
        usage_category_blocks_added["whole_hist"],
    ),
    combine_window_blocks(
        base_keys,
        usage_aggregate_blocks_added["whole_hist"],
    ),
)

df_ft_feature_usage_change_30d_vs_prev = make_change_table(
    df_ft_feature_usage_all,
    df_ft_feature_usage_prior_30d,
)

df_ft_feature_usage_all = join_feature_table(
    df_ft_feature_usage_all,
    df_ft_feature_usage_whole_hist,
)

df_ft_feature_usage_all = join_feature_table(
    df_ft_feature_usage_all,
    df_ft_feature_usage_whole_hist_activity,
)

df_ft_feature_usage_all = join_feature_table(
    df_ft_feature_usage_all,
    df_ft_feature_usage_change_30d_vs_prev,
)

df_ft_accounts_ohe_with_usage = join_feature_table(
    df_ft_df_accounts_ohe,
    df_ft_feature_usage_all,
)


# ============================================================
# 4. SUPPORT-TICKET FEATURES
# ============================================================

# ------------------------------------------------------------
# 4.1 Prepare support source
# ------------------------------------------------------------

df_support_source = df_support_tickets.copy()

df_support_source["submitted_at"] = pd.to_datetime(
    df_support_source["submitted_at"]
).dt.normalize()

df_support_source["escalation_flag"] = to_binary(
    df_support_source["escalation_flag"]
)

for column in [
    "resolution_time_hours",
    "first_response_time_minutes",
    "satisfaction_score",
]:
    df_support_source[column] = pd.to_numeric(
        df_support_source[column],
        errors="coerce",
    )

# Preserve your satisfaction midpoint logic
satisfaction_midpoint = (
    df_support_source["satisfaction_score"].max()
    + df_support_source["satisfaction_score"].min()
) / 2

df_support_source["satisfaction_score"] = (
    df_support_source["satisfaction_score"]
    .fillna(satisfaction_midpoint)
)

df_support_snapshot_events = base_keys.merge(
    df_support_source,
    how="left",
    on="account_id",
)


# ------------------------------------------------------------
# 4.2 Generate support features
# ------------------------------------------------------------

support_category_blocks = []
support_aggregate_blocks = []
support_category_blocks_added = {
    "prior_30d": [],
    "whole_hist": [],
}
support_aggregate_blocks_added = {
    "prior_30d": [],
    "whole_hist": [],
}

for (
    window_name,
    lookback_days,
    end_offset_days,
) in FEATURE_WINDOWS:

    df_support_exposure = make_exposure_table(
        df_ft_activity_base,
        lookback_days,
        end_offset_days=end_offset_days,
    )

    df_support_window_input = (
        df_support_snapshot_events.merge(
            df_support_exposure[
                KEYS
                + [
                    "window_start",
                    "window_end",
                    "exposure_days",
                    "calendar_weeks",
                    "calendar_months",
                ]
            ],
            how="left",
            on=KEYS,
            validate="many_to_one",
        )
    )

    df_support_window = apply_window_filter(
        df_support_window_input,
        activity_date_column="submitted_at",
        lookback_days=lookback_days,
    )

    df_support_window["calendar_week"] = (
        df_support_window["submitted_at"]
        .dt.to_period("W-SUN")
        .astype(str)
    )

    # ========================================================
    # 4.2.1 Support features by priority category
    # ========================================================

    df_support_category_long = (
        df_support_window
        .groupby(
            KEYS + ["priority"],
            as_index=False,
            dropna=False,
        )
        .agg(
            num_ticket=(
                "ticket_id",
                "nunique",
            ),
            sum_resolution_time_hours=(
                "resolution_time_hours",
                "sum",
            ),
            sum_first_response_time_minutes=(
                "first_response_time_minutes",
                "sum",
            ),
            sum_satisfaction_score=(
                "satisfaction_score",
                "sum",
            ),
            sum_escalation_flag=(
                "escalation_flag",
                "sum",
            ),
        )
    )

    ticket_denominator = (
        df_support_category_long["num_ticket"]
        .replace(0, np.nan)
    )

    df_support_category_long[
        "avg_resolution_time_hours"
    ] = (
        df_support_category_long[
            "sum_resolution_time_hours"
        ]
        / ticket_denominator
    )

    df_support_category_long[
        "avg_first_response_time_minutes"
    ] = (
        df_support_category_long[
            "sum_first_response_time_minutes"
        ]
        / ticket_denominator
    )

    df_support_category_long[
        "avg_satisfaction_score"
    ] = (
        df_support_category_long[
            "sum_satisfaction_score"
        ]
        / ticket_denominator
    )

    df_support_category_long[
        "avg_escalation_flag"
    ] = (
        df_support_category_long[
            "sum_escalation_flag"
        ]
        / ticket_denominator
    )

    df_support_category_total = (
        df_support_category_long
        .groupby(
            KEYS,
            as_index=False,
        )
        .agg(
            num_ticket_total=(
                "num_ticket",
                "sum",
            ),
        )
    )

    df_support_category_long = (
        df_support_category_long.merge(
            df_support_category_total,
            how="left",
            on=KEYS,
            validate="many_to_one",
        )
    )

    # Priority ticket count / all ticket count
    df_support_category_long[
        "ticket_pct"
    ] = np.where(
        df_support_category_long[
            "num_ticket_total"
        ] > 0,
        (
            df_support_category_long["num_ticket"]
            / df_support_category_long[
                "num_ticket_total"
            ]
        ),
        0,
    )

    df_support_category_long = (
        df_support_category_long.drop(
            columns=[
                "num_ticket_total",
                "sum_resolution_time_hours",
                "sum_first_response_time_minutes",
                "sum_satisfaction_score",
                "sum_escalation_flag",
            ]
        )
    )

    df_support_category_wide = (
        df_support_category_long.pivot_table(
            index=KEYS,
            columns="priority",
            values=[
                "num_ticket",
                "ticket_pct",
                "avg_resolution_time_hours",
                "avg_first_response_time_minutes",
                "avg_satisfaction_score",
                "avg_escalation_flag",
            ],
            aggfunc="mean",
            fill_value=0,
        )
    )

    df_support_category_wide.columns = [
        (
            f"{metric}_"
            f"{safe_feature_name(priority)}_"
            f"{window_name}"
        )
        for metric, priority
        in df_support_category_wide.columns
    ]

    df_support_category_wide = (
        df_support_category_wide.reset_index()
    )

    store_window_frame(
        support_category_blocks,
        support_category_blocks_added,
        window_name,
        df_support_category_wide,
    )

    # ========================================================
    # 4.2.2 Aggregate support amount/frequency/latency
    # ========================================================

    df_support_aggregate_events = (
        df_support_window
        .groupby(
            KEYS,
            as_index=False,
        )
        .agg(
            support_ticket_count_total=(
                "ticket_id",
                "nunique",
            ),
            support_resolution_hours_total=(
                "resolution_time_hours",
                "sum",
            ),
            support_first_response_minutes_total=(
                "first_response_time_minutes",
                "sum",
            ),
            support_satisfaction_score_total=(
                "satisfaction_score",
                "sum",
            ),
            support_escalation_count_total=(
                "escalation_flag",
                "sum",
            ),
            support_active_days=(
                "submitted_at",
                "nunique",
            ),
            support_active_weeks=(
                "calendar_week",
                "nunique",
            ),
            support_last_activity_date=(
                "submitted_at",
                "max",
            ),
        )
    )

    df_support_aggregate = (
        df_support_exposure.merge(
            df_support_aggregate_events,
            how="left",
            on=KEYS,
            validate="one_to_one",
        )
    )

    support_count_columns = [
        "support_ticket_count_total",
        "support_resolution_hours_total",
        "support_first_response_minutes_total",
        "support_satisfaction_score_total",
        "support_escalation_count_total",
        "support_active_days",
        "support_active_weeks",
    ]

    df_support_aggregate[
        support_count_columns
    ] = (
        df_support_aggregate[
            support_count_columns
        ].fillna(0)
    )

    support_ticket_denominator = (
        df_support_aggregate[
            "support_ticket_count_total"
        ]
        .replace(0, np.nan)
    )

    df_support_aggregate[
        "support_avg_resolution_time_hours"
    ] = (
        df_support_aggregate[
            "support_resolution_hours_total"
        ]
        / support_ticket_denominator
    )

    df_support_aggregate[
        "support_avg_first_response_time_minutes"
    ] = (
        df_support_aggregate[
            "support_first_response_minutes_total"
        ]
        / support_ticket_denominator
    )

    df_support_aggregate[
        "support_avg_satisfaction_score"
    ] = (
        df_support_aggregate[
            "support_satisfaction_score_total"
        ]
        / support_ticket_denominator
    )

    df_support_aggregate[
        "support_avg_escalation_flag"
    ] = (
        df_support_aggregate[
            "support_escalation_count_total"
        ]
        / support_ticket_denominator
    )

    # Tickets per observed day
    df_support_aggregate[
        "support_activities_per_day"
    ] = (
        df_support_aggregate[
            "support_ticket_count_total"
        ]
        / df_support_aggregate["exposure_days"]
    )

    # Number of active support days per calendar week
    df_support_aggregate[
        "support_active_days_per_calendar_week"
    ] = (
        df_support_aggregate[
            "support_active_days"
        ]
        / df_support_aggregate["calendar_weeks"]
    )

    # Number of active support weeks per calendar month
    df_support_aggregate[
        "support_active_weeks_per_calendar_month"
    ] = (
        df_support_aggregate[
            "support_active_weeks"
        ]
        / df_support_aggregate["calendar_months"]
    )

    # Days since most recent support ticket
    df_support_aggregate[
        "support_days_since_last_activity"
    ] = (
        df_support_aggregate["window_end"]
        - df_support_aggregate[
            "support_last_activity_date"
        ]
    ).dt.days

    # No ticket: latency equals full exposure
    df_support_aggregate[
        "support_days_since_last_activity"
    ] = (
        df_support_aggregate[
            "support_days_since_last_activity"
        ]
        .fillna(
            df_support_aggregate["exposure_days"]
        )
    )

    support_average_columns = [
        "support_avg_resolution_time_hours",
        "support_avg_first_response_time_minutes",
        "support_avg_satisfaction_score",
        "support_avg_escalation_flag",
    ]

    df_support_aggregate[
        support_average_columns
    ] = (
        df_support_aggregate[
            support_average_columns
        ].fillna(0)
    )

    # Denominators and intermediate sums are not final features
    df_support_aggregate = (
        df_support_aggregate.drop(
            columns=[
                "signup_date",
                "window_start",
                "window_end",
                "exposure_days",
                "calendar_weeks",
                "calendar_months",
                "support_last_activity_date",
                "support_resolution_hours_total",
                "support_first_response_minutes_total",
                "support_satisfaction_score_total",
                "support_escalation_count_total",
            ]
        )
    )

    df_support_aggregate = (
        df_support_aggregate.rename(
            columns={
                column: f"{column}_{window_name}"
                for column
                in df_support_aggregate.columns
                if column not in KEYS
            }
        )
    )

    store_window_frame(
        support_aggregate_blocks,
        support_aggregate_blocks_added,
        window_name,
        df_support_aggregate,
    )


# Whole history contributes only one binary support indicator.
df_support_whole_hist_exposure = make_exposure_table(
    df_ft_activity_base,
    lookback_days=None,
)

df_support_whole_hist_input = (
    df_support_snapshot_events.merge(
        df_support_whole_hist_exposure[
            KEYS + ["window_start"]
        ],
        how="left",
        on=KEYS,
        validate="many_to_one",
    )
)

df_support_whole_hist_events = apply_window_filter(
    df_support_whole_hist_input,
    activity_date_column="submitted_at",
    lookback_days=None,
)

df_support_whole_hist_indicator = (
    df_support_whole_hist_events[KEYS]
    .drop_duplicates()
    .assign(has_support_whole_hist=1)
)

df_ft_support_whole_hist = (
    base_keys.merge(
        df_support_whole_hist_indicator,
        how="left",
        on=KEYS,
        validate="one_to_one",
    )
)

df_ft_support_whole_hist[
    "has_support_whole_hist"
] = (
    df_ft_support_whole_hist[
        "has_support_whole_hist"
    ]
    .fillna(0)
    .astype(int)
)


# ------------------------------------------------------------
# 4.3 Support output tables
# ------------------------------------------------------------

df_ft_support_category = combine_window_blocks(
    base_keys,
    support_category_blocks,
)

df_ft_support_aggregate = combine_window_blocks(
    base_keys,
    support_aggregate_blocks,
)

df_ft_support_all = join_feature_table(
    df_ft_support_category,
    df_ft_support_aggregate,
)

df_ft_support_prior_30d = join_feature_table(
    combine_window_blocks(
        base_keys,
        support_category_blocks_added["prior_30d"],
    ),
    combine_window_blocks(
        base_keys,
        support_aggregate_blocks_added["prior_30d"],
    ),
)

df_ft_support_whole_hist_activity = join_feature_table(
    combine_window_blocks(
        base_keys,
        support_category_blocks_added["whole_hist"],
    ),
    combine_window_blocks(
        base_keys,
        support_aggregate_blocks_added["whole_hist"],
    ),
)

df_ft_support_change_30d_vs_prev = make_change_table(
    df_ft_support_all,
    df_ft_support_prior_30d,
)

df_ft_support_all = join_feature_table(
    df_ft_support_all,
    df_ft_support_whole_hist,
)

df_ft_support_all = join_feature_table(
    df_ft_support_all,
    df_ft_support_whole_hist_activity,
)

df_ft_support_all = join_feature_table(
    df_ft_support_all,
    df_ft_support_change_30d_vs_prev,
)

df_ft_accounts_ohe_with_support = join_feature_table(
    df_ft_df_accounts_ohe,
    df_ft_support_all,
)


# ============================================================
# 5. Final combined model table
# ============================================================

# Original df_ft_df_accounts_ohe remains unchanged
df_ft_accounts_model_final = join_feature_table(
    df_ft_df_accounts_ohe,
    df_ft_feature_usage_all,
)

df_ft_accounts_model_final = join_feature_table(
    df_ft_accounts_model_final,
    df_ft_support_all,
)

# Put identifiers and target first
model_info_columns = [
    "snapshot_dt",
    "account_id",
    "flag_churn",
]

df_ft_accounts_model_final = (
    df_ft_accounts_model_final[
        model_info_columns
        + [
            column
            for column
            in df_ft_accounts_model_final.columns
            if column not in model_info_columns
        ]
    ]
)


# COMMAND ----------

# MAGIC %md
# MAGIC # Product tables
# MAGIC
# MAGIC - `df_user_input_wtarget_all`: account-snapshot universe and target
# MAGIC - `df_ft_df_accounts_ohe`: account and subscription features
# MAGIC - `df_ft_feature_usage_all`: 30-day usage, whole-history usage, and 30-day change
# MAGIC - `df_ft_support_all`: 30-day support, whole-history support, and 30-day change
# MAGIC - `df_ft_accounts_model_final`: final joined model table
# MAGIC
# MAGIC Change is the original 30-day window minus the preceding 30-day window, `(snapshot - 60 days, snapshot - 30 days]`.

# COMMAND ----------

df_ft_accounts_model_final