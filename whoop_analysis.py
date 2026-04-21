# whoop_analysis.py
# author: Austin Juhl

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TIME_WINDOWS = ["1M", "3M", "6M", "1Y", "2Y", "All Time"]
HABIT_COLUMNS = [
    "Alcohol",
    "Screen",
    "Read Book",
    "Caffeine",
    "Sick",
    "Shared Bed",
    "Activity",
]

BLUE = "#4C78D8"
BLUE_SOFT = "rgba(76, 120, 216, 0.35)"
RED = "#C44E52"
RED_SOFT = "rgba(196, 78, 82, 0.35)"
GRID_ZERO = "rgba(255, 255, 255, 0.4)"


def add_utc_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Cycle timezone" not in df.columns:
        return df

    tz = df["Cycle timezone"].astype(str).str.replace("UTC", "", regex=False).str.strip()
    tz = tz.replace(["nan", "NaN", "None", ""], "+00:00")
    offsets = pd.to_timedelta(tz + ":00")

    if "Cycle start time" in df.columns:
        df["Cycle start time"] = pd.to_datetime(df["Cycle start time"], errors="coerce")
        df["Cycle start time UTC"] = df["Cycle start time"] - offsets

    if "Cycle end time" in df.columns:
        df["Cycle end time"] = pd.to_datetime(df["Cycle end time"], errors="coerce")
        df["Cycle end time UTC"] = df["Cycle end time"] - offsets

    return df


def filter_time_window(df: pd.DataFrame, time_window: str) -> tuple[pd.DataFrame, int]:
    plot_df = df.copy()
    plot_df["Cycle start time UTC"] = pd.to_datetime(plot_df["Cycle start time UTC"])
    plot_df = plot_df.sort_values("Cycle start time UTC")

    if plot_df.empty:
        return plot_df, 1

    if time_window == "1M":
        start_date = plot_df["Cycle start time UTC"].max() - pd.DateOffset(months=1)
        window_size = 1
    elif time_window == "3M":
        start_date = plot_df["Cycle start time UTC"].max() - pd.DateOffset(months=3)
        window_size = 3
    elif time_window == "6M":
        start_date = plot_df["Cycle start time UTC"].max() - pd.DateOffset(months=6)
        window_size = 7
    elif time_window == "1Y":
        start_date = plot_df["Cycle start time UTC"].max() - pd.DateOffset(years=1)
        window_size = 14
    elif time_window == "2Y":
        start_date = plot_df["Cycle start time UTC"].max() - pd.DateOffset(years=2)
        window_size = 30
    else:
        start_date = plot_df["Cycle start time UTC"].min()
        window_size = 60

    plot_df = plot_df[plot_df["Cycle start time UTC"] >= start_date].copy()
    return plot_df, window_size


def load_and_prepare_data(data_dir: str | Path) -> dict[str, object]:
    data_dir = Path(data_dir)

    cycles_df = pd.read_csv(data_dir / "physiological_cycles.csv").drop(index=0).reset_index(drop=True)
    journal_entries_df = pd.read_csv(data_dir / "journal_entries.csv")
    workouts_df = pd.read_csv(data_dir / "workouts.csv")

    dfs = [cycles_df, journal_entries_df, workouts_df]
    for df in dfs:
        for col in ["Cycle start time", "Cycle end time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    cycles_df = add_utc_columns(cycles_df)
    journal_entries_df = add_utc_columns(journal_entries_df)
    workouts_df = add_utc_columns(workouts_df)

    valid_metrics_df = cycles_df.dropna(subset=["Blood oxygen %", "Skin temp (celsius)"])
    earliest_date = valid_metrics_df["Cycle start time UTC"].min()
    latest_date = valid_metrics_df["Cycle end time UTC"].max()

    cycles_df = cycles_df[cycles_df["Cycle start time UTC"] >= earliest_date].reset_index(drop=True)
    journal_entries_df = journal_entries_df[
        journal_entries_df["Cycle start time UTC"] >= earliest_date
    ].reset_index(drop=True)
    workouts_df = workouts_df[workouts_df["Cycle start time UTC"] >= earliest_date].reset_index(drop=True)

    cycles_df = cycles_df.sort_values("Cycle start time UTC").reset_index(drop=True)
    cycles_df["Next start"] = cycles_df["Cycle start time UTC"].shift(-1)
    cycles_df["Gap"] = cycles_df["Next start"] - cycles_df["Cycle end time UTC"]

    anomalous_gaps = cycles_df[cycles_df["Gap"] >= pd.Timedelta(hours=16)]
    drop_indices: set[int] = set()
    for idx in anomalous_gaps.index:
        drop_indices.update([idx, idx + 1, idx + 2])

    valid_drop_indices = [idx for idx in drop_indices if idx in cycles_df.index]
    cycles_df = cycles_df.drop(index=valid_drop_indices).reset_index(drop=True)
    cycles_df = cycles_df.drop(columns=["Next start", "Gap"])

    journal_entries_df = journal_entries_df[
        journal_entries_df["Cycle start time"] != pd.to_datetime("2026-04-14 22:51:32")
    ].copy()
    questions_to_keep = [
        "Consumed caffeine?",
        "Feeling sick or ill?",
        "Have any alcoholic drinks?",
        "Read (non-screened device) while in bed?",
        "Shared your bed?",
        "Viewed a screen device in bed?",
    ]
    journal_entries_df = journal_entries_df[
        journal_entries_df["Question text"].isin(questions_to_keep)
    ].copy()

    label_map = {
        "Consumed caffeine?": "Caffeine",
        "Feeling sick or ill?": "Sick",
        "Have any alcoholic drinks?": "Alcohol",
        "Read (non-screened device) while in bed?": "Read Book",
        "Shared your bed?": "Shared Bed",
        "Viewed a screen device in bed?": "Screen",
    }
    journal_entries_df["Question text"] = journal_entries_df["Question text"].replace(label_map)
    journal_entries_df["Answered yes"] = journal_entries_df["Answered yes"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )

    journal_wide_df = (
        journal_entries_df.pivot_table(
            index=["Cycle start time UTC", "Cycle end time UTC"],
            columns="Question text",
            values="Answered yes",
            aggfunc="first",
        )
        .reset_index()
    )

    active_cycles = workouts_df[["Cycle start time UTC"]].drop_duplicates().copy()
    active_cycles["Activity"] = True

    cycles_df = pd.merge(cycles_df, active_cycles, on="Cycle start time UTC", how="left")
    cycles_df["Activity"] = cycles_df["Activity"].fillna(False).astype(bool)
    cycles_df = pd.merge(
        cycles_df,
        journal_wide_df,
        on=["Cycle start time UTC", "Cycle end time UTC"],
        how="left",
    )

    numeric_cols = cycles_df.select_dtypes(include=["number"]).columns
    cycles_df[numeric_cols] = cycles_df[numeric_cols].interpolate(method="linear").round(2)
    cycles_df = cycles_df.ffill().bfill()

    numeric_variables = sorted(
        [
            col
            for col in cycles_df.select_dtypes(include=["number"]).columns
            if col not in {"Cycle timezone"}
        ]
    )

    return {
        "cycles_df": cycles_df,
        "journal_entries_df": journal_entries_df,
        "workouts_df": workouts_df,
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "anomalous_rows_dropped": len(valid_drop_indices),
        "numeric_variables": numeric_variables,
        "summary_stats": cycles_df[numeric_variables].agg(["mean", "std", "min", "max"]).round(2),
    }


def build_variable_diagnostics_figure(
    df: pd.DataFrame,
    col_name: str,
    time_window: str = "All Time",
    top_n_corrs: int = 10,
) -> tuple[go.Figure, pd.DataFrame]:
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in dataframe.")

    plot_df, window_size = filter_time_window(df, time_window)
    if plot_df.empty:
        raise ValueError(f"No data available for the '{time_window}' window.")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{col_name} Over Time ({time_window})",
            f"Distribution of {col_name}",
            f"Top {top_n_corrs} Correlations with {col_name}",
            f"Impact of Habits on {col_name}",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    smoothed = plot_df[col_name].rolling(window=window_size, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Cycle start time UTC"],
            y=plot_df[col_name],
            mode="lines",
            name="Raw",
            line={"color": RED_SOFT, "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Cycle start time UTC"],
            y=smoothed,
            mode="lines",
            name=f"{window_size}-Day Avg",
            line={"color": RED, "width": 3},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=plot_df[col_name],
            nbinsx=30,
            name="Distribution",
            marker_color=BLUE,
            opacity=0.85,
        ),
        row=1,
        col=2,
    )

    numeric_df = plot_df.select_dtypes(include=["number"])
    if col_name in numeric_df.columns and len(numeric_df.columns) > 1:
        corrs = numeric_df.corr(numeric_only=True)[col_name].drop(labels=[col_name]).dropna()
        top_corrs = corrs.abs().sort_values(ascending=False).head(top_n_corrs).index
        corr_data = corrs[top_corrs].sort_values()
        colors = ["#ca3f3f" if value < 0 else "#3f72ca" for value in corr_data.values]
        fig.add_trace(
            go.Bar(
                x=corr_data.values,
                y=corr_data.index,
                orientation="h",
                name="Correlation",
                marker_color=colors,
            ),
            row=2,
            col=1,
        )

    habit_cols = [col for col in HABIT_COLUMNS if col in plot_df.columns]
    if habit_cols and pd.api.types.is_numeric_dtype(plot_df[col_name]):
        habit_frames = []
        for habit in habit_cols:
            temp = plot_df[[col_name, habit]].dropna().copy()
            if temp.empty:
                continue
            temp["Habit"] = habit
            temp["Status"] = temp[habit].map({True: "Yes", False: "No"}).fillna(temp[habit].astype(str))
            habit_frames.append(temp[[col_name, "Habit", "Status"]])

        if habit_frames:
            melted = pd.concat(habit_frames, ignore_index=True)
            for status, color in [("No", "#e57373"), ("Yes", "#64b5f6")]:
                status_df = melted[melted["Status"] == status]
                if status_df.empty:
                    continue
                fig.add_trace(
                    go.Box(
                        x=status_df["Habit"],
                        y=status_df[col_name],
                        name=status,
                        marker_color=color,
                        line={"color": color},
                        fillcolor=RED_SOFT if status == "No" else BLUE_SOFT,
                        boxmean=True,
                    ),
                    row=2,
                    col=2,
                )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text=col_name, row=1, col=1)
    fig.update_xaxes(title_text=col_name, row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Correlation Coefficient (r)", range=[-1.1, 1.1], row=2, col=1)
    fig.update_xaxes(title_text="Habit", row=2, col=2)
    fig.update_yaxes(title_text=col_name, row=2, col=2)
    fig.add_vline(x=0, line_width=1, line_color="black", row=2, col=1)

    fig.update_layout(
        height=900,
        hovermode="closest",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 40, "r": 30, "t": 80, "b": 40},
    )

    return fig, plot_df


def prepare_modeling_results(df: pd.DataFrame) -> dict[str, object]:
    model_df = df.copy().sort_values("Cycle start time UTC").reset_index(drop=True)


    model_df['Time_Since_Last'] = model_df['Cycle start time UTC'].diff()
    gap_threshold = pd.Timedelta(hours=36)
    new_segment_mask = model_df['Time_Since_Last'] > gap_threshold
    model_df['Segment_ID'] = new_segment_mask.cumsum()

    cols_to_lag = [
        "Recovery score %",
        "Heart rate variability (ms)",
        "Day Strain",
        "Average HR (bpm)",
        "Sleep performance %",
        "Sleep consistency %",
        "Respiratory rate (rpm)",
        "Alcohol",
    ]

    lag_features: list[str] = []
    for col in cols_to_lag:
        if col in model_df.columns:
            for lag in [1, 2, 3]:
                new_col_name = f"{col}_lag{lag}"
                model_df[new_col_name] = model_df.groupby('Segment_ID')[col].shift(lag)
                lag_features.append(new_col_name)

    model_df['Target_Recovery_Change'] = model_df.groupby('Segment_ID')['Recovery score %'].shift(-1) - model_df['Recovery score %']
    model_df = model_df.dropna(subset=["Target_Recovery_Change"] + lag_features).reset_index(drop=True)

    base_features = [
        "Recovery score %",
        "Heart rate variability (ms)",
        "Day Strain",
        "Average HR (bpm)",
        "Sleep performance %",
        "Sleep consistency %",
        "Respiratory rate (rpm)",
        "Asleep duration (min)",
    ]
    features = [feature for feature in base_features + lag_features if feature in model_df.columns]

    X = model_df[features].astype(float)
    y = model_df["Target_Recovery_Change"].astype(float)

    split_idx = int(len(model_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    lr_model = LinearRegression()
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train_scaled, y_train)

    predictions = {
        "Random Forest": {
            "test": rf_model.predict(X_test),
            "train": rf_model.predict(X_train),
            "model": rf_model,
        },
        "Linear Regression (Normalized)": {
            "test": lr_model.predict(X_test_scaled),
            "train": lr_model.predict(X_train_scaled),
            "model": lr_model,
        },
        "Baseline: Predict Zero Change": {
            "test": np.zeros_like(y_test),
            "train": np.zeros_like(y_train),
            "model": None,
        },
    }

    results = []
    for name, preds in predictions.items():
        test_preds = preds["test"]
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, test_preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
                "R2": r2_score(y_test, test_preds),
            }
        )
    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True).round(3)

    return {
        "model_df": model_df,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "dates_train": model_df.loc[y_train.index, "Cycle start time UTC"],
        "dates_test": model_df.loc[y_test.index, "Cycle start time UTC"],
        "today_recovery_train": X_train["Recovery score %"],
        "today_recovery_test": X_test["Recovery score %"],
        "predictions": predictions,
        "results_df": results_df,
    }


def _feature_importance_frame(model, feature_names: list[str]) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame(columns=["Feature", "Value"])

    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame(
            {"Feature": feature_names, "Value": model.feature_importances_, "Metric": "Importance"}
        )
        return imp_df.sort_values("Value", ascending=True).tail(10)

    if hasattr(model, "coef_"):
        coefs = model.coef_
        if getattr(coefs, "ndim", 1) > 1:
            coefs = coefs[0]
        imp_df = pd.DataFrame({"Feature": feature_names, "Value": coefs, "Metric": "Coefficient"})
        imp_df["AbsValue"] = imp_df["Value"].abs()
        return imp_df.sort_values("AbsValue", ascending=False).head(10).sort_values("Value", ascending=True)

    return pd.DataFrame(columns=["Feature", "Value"])


def build_model_diagnostics_figure(
    y_true: pd.Series,
    y_pred: np.ndarray,
    today_recovery: pd.Series,
    dates: pd.Series,
    model_name: str,
    model=None,
    feature_names: list[str] | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    actual_next_day = (today_recovery + y_true).clip(1, 100)
    pred_next_day = (today_recovery + y_pred).clip(1, 100)
    residuals = y_true - y_pred

    diag_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Actual Change": y_true,
            "Predicted Change": y_pred,
            "Actual Next-Day Recovery": actual_next_day,
            "Predicted Next-Day Recovery": pred_next_day,
            "Residual": residuals,
        }
    ).sort_values("Date")
    diag_df["Residual 14-Day Avg"] = diag_df["Residual"].rolling(window=14, min_periods=1).mean()

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Next-Day Recovery Over Time",
            "Residuals Over Time",
            "Actual vs Predicted Change",
            "Residuals vs Predicted Change",
            "Distribution of Residuals",
            "Feature Importance",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=diag_df["Date"],
            y=diag_df["Actual Next-Day Recovery"],
            mode="lines",
            name="Actual",
            line={"color": BLUE, "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=diag_df["Date"],
            y=diag_df["Predicted Next-Day Recovery"],
            mode="lines",
            name="Predicted",
            line={"color": RED, "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=diag_df["Date"],
            y=diag_df["Residual"],
            mode="lines",
            name="Residuals",
            line={"color": BLUE_SOFT, "width": 2},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=diag_df["Date"],
            y=diag_df["Residual 14-Day Avg"],
            mode="lines",
            name="14-Day Avg",
            line={"color": RED, "width": 3},
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=diag_df["Actual Change"],
            y=diag_df["Predicted Change"],
            mode="markers",
            name="Predictions",
            marker={"color": BLUE, "opacity": 0.65},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[-100, 100],
            y=[-100, 100],
            mode="lines",
            name="Perfect Prediction",
            line={"color": RED, "dash": "dash"},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=diag_df["Predicted Change"],
            y=diag_df["Residual"],
            mode="markers",
            name="Residuals vs Predicted",
            marker={"color": BLUE, "opacity": 0.65},
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=diag_df["Residual"],
            nbinsx=30,
            name="Residual Distribution",
            marker_color=BLUE,
            opacity=0.85,
        ),
        row=3,
        col=1,
    )

    feature_df = _feature_importance_frame(model, feature_names or [])
    if not feature_df.empty:
        metric_name = feature_df["Metric"].iloc[0]
        bar_colors = [BLUE if value >= 0 else RED for value in feature_df["Value"]]
        fig.add_trace(
            go.Bar(
                x=feature_df["Value"],
                y=feature_df["Feature"],
                orientation="h",
                name=metric_name,
                marker={
                    "color": bar_colors,
                    "line": {"color": [RED if value >= 0 else BLUE for value in feature_df["Value"]], "width": 1},
                },
            ),
            row=3,
            col=2,
        )
        feature_title = "Feature Importance" if metric_name == "Importance" else "Linear Coefficients"
        fig.layout.annotations[5].update(text=feature_title)
    else:
        fig.layout.annotations[5].update(text="Feature Importance")

    fig.add_hline(y=0, line_dash="dash", line_color=GRID_ZERO, row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=GRID_ZERO, row=2, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="grey", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color=RED, row=3, col=1)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Recovery Score %", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    fig.update_xaxes(title_text="Actual Change", range=[-100, 100], row=2, col=1)
    fig.update_yaxes(title_text="Predicted Change", range=[-100, 100], row=2, col=1)
    fig.update_xaxes(title_text="Predicted Change", row=2, col=2)
    fig.update_yaxes(title_text="Residual", row=2, col=2)
    fig.update_xaxes(title_text="Residual", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Value", row=3, col=2)
    fig.update_yaxes(title_text="", row=3, col=2)
    fig.update_layout(
        height=1200,
        hovermode="closest",
        margin={"l": 40, "r": 30, "t": 80, "b": 40},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    return fig, diag_df
