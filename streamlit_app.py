# streamlit_app.py
# author: Austin Juhl

from __future__ import annotations

from pathlib import Path

import streamlit as st

from whoop_analysis import (
    TIME_WINDOWS,
    build_model_diagnostics_figure,
    build_variable_diagnostics_figure,
    load_and_prepare_data,
    prepare_modeling_results,
)


st.set_page_config(
    page_title="WHOOP Recovery Explorer",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def get_analysis_data():
    return load_and_prepare_data(DATA_DIR)


@st.cache_data(show_spinner=False)
def get_model_data():
    analysis_data = get_analysis_data()
    return prepare_modeling_results(analysis_data["cycles_df"])


def render_home_page(analysis_data: dict[str, object]) -> None:
    cycles_df = analysis_data["cycles_df"]
    summary_stats = analysis_data["summary_stats"]

    st.title("WHOOP Recovery Explorer")
    st.markdown(
        """
        This app turns my WHOOP Data into an explorable dashboard.
        It cleans and aligns the cycle, sleep, journal, and workout data, then lets you:

        - explore how a chosen physiological variable behaves across different time windows
        - compare distributions, rolling trends, correlations, and habit effects
        - inspect next-day recovery model performance with side-by-side diagnostics
        """
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Cycles analyzed", f"{len(cycles_df):,}")
    metric_cols[1].metric("Variables available", f"{len(analysis_data['numeric_variables']):,}")
    metric_cols[2].metric("Tracking start", analysis_data["earliest_date"].strftime("%Y-%m-%d"))
    metric_cols[3].metric("Tracking end", analysis_data["latest_date"].strftime("%Y-%m-%d"))
    metric_cols[4].metric("Anomalous rows removed", f"{analysis_data['anomalous_rows_dropped']:,}")

    st.subheader("Project Flow")
    st.markdown(
        """
        1. Raw WHOOP exports are standardized to UTC and trimmed to the earliest date with valid blood oxygen and skin temperature measurements.
        2. Long tracking gaps are treated as anomalies and removed from the main physiological timeline.
        3. Journal habits and workout activity are merged into the cycle-level table.
        4. Numeric features are interpolated, remaining gaps are forward/back filled, and the cleaned data is used for both EDA and modeling.
        """
    )

    st.subheader("Quick Summary")
    st.dataframe(summary_stats.transpose(), use_container_width=True)

    with st.expander("Preview cleaned cycle data"):
        st.dataframe(cycles_df.head(50), use_container_width=True)


def render_data_analysis_page(analysis_data: dict[str, object]) -> None:
    cycles_df = analysis_data["cycles_df"]
    numeric_variables = analysis_data["numeric_variables"]

    st.title("Data Analysis")
    st.markdown(
        "Choose a metric, time window, and the number of top correlations to explore."
    )

    controls = st.columns([1.4, 1, 1])
    selected_variable = controls[0].selectbox(
        "Metric",
        options=numeric_variables,
        index=numeric_variables.index("Recovery score %") if "Recovery score %" in numeric_variables else 0,
    )
    selected_window = controls[1].selectbox("Time window", options=TIME_WINDOWS, index=len(TIME_WINDOWS) - 1)
    top_n_corrs = controls[2].slider("Top correlations", min_value=5, max_value=20, value=10, step=1)

    fig, plot_df = build_variable_diagnostics_figure(
        cycles_df,
        selected_variable,
        time_window=selected_window,
        top_n_corrs=top_n_corrs,
    )
    st.plotly_chart(fig, use_container_width=True)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Rows in window", f"{len(plot_df):,}")
    summary_cols[1].metric("Mean", f"{plot_df[selected_variable].mean():.2f}")
    summary_cols[2].metric("Std dev", f"{plot_df[selected_variable].std():.2f}")
    summary_cols[3].metric("Latest value", f"{plot_df[selected_variable].iloc[-1]:.2f}")

    with st.expander("Filtered data preview"):
        preview_cols = ["Cycle start time UTC", selected_variable]
        available_preview_cols = [col for col in preview_cols if col in plot_df.columns]
        st.dataframe(plot_df[available_preview_cols].tail(100), use_container_width=True)


def render_model_page(model_data: dict[str, object]) -> None:
    st.title("Model Diagnostics")
    st.markdown(
        "The models predict next-day change in recovery score from current physiological features and lagged signals. Select a model and split to view its performance."
    )

    st.subheader("Test Set Comparison")
    st.dataframe(model_data["results_df"], use_container_width=True)

    controls = st.columns([1.5, 1])
    model_name = controls[0].selectbox("Model", options=list(model_data["predictions"].keys()))
    dataset_split = controls[1].radio("Split", options=["test", "train"], horizontal=True, format_func=str.title)

    prediction_bundle = model_data["predictions"][model_name]
    model = prediction_bundle["model"]
    y_true = model_data["y_test"] if dataset_split == "test" else model_data["y_train"]
    y_pred = prediction_bundle[dataset_split]
    today_recovery = (
        model_data["today_recovery_test"] if dataset_split == "test" else model_data["today_recovery_train"]
    )
    dates = model_data["dates_test"] if dataset_split == "test" else model_data["dates_train"]

    fig, diag_df = build_model_diagnostics_figure(
        y_true=y_true,
        y_pred=y_pred,
        today_recovery=today_recovery,
        dates=dates,
        model_name=f"{model_name} ({dataset_split.title()})",
        model=model,
        feature_names=model_data["features"],
    )
    st.plotly_chart(fig, use_container_width=True)

    residual_mae = (diag_df["Residual"].abs()).mean()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Observations", f"{len(diag_df):,}")
    metric_cols[1].metric("Mean abs residual", f"{residual_mae:.2f}")
    metric_cols[2].metric("Mean predicted change", f"{diag_df['Predicted Change'].mean():.2f}")
    metric_cols[3].metric("Mean actual change", f"{diag_df['Actual Change'].mean():.2f}")

    with st.expander("Prediction table"):
        st.dataframe(diag_df, use_container_width=True)


def main() -> None:
    analysis_data = get_analysis_data()
    model_data = get_model_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Diagnostics"])

    if page == "Home":
        render_home_page(analysis_data)
    elif page == "Data Analysis":
        render_data_analysis_page(analysis_data)
    else:
        render_model_page(model_data)


if __name__ == "__main__":
    main()
