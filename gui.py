import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import aaia

st.set_page_config(page_title="AAIA", layout="wide")

st.title("AAIA")
st.write("Upload CSV file, apply filters and run clustering")

ACCENT_COLOR = "#1f77b4"
PANEL_COLOR = "#f6f8fb"

uploaded_file = st.file_uploader("Select CSV file", type="csv")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.header("Filters")

    st.sidebar.subheader("AAIA parameters")
    max_iterations = int(
        st.sidebar.number_input(
            "Iterations",
            min_value=1,
            max_value=5000000,
            value=500,
            step=10,
        )
    )
    population_size = int(
        st.sidebar.number_input(
            "Population size",
            min_value=1,
            max_value=10000000,
            value=50,
            step=10,
        )
    )
    early_stop = int(
        st.sidebar.number_input(
            "Early stopping rounds (0=disabled)", min_value=0, value=0, step=1
        )
    )
    tol = float(
        st.sidebar.number_input(
            "Improvement tolerance (tol)", min_value=0.0, value=0.0, step=1e-6, format="%.6f"
        )
    )

    st.sidebar.subheader("Nearest results")
    nearest_mode = st.sidebar.selectbox(
        "Limit nearest samples by",
        ["Percentage", "Count"],
    )
    nearest_percent = int(
        st.sidebar.slider(
            "Nearest samples (%)",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            disabled=nearest_mode != "Percentage",
        )
    )
    nearest_count = int(
        st.sidebar.number_input(
            "Nearest samples (count)",
            min_value=1,
            value=20,
            step=1,
            disabled=nearest_mode != "Count",
        )
    )

    available_features = df_raw.columns.tolist()
    numeric_features = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Features to consider",
        available_features,
        default=numeric_features if numeric_features else available_features,
    )

    df_filtered = df_raw.copy()

    st.sidebar.subheader("Ranges and values")
    for col in selected_features:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df_raw[col] = pd.to_datetime(df_raw[col])

                min_date = df_raw[col].min().date()
                max_date = df_raw[col].max().date()

                date_range = st.sidebar.datetime_input(
                    f"Wybierz zakres dat: {col}",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )

                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    df_filtered = df_filtered[
                        (df_filtered[col].dt.date >= start_date)
                        & (df_filtered[col].dt.date <= end_date)
                    ]
                continue
            except:
                pass

        if pd.api.types.is_numeric_dtype(df_raw[col]):
            min_val = float(df_raw[col].min())
            max_val = float(df_raw[col].max())
            step = (max_val - min_val) / 100 if max_val != min_val else 0.1

            res = st.sidebar.slider(
                f"Range: {col}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                step=step,
            )
            df_filtered = df_filtered[
                (df_filtered[col] >= res[0]) & (df_filtered[col] <= res[1])
            ]

        else:
            options = df_raw[col].unique().tolist()
            selected_options = st.sidebar.multiselect(
                f"Values: {col}", options, default=options
            )

            df_filtered = df_filtered[df_filtered[col].isin(selected_options)]

    col_info, col_data = st.columns([1, 2])
    with col_info:
        st.metric("Row count after filtering", df_filtered.shape[0])
        st.write("Data preview")
    with col_data:
        st.dataframe(df_filtered.head(5), width="stretch")

    if st.button("Run AAIA clustering"):
        if df_filtered.shape[0] < 10:
            st.error("Too little data")
        else:
            X_df = df_filtered[selected_features].select_dtypes(include=[np.number])

            if X_df.empty:
                st.error("Selected features must be numerical")
            else:
                X = X_df.values
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

                with st.spinner("Calculation..."):
                    best_sol, centroid_history, iterations_done = aaia.find_solution(
                        X_scaled,
                        max_iterations=max_iterations,
                        population_size=population_size,
                        return_history=True,
                        early_stopping_rounds=early_stop,
                        tol=tol,
                    )

                    distances = np.sum(np.abs(X_scaled - best_sol), axis=1)
                    mean_distance = float(np.mean(distances))

                    total_samples = len(distances)
                    if nearest_mode == "Percentage":
                        top_n = max(1, int(np.ceil(total_samples * nearest_percent / 100)))
                    else:
                        top_n = min(nearest_count, total_samples)

                    nearest_indices = np.argsort(distances)[:top_n]
                    nearest_df = df_filtered.iloc[nearest_indices].copy()
                    nearest_df["manhattan_distance"] = distances[nearest_indices]
                    nearest_df = nearest_df.sort_values("manhattan_distance")
                    centroid_history_df = pd.DataFrame(
                        centroid_history,
                        columns=[f"centroid_{feature}" for feature in X_df.columns],
                    )
                    centroid_history_df.insert(
                        0,
                        "iteration",
                        np.arange(len(centroid_history_df)),
                    )
                    centroid_history_df = pd.DataFrame(
                        centroid_history,
                        columns=[f"centroid_{feature}" for feature in X_df.columns],
                    )
                    centroid_history_df.insert(0, "iteration", np.arange(len(centroid_history_df)))

                    # store AAIA outputs in session state so plots can update without rerunning
                    st.session_state["aaia"] = {
                        "X_scaled": X_scaled,
                        "X_df": X_df,
                        "df_results": df_filtered.reset_index(drop=True),
                        "best_sol": best_sol,
                        "distances": distances,
                        "centroid_history": centroid_history,
                        "iterations_done": iterations_done,
                        "total_samples": total_samples,
                    }

                    st.success("Clustering finished")
    
    # Render results from session state so changing plot controls doesn't re-run AAIA
    if "aaia" in st.session_state:
        data = st.session_state["aaia"]
        X_scaled = data["X_scaled"]
        X_df = data["X_df"]
        df_results = data.get("df_results", df_filtered.reset_index(drop=True))
        best_sol = data["best_sol"]
        distances = data["distances"]
        centroid_history = data["centroid_history"]
        iterations_done = data.get("iterations_done", None)
        total_samples = data.get("total_samples", len(distances))

        # ensure all arrays/dataframes are aligned to avoid out-of-bounds after filter changes
        aligned_n = min(len(distances), len(X_scaled), len(df_results))
        if aligned_n == 0:
            st.warning("No stored samples to display. Run AAIA clustering again.")
            st.stop()

        distances = distances[:aligned_n]
        X_scaled = X_scaled[:aligned_n]
        df_results = df_results.iloc[:aligned_n].reset_index(drop=True)
        total_samples = aligned_n

        # compute current top_n
        if nearest_mode == "Percentage":
            top_n = max(1, int(np.ceil(total_samples * nearest_percent / 100)))
        else:
            top_n = min(nearest_count, total_samples)

        nearest_indices = np.argsort(distances)[:top_n]
        nearest_df = df_results.iloc[nearest_indices].copy()
        nearest_df["manhattan_distance"] = distances[nearest_indices]
        nearest_df = nearest_df.sort_values("manhattan_distance")

        centroid_history_df = pd.DataFrame(
            centroid_history, columns=[f"centroid_{feature}" for feature in X_df.columns]
        )
        centroid_history_df.insert(0, "iteration", np.arange(len(centroid_history_df)))

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Mean Manhattan distance to AAIA solution", f"{float(np.mean(distances)):.4f}")
        with metric_col2:
            st.metric("Nearest samples returned", f"{top_n} / {total_samples}")
        with metric_col3:
            st.metric("Iterations run", f"{iterations_done}")

        center_df = pd.DataFrame({"feature": X_df.columns, "center_scaled": best_sol})
        st.write("AAIA solution (in scaled feature space)")
        st.dataframe(center_df, width="stretch")

        st.write("Nearest samples to AAIA solution")
        st.dataframe(nearest_df, width="stretch")

        st.write("Centroid change by iteration")
        st.dataframe(centroid_history_df, width="stretch", height=280)

        feature_options = list(X_df.columns)
        if len(feature_options) >= 2:
            controls_col1, controls_col2, controls_col3 = st.columns([1, 1, 1])
            with controls_col1:
                x_feat = st.selectbox("X feature", feature_options, index=0, key="x_feat")
            with controls_col2:
                y_feat = st.selectbox("Y feature", feature_options, index=1, key="y_feat")
            with controls_col3:
                only_nearest = st.checkbox("Show only nearest samples", value=False, key="only_nearest")

            xi = feature_options.index(x_feat)
            yi = feature_options.index(y_feat)
            plot_idx = nearest_indices if only_nearest else np.arange(len(distances))

            plot_col1, plot_col2 = st.columns([1.25, 1])

            with plot_col1:
                fig_hist, ax_hist = plt.subplots(figsize=(5.2, 3.4))
                ax_hist.set_facecolor(PANEL_COLOR)
                ax_hist.hist(distances, bins=18, color=ACCENT_COLOR, alpha=0.85, edgecolor="white")
                ax_hist.set_title("Distance distribution to AAIA solution", fontsize=10, pad=8)
                ax_hist.set_xlabel("Manhattan distance")
                ax_hist.set_ylabel("Count")
                ax_hist.spines["top"].set_visible(False)
                ax_hist.spines["right"].set_visible(False)
                ax_hist.grid(axis="y", alpha=0.2)
                fig_hist.tight_layout()
                st.pyplot(fig_hist)

            with plot_col2:
                fig_scatter, ax_scatter = plt.subplots(figsize=(4.0, 3.0))
                sc = ax_scatter.scatter(
                    X_scaled[plot_idx, xi],
                    X_scaled[plot_idx, yi],
                    c=distances[plot_idx],
                    cmap="viridis",
                    s=14,
                    alpha=0.9,
                    linewidths=0,
                )
                ax_scatter.scatter(
                    X_scaled[nearest_indices, xi],
                    X_scaled[nearest_indices, yi],
                    facecolors="none",
                    edgecolors="red",
                    s=42,
                    linewidths=1.0,
                )
                ax_scatter.scatter(
                    best_sol[xi], best_sol[yi], marker="*", color="gold", s=90, edgecolors="k"
                )
                ax_scatter.set_xlabel(x_feat)
                ax_scatter.set_ylabel(y_feat)
                ax_scatter.set_title("Scatter by distance", fontsize=10, pad=8)
                ax_scatter.spines["top"].set_visible(False)
                ax_scatter.spines["right"].set_visible(False)
                fig_scatter.colorbar(sc, ax=ax_scatter, label="distance", fraction=0.045, pad=0.03)
                fig_scatter.tight_layout()
                st.pyplot(fig_scatter)

            csv_data = nearest_df.to_csv(index=False)
            st.download_button(
                "Download nearest samples as CSV",
                data=csv_data,
                file_name="nearest_samples.csv",
                mime="text/csv",
            )
        else:
            st.info("Select at least two features to enable scatter plot")

else:
    st.info("Upload CSV file to commence")
