import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import aaia

st.set_page_config(page_title="AAIA Dashboard", layout="wide")

st.title("AAIA Dashboard")
st.write("Upload CSV file, apply filters and run classification")

uploaded_file = st.file_uploader("Select CSV file", type="csv")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.header("Filters")

    target_column = st.sidebar.selectbox(
        "Select label",
        df_raw.columns,
        index=len(df_raw.columns) - 1,
    )

    available_features = [col for col in df_raw.columns if col != target_column]
    selected_features = st.sidebar.multiselect(
        "Features to consider", available_features, default=available_features
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
        st.dataframe(df_filtered.head(5), use_container_width=True)

    if st.button("Run AAIA classification"):
        if df_filtered.shape[0] < 10:
            st.error("Too little data")
        else:
            y = pd.factorize(df_filtered[target_column])[0]

            X_df = df_filtered[selected_features].select_dtypes(include=[np.number])

            if X_df.empty:
                st.error("Selected features must be numerical")
            else:
                X = X_df.values
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3
                )

                with st.spinner("Calculation..."):
                    best_sol = aaia.find_solution(
                        X_train, max_iterations=500, population_size=50
                    )
                    labels, _, _, _ = aaia.classify(X_train, y_train, X_test, best_sol)

                    acc = accuracy_score(y_test, labels)

                    st.success(f"Accuracy result: {acc:.4f}")

                    fig, ax = plt.subplots()
                    ax.bar(["Accuracy"], [acc], color="skyblue")
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)

else:
    st.info("Upload CSV file to commence")
