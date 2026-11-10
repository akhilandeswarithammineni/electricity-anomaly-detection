import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Smart Building Energy Monitor",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Building Energy Monitor")
st.subheader("Detection of Abnormal Electricity Consumption")
st.write(
    "Upload electricity data to identify unusual power "
    "consumption patterns using machine learning."
)

st.info(
    "Anomaly alerts indicate unusual readings, not confirmed "
    "electrical faults or electricity theft."
)

uploaded_file = st.file_uploader(
    "Upload electricity data",
    type=["csv", "txt"]
)

contamination = st.sidebar.slider(
    "Expected proportion of anomalies",
    min_value=0.01,
    max_value=0.10,
    value=0.02,
    step=0.01
)

if uploaded_file is not None:

    try:
        # Read either a comma-separated CSV or a semicolon file
        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            na_values=["?"]
        )

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        # Support the original UCI dataset
        if "Global_active_power" in df.columns:

            df["Global_active_power"] = pd.to_numeric(
                df["Global_active_power"],
                errors="coerce"
            )

            if "Date" in df.columns and "Time" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["Date"].astype(str) + " "
                    + df["Time"].astype(str),
                    dayfirst=True,
                    errors="coerce"
                )

            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["datetime"],
                    errors="coerce"
                )

            else:
                raise ValueError(
                    "The original dataset needs Date and Time columns."
                )

            df = df.dropna(
                subset=["datetime", "Global_active_power"]
            )

            df = df.set_index("datetime").sort_index()

            # Convert minute readings to hourly averages
            hourly = df["Global_active_power"].resample("h").mean()
            data = hourly.dropna().to_frame("power_kw")

        else:
            # Support cleaned CSV files and ordinary CSV data
            power_candidates = [
                "power_kw",
                "Global_active_power",
                "power",
                "power_consumption",
                "consumption",
                "electricity_consumption"
            ]

            power_column = next(
                (
                    col for col in power_candidates
                    if col in df.columns
                ),
                None
            )

            if power_column is None:
                raise ValueError(
                    "Power column not found. Use power_kw, "
                    "power, consumption, or Global_active_power."
                )

            df[power_column] = pd.to_numeric(
                df[power_column],
                errors="coerce"
            )

            time_column = next(
                (
                    col for col in [
                        "datetime", "timestamp", "DateTime", "date"
                    ]
                    if col in df.columns
                ),
                None
            )

            if time_column:
                df["datetime"] = pd.to_datetime(
                    df[time_column],
                    errors="coerce",
                    dayfirst=True
                )

                df = df.dropna(
                    subset=["datetime", power_column]
                )

                df = df.sort_values("datetime")

                data = df.set_index("datetime")[
                    [power_column]
                ].rename(
                    columns={power_column: "power_kw"}
                )

            else:
                # If no timestamps exist, use row order.
                data = pd.DataFrame({
                    "power_kw": df[power_column]
                }).dropna()

                data.index = pd.RangeIndex(len(data))
                data.index.name = "reading_number"

        # Validate data before training
        data["power_kw"] = pd.to_numeric(
            data["power_kw"],
            errors="coerce"
        )

        data = data.replace(
            [np.inf, -np.inf], np.nan
        ).dropna(subset=["power_kw"])

        data = data[
            data["power_kw"] >= 0
        ].copy()

        if len(data) < 20:
            raise ValueError(
                "Please upload at least 20 valid power readings."
            )

        # Create time features
        if isinstance(data.index, pd.DatetimeIndex):
            data["hour"] = data.index.hour
            data["day_of_week"] = data.index.dayofweek
        else:
            data["hour"] = 0
            data["day_of_week"] = 0

        features = data[
            ["power_kw", "hour", "day_of_week"]
        ]

        # Train the anomaly detection model
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )

        data["prediction"] = model.fit_predict(features)

        data["status"] = data["prediction"].map({
            1: "Normal",
            -1: "Abnormal"
        })

        abnormal = data[
            data["status"] == "Abnormal"
        ]

        # Dashboard metrics
        st.success("Analysis completed.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total readings",
            f"{len(data):,}"
        )

        col2.metric(
            "Normal readings",
            f"{(data['status'] == 'Normal').sum():,}"
        )

        col3.metric(
            "Abnormal readings",
            f"{len(abnormal):,}"
        )

        col4.metric(
            "Average power",
            f"{data['power_kw'].mean():.2f} kW"
        )

        # Consumption chart
        st.subheader("Electricity Consumption Over Time")

        chart_data = data[["power_kw"]].copy()
        chart_data["status"] = data["status"]

        st.line_chart(chart_data["power_kw"])

        # Anomaly scatter plot
        st.subheader("Detected Abnormal Readings")

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            data.index,
            data["power_kw"],
            label="Power consumption",
            linewidth=0.8
        )

        ax.scatter(
            abnormal.index,
            abnormal["power_kw"],
            color="red",
            label="Anomaly",
            s=20
        )

        ax.set_xlabel("Time or reading number")
        ax.set_ylabel("Average power (kW)")
        ax.legend()
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Display detailed results
        st.subheader("Analysis Results")

        st.dataframe(
            data.reset_index(),
            use_container_width=True
        )

        st.subheader("Abnormal Readings Report")

        if len(abnormal) > 0:
            st.dataframe(
                abnormal.reset_index(),
                use_container_width=True
            )
        else:
            st.write("No abnormal readings were flagged.")

        # Download all results
        all_csv = data.reset_index().to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download All Results",
            data=all_csv,
            file_name="electricity_results.csv",
            mime="text/csv"
        )

        # Download anomalies separately
        abnormal_csv = abnormal.reset_index().to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Abnormal Readings",
            data=abnormal_csv,
            file_name="abnormal_readings.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Could not process this file: {error}")

else:
    st.write(
        "Upload your electricity CSV or the original "
        "UCI household power consumption TXT file to begin."
    )

    st.markdown(
        """
        **Required data:** a numeric electricity power column.

        Supported power column names include `power_kw`,
        `power`, `consumption`, and `Global_active_power`.

        If timestamps are available, include them for time-based
        visualizations.
        """
    )

st.caption(
    "Academic project prototype | Isolation Forest "
    "| Results require validation before real-world use"
)
