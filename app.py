import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---- Settings taken from the training notebook ----
LOOKBACK = 30          # 30 previous days
N_FEATURES = 1         # Sales only
MODEL_PATH = "lstm_sales_model_20261115_112956.pkl"
SCALER_PATH = "lstm_sales_scaler_20261115_112956.pkl"

st.set_page_config(page_title="Store Sales Forecast", layout="centered")
st.title("Store Sales Forecast (LSTM)")
st.caption("Uses the previous 30 days of sales to predict upcoming days.")


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(
        "Could not load the model/scaler. Make sure both .pkl files are in the "
        "same folder as app.py and that the TensorFlow/Keras and scikit-learn "
        f"versions match the ones used in Colab.\n\nDetails: {e}"
    )
    st.stop()


def forecast(history, steps):
    """Recursive forecast: each prediction is fed back as input for the next day."""
    window = list(history[-LOOKBACK:])
    preds = []
    for _ in range(steps):
        x = np.array(window[-LOOKBACK:], dtype="float32").reshape(-1, 1)
        x = scaler.transform(x).reshape(1, LOOKBACK, N_FEATURES)
        p_scaled = model.predict(x, verbose=0)
        p = float(scaler.inverse_transform(p_scaled)[0][0])
        p = max(p, 0.0)  # sales can't be negative
        preds.append(p)
        window.append(p)
    return preds


mode = st.radio("Input method", ["Upload CSV", "Paste values"], horizontal=True)
steps = st.slider("Days to forecast", 1, 14, 7)

history, last_date = None, None

if mode == "Upload CSV":
    st.write("CSV needs a **Sales** column. A **Date** column is optional but recommended.")
    up = st.file_uploader("Daily sales CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "Sales" not in df.columns:
            st.error("The CSV must have a 'Sales' column.")
        else:
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")
                last_date = df["Date"].iloc[-1]
            history = df["Sales"].astype(float).tolist()
else:
    raw = st.text_area(
        f"Enter the last {LOOKBACK} daily sales values, comma separated",
        placeholder="5530, 4327, 4486, ...",
    )
    if raw.strip():
        try:
            history = [float(v) for v in raw.replace("\n", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Please enter only numbers separated by commas.")

if history is not None:
    if len(history) < LOOKBACK:
        st.error(f"Need at least {LOOKBACK} days of sales; got {len(history)}.")
    elif st.button("Forecast"):
        preds = forecast(history, steps)

        if last_date is not None:
            future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps)
            past_idx = pd.date_range(end=last_date, periods=LOOKBACK)
        else:
            future_idx = range(LOOKBACK, LOOKBACK + steps)
            past_idx = range(LOOKBACK)

        out = pd.DataFrame({"Predicted Sales": np.round(preds, 0)}, index=future_idx)
        st.subheader("Forecast")
        st.dataframe(out)

        chart = pd.concat(
            [
                pd.Series(history[-LOOKBACK:], index=past_idx, name="Actual (last 30 days)"),
                pd.Series(preds, index=future_idx, name="Forecast"),
            ],
            axis=1,
        )
        st.line_chart(chart)

        st.info(
            "Note: this model was trained on a single store and had a validation "
            "MAE of about 1,190 sales units. It does not predict closed days "
            "(zero sales) well, so treat forecasts as rough estimates."
        )
