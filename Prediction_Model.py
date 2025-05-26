# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from joblib import load
from datetime import datetime, timedelta
import yfinance as yf
from tensorflow.keras.losses import MeanAbsoluteError
import plotly.graph_objects as go

# --- Konstanta fitur ---
FEATURES = ['Open', 'Volume', 'ROA', 'ROE', 'DER']
TARGET = ['Close']

# --- Fungsi: Load model dan scaler ---


def load_model_and_scalers():
    model = load_model("lstm_model.h5", custom_objects={
                       "mae": MeanAbsoluteError()})
    scaler_input = load("scaler_input2.pkl")
    scaler_close = load("scaler_close2.pkl")
    return model, scaler_input, scaler_close

# --- Fungsi: Ambil data saham dari Yahoo Finance ---


@st.cache_data
def load_data(ticker):
    START = "2023-12-31"
    TODAY = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index, errors='coerce')
    return data[data.index >= pd.to_datetime("2023-12-31")]

# --- Fungsi: Ambil dan ubah income statement ---

@st.cache_data
def income_statement(ticker):
    stock = yf.Ticker(ticker)
    income = stock.quarterly_financials
    income = income.rename(columns={pd.to_datetime(
        "2023-09-30"): pd.to_datetime("2023-12-31")})
    income["2023-12-31"] = 5600000000000.0
    income = income.T[['Net Income', 'Total Revenue']]
    return income[income.index >= pd.to_datetime("2023-12-31")]

# --- Fungsi: Ambil dan ubah balance sheet ---

@st.cache_data
def balance_sheet(ticker):
    stock = yf.Ticker(ticker)
    balance = stock.balance_sheet.T
    balance = balance[['Stockholders Equity',
                       'Long Term Debt', 'Total Assets', 'Total Liabilities Net Minority Interest']]
    return balance[balance.index >= pd.to_datetime("2023-12-31")]


def ratio(x, y):
    return x/y

# --- Fungsi: Prediksi harga saham ke depan ---


def predict_future_prices(model, scaler_input, scaler_close, data, window_size, days_ahead):
    if 'Volume' not in data.columns:
        st.error("❌ Kolom 'Volume' tidak ada dalam data saham. Mohon cek kembali.")
        return []

    valid_features = [f for f in FEATURES if f in data.columns]
    if not valid_features:
        st.error("❌ Tidak ada fitur yang valid untuk prediksi.")
        return []

    scaled_data = scaler_input.transform(data[valid_features])
    last_known_data = scaled_data[-window_size:]

    predictions = []
    for _ in range(days_ahead):
        input_data = last_known_data.reshape(
            1, window_size, len(valid_features))
        predicted_scaled = model.predict(input_data, verbose=0)[0, 0]
        predictions.append(predicted_scaled)

        next_input = last_known_data[-1].copy()
        next_input[0] = predicted_scaled
        last_known_data = np.vstack((last_known_data[1:], next_input))

    return scaler_close.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# --- Aplikasi Streamlit utama ---


def main():
    st.set_page_config(page_title="Stocks Market", page_icon=":material/edit:")

    # Side bar
    # news = st.Page("Pages/News.py", title="📰News", )
    # st.sidebar.page_link("pages/News.py", label="📰Berita")
    
    # Financial = st.Page("Pages/Financial.py", title="💰Financial")
    # st.sidebar.page_link("Pages/Financial.py", label="💰 Keuangan")

    st.title("📈 Prediksi Harga Saham PT Telkom Indonesia")
    st.write(
        "Prediksi harga saham berdasarkan indikator fundamental dan historis dengan BI-LSTM.")

    stocks = ("TLKM.JK",)
    selected_stock = st.selectbox("Pilih saham", stocks)

    # st.text("🔄 Memuat data...")
    data = load_data(selected_stock)
    st.success("✅ Data berhasil dimuat!")

    st.subheader("📊 Data Saham Historis")
    st.dataframe(data.tail(len(data)), height=250)

    # ✅ Tambahkan grafik harga saham
    st.subheader("📉 Grafik Candlestick Harga Saham")
    candlestick_fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    candlestick_fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        xaxis_rangeslider_visible=False,
        height=500
    )
    st.plotly_chart(candlestick_fig, use_container_width=True)
    

    st.subheader(f"💰 Income Statement: {selected_stock}")
    try:
        income_data = income_statement(selected_stock)
        st.dataframe(income_data, height=150)
    except Exception as e:
        st.error(f"Gagal memuat income statement: {e}")

    st.subheader(f"🏦 Balance Sheet: {selected_stock}")
    try:
        balance_data = balance_sheet(selected_stock)
        st.dataframe(balance_data, height=150)
    except Exception as e:
        st.error(f"Gagal memuat balance sheet: {e}")

    merged_data = pd.concat([data, income_data, balance_data]).sort_index()
    merged_data = merged_data.fillna(method="ffill")
    merged_data = merged_data[~merged_data.index.duplicated(keep='last')]
    dataset_lama = merged_data[merged_data.index >=
                               pd.to_datetime("2024-01-01")]

    # Penggabungan data
    st.subheader("🧾 Data Gabungan")
    st.dataframe(dataset_lama.tail(), height=250)
    
    # Perumusan ratio
    st.subheader("📐 Rasio Keuangan")
    dataset_lama["DER"] = ratio(
        dataset_lama['Long Term Debt'], dataset_lama['Stockholders Equity'])
    dataset_lama["ROA"] = ratio(
        dataset_lama["Net Income"], dataset_lama["Total Assets"])
    dataset_lama["ROE"] = ratio(
        dataset_lama["Net Income"], dataset_lama["Stockholders Equity"])
    st.dataframe(dataset_lama[["DER", "ROA", "ROE"]].tail(), height=150)

    dataset_baru = dataset_lama[FEATURES]
    st.subheader("🔧 Dataset untuk prediksi")
    st.dataframe(dataset_baru.tail(len(dataset_baru)), height=150)

#   Prediksi model
    model, scaler_input, scaler_close = load_model_and_scalers()
    window_size = 30
    days_ahead = st.number_input(
        "📅 Prediksi berapa hari ke depan?", min_value=1, max_value=60, value=7)

    if st.button("📈 Prediksi Harga"):
        if len(dataset_baru) < window_size:
            st.error(
                f"❌ Data tidak cukup untuk prediksi (minimal {window_size} baris).")
            return

        predictions = predict_future_prices(
            model, scaler_input, scaler_close, dataset_baru, window_size, days_ahead)
        if len(predictions) == 0:
            return

        last_date = dataset_baru.index[-1]
        future_dates = [last_date + timedelta(days=i)
                        for i in range(1, days_ahead + 1)]
        prediction_df = pd.DataFrame(
            {'Tanggal': future_dates, 'Harga Prediksi': predictions})

        st.subheader("🔮 Hasil Prediksi")
        st.write(prediction_df)

        st.subheader("📉 Grafik Prediksi Harga (Interaktif)")

        fig = go.Figure()

        # Tambahkan harga historis
        fig.add_trace(go.Scatter(
            x=dataset_baru.index,
            y=dataset_lama.loc[dataset_baru.index, 'Close'],
            mode='lines',
            name='Harga Historis',
            line=dict(color='blue')
        ))

        # Tambahkan harga prediksi
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Harga Prediksi',
            line=dict(color='orange', dash='dash')
        ))

        fig.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Harga (IDR)",
            hovermode="x unified",
            height=500,
            xaxis_rangeslider_visible=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
