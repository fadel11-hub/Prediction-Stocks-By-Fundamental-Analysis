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
import plotly.express as px

FEATURES = ['Open', 'High', 'Low', 'ROA', 'ROE', 'DER']
TARGET = ['Close']

def load_model_and_scalers():
    model = load_model("bilstm_model_3.h5", custom_objects={"mae": MeanAbsoluteError()})
    scaler_input = load("scaler_input_3.pkl")
    scaler_close = load("scaler_close_3.pkl")
    # st.text(model.summary())
    return model, scaler_input, scaler_close

@st.cache_data
def load_data(ticker):
    START = "2023-12-31"
    TODAY = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index, errors='coerce')
    return data[data.index >= pd.to_datetime("2023-12-31")]

@st.cache_data
def income_statement(ticker):
    stock = yf.Ticker(ticker)
    try:
        income = stock.quarterly_financials
        if income.empty:
            st.warning(f"Tidak ada data Income Statement untuk {ticker}.")
            return pd.DataFrame()

        # Koreksi label kolom
        if pd.to_datetime("2023-09-30") in income.columns:
            income.rename(columns={pd.to_datetime("2023-09-30"): pd.to_datetime("2023-12-31")}, inplace=True)

        # Override nilai jika perlu
        income["2023-12-31"] = 5600000000000.0

        # Transpose dan proses
        income = income.T[['Net Income', 'Total Revenue']]
        income.index = pd.to_datetime(income.index, errors='coerce')
        income['Year'] = income.index.year
        income['Quarter'] = income.index.quarter
        income['Year-Quarter'] = income['Year'].astype(str) + "Q" + income['Quarter'].astype(str)

        # Filter awal data
        return income[income.index >= pd.to_datetime("2023-12-31")]

    except Exception as e:
        st.error(f"Gagal memuat income statement: {e}")
        return pd.DataFrame()


@st.cache_data
def balance_sheet(ticker):
    try:
        stock = yf.Ticker(ticker)
        balance = stock.balance_sheet.T
        if balance.empty:
            st.warning(f"Tidak ada data Balance Sheet untuk {ticker}.")
            return pd.DataFrame()

        # Ambil kolom penting
        balance = balance[['Stockholders Equity', 'Long Term Debt', 'Total Assets', 'Total Liabilities Net Minority Interest']]
        balance.index = pd.to_datetime(balance.index, errors='coerce')
        balance['Year'] = balance.index.year
        balance['Quarter'] = balance.index.quarter
        balance['Label'] = balance['Year'].astype(str) + "Q" + balance['Quarter'].astype(str)

        # Filter awal data (sama seperti income)
        return balance[balance.index >= pd.to_datetime("2023-12-31")]

    except Exception as e:
        st.error(f"Gagal memuat balance sheet: {e}")
        return pd.DataFrame()


def ratio(x, y):
    return x / y

def predict_future_prices(model, scaler_input, scaler_close, data, window_size, days_ahead):
    valid_features = [f for f in FEATURES if f in data.columns]
    if not valid_features:
        st.error("❌ Tidak ada fitur yang valid untuk prediksi.")
        return []

    scaled_data = scaler_input.transform(data[valid_features])
    last_known_data = scaled_data[-window_size:]

    predictions = []
    for _ in range(days_ahead):
        input_data = last_known_data.reshape(1, window_size, len(valid_features))
        predicted_scaled = model.predict(input_data, verbose=0)[0, 0]
        predictions.append(predicted_scaled)
        next_input = last_known_data[-1].copy()
        next_input[0] = predicted_scaled
        last_known_data = np.vstack((last_known_data[1:], next_input))

    return scaler_close.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def main():
    st.set_page_config(page_title="Stocks Market", page_icon=":material/edit:")
    st.title("📈 Prediksi Harga Saham PT Telkom Indonesia")
    st.write("Prediksi harga saham berdasarkan indikator fundamental dan historis dengan BI-LSTM.")

    stocks = ("TLKM.JK",)
    selected_stock = st.selectbox("Pilih saham", stocks)
    data = load_data(selected_stock)
    st.success("✅ Data berhasil dimuat!")

    st.subheader(f"📉 Harga Saham {selected_stock}")
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

    st.subheader(f"💰 Data Keuangan {selected_stock}")

    # st.subheader(f"💰 Income Statement: {selected_stock}")
    income_data = income_statement(selected_stock)

    # st.subheader(f"🏦 Balance Sheet: {selected_stock}")
    balance_data = balance_sheet(selected_stock)

    available_years = sorted(set(income_data['Year'].dropna().astype(int)).union(balance_data['Year'].dropna().astype(int)))
    if available_years:
        selected_year = st.selectbox("📅 Pilih Tahun", options=available_years, index=len(available_years)-1)
    else:
        selected_year = None
        st.warning("⚠️ Tidak ditemukan data untuk Income Statement atau Balance Sheet pada tahun manapun.")

    if selected_year and not income_data.empty:
        income_filtered = income_data[income_data['Year'] == selected_year].sort_values(by=['Year', 'Quarter'])

        if income_filtered.empty:
            st.warning(f"⚠️ Tidak ada data Income Statement untuk tahun {selected_year}.")
        elif income_filtered[['Total Revenue', 'Net Income']].isnull().values.any():
            st.warning("⚠️ Data Income Statement mengandung nilai kosong. Visualisasi tidak ditampilkan.")
            st.dataframe(income_filtered, height=200)
        else:
            df_long = pd.melt(income_filtered, id_vars=['Year-Quarter'], value_vars=['Total Revenue', 'Net Income'],
                              var_name='Metric', value_name='Value')
            fig = px.bar(df_long, x="Year-Quarter", y="Value", color='Metric', barmode='group',
                         title="Income Statement", template='plotly_white')
            fig.update_layout(xaxis_title="Quarter", yaxis_title="Value",
                              legend=dict(orientation="h", y=-0.3, x=0.5),
                              margin=dict(t=50, b=100), height=500)
            st.plotly_chart(fig)
    else:
        st.error("Data Income Statement tidak tersedia.")

    if selected_year and not balance_data.empty:
        balance_filtered = balance_data[balance_data['Year'] == selected_year].sort_values(by=['Year', 'Quarter'])

        if balance_filtered.empty:
            st.warning(f"⚠️ Tidak ada data Balance Sheet untuk tahun {selected_year}.")
        elif balance_filtered[['Total Assets', 'Total Liabilities Net Minority Interest']].isnull().values.any():
            st.warning("⚠️ Data Balance Sheet mengandung nilai kosong. Visualisasi tidak ditampilkan.")
            st.dataframe(balance_filtered, height=200)
        else:
            df_long_bs = pd.melt(balance_filtered, id_vars=['Label'],
                                 value_vars=['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity'],
                                 var_name='Metric', value_name='Value')
            fig_bs = px.bar(df_long_bs, x="Label", y="Value", color='Metric', barmode='group',
                            title="Balance Sheet", template='plotly_white')
            fig_bs.update_layout(xaxis_title="Quarter", yaxis_title="Value",
                                 legend=dict(orientation="h", y=-0.3, x=0.5),
                                 margin=dict(t=50, b=100), height=500)
            st.plotly_chart(fig_bs)
    else:
        st.error("Data Balance Sheet tidak tersedia.")

    merged_data = pd.concat([data, income_data, balance_data]).sort_index()
    merged_data = merged_data.fillna(method="ffill")
    merged_data = merged_data[~merged_data.index.duplicated(keep='last')]
    dataset_lama = merged_data[merged_data.index >= pd.to_datetime("2024-01-01")]

    # st.subheader("🧾 Data Gabungan")
    # st.dataframe(dataset_lama.tail(len(dataset_lama)), height=250)

    # st.subheader("📐 Ratio Keuangan")
    dataset_lama["DER"] = ratio(dataset_lama['Long Term Debt'], dataset_lama['Stockholders Equity'])
    dataset_lama["ROA"] = ratio(dataset_lama["Net Income"], dataset_lama["Total Assets"])
    dataset_lama["ROE"] = ratio(dataset_lama["Net Income"], dataset_lama["Stockholders Equity"])
    # st.dataframe(dataset_lama[["DER", "ROA", "ROE"]].tail(), height=150)
    
    # st.subheader("📊 Perbandingan Rasio Keuangan per Kuartal")

    # Siapkan data rasio unik per kuartal
    ratio_df = dataset_lama.copy().reset_index()
    ratio_df['Tahun'] = ratio_df['index'].dt.year
    ratio_df['Kuartal'] = ratio_df['index'].dt.quarter
    ratio_df['Periode'] = ratio_df['Tahun'].astype(str) + "Q" + ratio_df['Kuartal'].astype(str)

    # Ambil nilai unik (terakhir) per kuartal
    rasio_per_quarter = ratio_df.groupby('Periode')[['ROA', 'ROE', 'DER']].last().reset_index()

    # Ubah format ke long
    rasio_long = pd.melt(rasio_per_quarter, id_vars='Periode', var_name='Rasio', value_name='Nilai')

    # Plot dengan batas y maksimum 1
    fig = px.bar(rasio_long,
                 x='Periode',
                 y='Nilai',
                 color='Rasio',
                 barmode='group',
                 title='📊 Rasio keuangan',
                 template='plotly_white',
                 height=500,
                 color_discrete_sequence=['#00BFC4', '#F8766D', '#7CAE00'])

    fig.update_layout(
        yaxis=dict(range=[0, 0.5]),
        xaxis_title="Periode Kuartal",
        yaxis_title="Nilai Rasio",
        legend_title="Jenis Rasio",
        margin=dict(t=60, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)


    dataset_baru = dataset_lama[FEATURES]
    st.subheader("🔧 Dataset untuk prediksi")
    st.dataframe(dataset_baru.tail(len(dataset_baru)), height=150)

    model, scaler_input, scaler_close = load_model_and_scalers()
    window_size = 30
    days_ahead = st.number_input("📅 Prediksi berapa hari ke depan?", min_value=1, max_value=60, value=7)

    if st.button("📈 Prediksi Harga"):
        if len(dataset_baru) < window_size:
            st.error(f"❌ Data tidak cukup untuk prediksi (minimal {window_size} baris).")
            return

        predictions = predict_future_prices(model, scaler_input, scaler_close, dataset_baru, window_size, days_ahead)
        if len(predictions) == 0:
            return

        last_date = dataset_baru.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
        prediction_df = pd.DataFrame({'Tanggal': future_dates, 'Harga Prediksi': predictions})

        st.subheader("🔮 Hasil Prediksi")
        st.write(prediction_df)

        st.subheader("📉 Grafik Prediksi Harga (Interaktif)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dataset_baru.index, y=dataset_lama.loc[dataset_baru.index, 'Close'],
                                 mode='lines', name='Harga Historis', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers',
                                 name='Harga Prediksi', line=dict(color='orange', dash='dash')))
        fig.update_layout(xaxis_title="Tanggal", yaxis_title="Harga (IDR)",
                          hovermode="x unified", height=500,
                          xaxis_rangeslider_visible=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
