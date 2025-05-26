import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Title
st.title("📊 Analisis Saham dan Rasio Keuangan")

# Sidebar
ticker = st.sidebar.text_input("Masukkan Ticker Saham (Contoh: TLKM.JK)", "TLKM.JK")
start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir", pd.to_datetime("2024-12-31"))

if ticker:
    st.subheader(f"Data Saham: {ticker}")
    
    # Ambil data harga saham
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    # Tampilkan harga saham
    st.line_chart(hist['Close'], use_container_width=True)
    
    st.subheader("Data Laporan Keuangan")
    bs = stock.balance_sheet
    is_ = stock.financials
    
    # Pastikan kita ambil dari periode tahunan (dan transpose agar tahun menjadi baris)
    bs = bs.T
    is_ = is_.T
    
    # Ambil kolom penting
    try:
        df_fin = pd.DataFrame({
            "Net Income": is_["Net Income"],
            "Total Assets": bs["Total Assets"],
            "Total Equity": bs["Stockholders Equity"],
            "Total Liabilities": bs["Long Term Debt"]
        })

        # Hitung rasio
        df_fin["ROE"] = df_fin["Net Income"] / df_fin["Total Equity"]
        df_fin["ROA"] = df_fin["Net Income"] / df_fin["Total Assets"]
        df_fin["DER"] = df_fin["Total Liabilities"] / df_fin["Total Equity"]

        # Tampilkan tabel
        st.dataframe(df_fin.style.format({
            "ROE": "{:.2%}", "ROA": "{:.2%}", "DER": "{:.2f}"
        }))

        # Visualisasi
        st.subheader("Visualisasi Rasio Keuangan")
        fig, ax = plt.subplots()
        df_fin[["ROE", "ROA", "DER"]].plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Kolom tidak ditemukan dalam laporan keuangan: {e}")
