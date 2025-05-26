import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from Prediction_Model import load_data
from Prediction_Model import income_statement
from Prediction_Model import balance_sheet

st.subheader("🧾 Statistik")
stocks = ("TLKM.JK",)
selected_stock = st.selectbox("Pilih saham", stocks)

df = load_data(selected_stock)
df['Year'] = df.index.year

# st.subheader("Balance Sheet")
# st.dataframe(df)
# st.dataframe(df.tail(len(df)), height=250)


# Layout tiga kolom
col1, col2, col3 = st.columns(3)

# Style agar semua container punya tinggi sama
container_style = """
<style>
    .element-container:has(.metric-container) {
        min-height: 0px;
        padding: 10px 0;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-container {
        flex-grow:1;
    }
    
</style>
"""
st.markdown(container_style, unsafe_allow_html=True)

with col1:
    with st.container(border=False):
        st.subheader("Harga Saham")
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        harga_penutupan = df['Close'].iloc[0]
        st.metric(label="📈 Harga Saham", value=f"Rp {harga_penutupan:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container(border=False):
        st.subheader("Volume Saham")
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        volume_penutupan = df['Volume'].iloc[0]
        st.metric(label="📊 Volume Saham", value=f"{volume_penutupan:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container(border=False):
        st.subheader("Tanggal")
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        today_date = datetime.today().date()
        st.metric(label="📅 Tanggal", value=today_date.strftime("%d %B %Y"))
        st.markdown("</div>", unsafe_allow_html=True)


# Baris 2

tahun_tersedia = sorted(df['Year'].unique(), reverse=True)
selected_year = st.selectbox("📅 Pilih Tahun", tahun_tersedia, index=0)
st.metric(label="📅 Tanggal", value=f"Tahun {selected_year}")

col4, col5 = st.columns(2)

with col4:
    # Income Statement
    chart_income_statement = income_statement(selected_stock)
    chart_income_statement['Year'] = chart_income_statement.index.year
    chart_income_statement['Quarter'] = chart_income_statement.index.quarter
    # Kolom kombinasi tahun dan kuartal
    chart_income_statement['Year-Quarter'] = chart_income_statement['Year'].astype(
        str) + "Q" + chart_income_statement["Quarter"].astype(str)

    # Filter Tahun
    chart_income_statement = chart_income_statement[chart_income_statement['Year'] == selected_year]

    # urutan berdasarkan tahun dan kuartal
    chart_income_statement = chart_income_statement.sort_values(
        by=['Year', 'Quarter'])

    # st.dataframe(chart_income_statement, height=250)
    if chart_income_statement.empty:
        st.warning(
            f"Tidak ada data Income Statement untuk tahun {selected_year}.")
        fig = go.figure()
        fig.update_layout(title_text="Tidak ada data Income Statement",
                          xaxis=dict(title="Quarter"),
                          yaxis=dict(title="Value"),
                          annotations=[dict(
                              text="Data tidak tersedia", x=0.5, y=0.5, showarrow=False, font=dict(size=20))],
                          template="plotly_white",
                          height=500,
                          margin=dict(t=50, b=100))
        st.plotly_chart(fig)
    else:
        value_vars = ['Total Revenue', 'Net Income']

        df_long = pd.melt(
            chart_income_statement,
            id_vars=['Year-Quarter'],
            value_vars=value_vars,
            var_name='Metric',
            value_name='Value',
        )
        fig = px.bar(
            df_long,
            x="Year-Quarter",
            y="Value",
            color='Metric',
            barmode='group',
            title="Income Statement",
        )
        fig.update_layout(
            xaxis_title="Quarter",
            yaxis_title="Value",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=100),
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig)

with col5:
    # Balance Sheet
    chart_balance_sheet = balance_sheet(selected_stock)
    chart_balance_sheet['Year'] = chart_balance_sheet.index.year
    chart_balance_sheet['Quarter'] = chart_balance_sheet.index.quarter
    # chart_balance_sheet['Year-Quarter'] = chart_balance_sheet['Year'].astype(str) + "Q" + chart_balance_sheet["Quarter"].astype(str)

    # Filter tahun
    chart_balance_sheet = chart_balance_sheet[chart_balance_sheet['Year']
                                              == selected_year]
    # Urutkan berdasarkan waktu
    chart_balance_sheet = chart_balance_sheet.sort_values(
        by=['Year', 'Quarter'])

    # st.dataframe(chart_balance_sheet, height=250)

    if chart_balance_sheet.empty:
        st.warning(
            f"Belum tersedia data Balance Sheet untuk tahun {selected_year}.")
        fig_bs = go.Figure()
        fig_bs.update_layout(
            title="Balance Statement",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Value"),
            annotations=[dict(text="Data tidak tersedia", x=0.5,
                              y=0.5, showarrow=False, font=dict(size=20))],
            template="plotly_white",
            height=500,
            margin=dict(t=50, b=100)
        )
        # st.plotly_chart(fig_bs)
    else:
        value_vars_bs = ['Total Assets',
                         'Total Liabilities Net Minority Interest']

        # Transformasi data menjadi format long
        df_long_bs = pd.melt(
            chart_balance_sheet,
            id_vars=['Year'],
            value_vars=value_vars_bs,
            var_name='Metric',
            value_name='Value',
        )

        df_long_bs['Year'] = df_long_bs['Year'].astype(str)
        # Plot
        fig_bs = px.bar(
            df_long_bs,
            x="Year",
            y="Value",
            color='Metric',
            barmode='group',
            title="Balance Statement",
        )

        fig_bs.update_layout(
            xaxis_title="Year",
            yaxis_title="Value",
            legend=dict(
                orientation="h",       # horizontal
                yanchor="bottom",      # anchor ke bawah
                # pindahkan ke bawah grafik (nilai negatif ke luar plot)
                y=-0.3,
                xanchor="center",
                x=0.5                  # posisikan di tengah
            ),
            margin=dict(t=50, b=100),  # tambahkan space bawah
            height=500,                 # sesuaikan tinggi agar tidak terlalu padat
            template='plotly_white'
        )

        st.plotly_chart(fig_bs)
