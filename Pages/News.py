import streamlit as st
import pandas as pd
import json 
from datetime import datetime
import requests

@st.cache_data
def load_news(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        ekonomi_news = data['data']
    
        return ekonomi_news
    else:
        print("Gagal mengambil data. Status:", response.status_code)
        
        
def show_news_Card(item):
    st.markdown(
     f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            background-color: #ffffff;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        ">
            <img src="{item['image']['large']}" style="width:100%; border-radius: 10px;"/>
            <h3 style="margin-top: 15px;">
                <a href="{item['link']}" target="_blank" style="text-decoration: none; color: #000000;">
                    {item['title']}
                </a>
            </h3>
            <p style="color: gray; margin: 5px 0 10px 0;">
                📅 {datetime.fromisoformat(item['isoDate'].replace('Z', '+00:00')).strftime('%d %B %Y %H:%M')}
            </p>
            <p>{item['contentSnippet']}</p>
        </div>
        """,
        unsafe_allow_html=True
        )
    
    
def main():
    st.set_page_config(page_title="Berita Pasar", page_icon="📰")
    st.title("📰 Halaman Berita")
    st.write("Ini adalah halaman berita dari saham atau informasi terkini.")
    
    data = load_news('https://berita-indo-api-next.vercel.app/api/cnbc-news/market')
    
    st.subheader("Data berita terbaru")
    
    for item in data:
        show_news_Card(item)
        

if __name__ == '__main__':
    main()