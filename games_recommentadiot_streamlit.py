import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем и готовим данные
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Название'] = df['Название'].fillna('')
        df['Описание'] = df['Описание'].fillna('')
        df['Жанры'] = df['Жанры'].fillna('')
        df['Цена'] = df['Цена'].fillna('')
        df['Разработчик'] = df['Разработчик'].fillna('')
        
        # Взвешивание признаков: Название — важнее, Жанры — тоже весомые
        df['features'] = (
            df['Название'] * 3 + ' ' +
            df['Жанры'] * 2 + ' ' +
            df['Описание'] + ' ' +
            df['Разработчик']
        )
        
        return df
    except FileNotFoundError:
        st.error("Файл не найден. Проверьте путь.")
        return pd.DataFrame()

# Ввод пути к файлу
file_path = st.text_input("Введите путь к файлу:", "games1.csv")

if file_path:
    df = load_data(file_path)

    if not df.empty:
        tab1, tab2 = st.tabs(["Поиск", "Рекомендации"])

        with tab1:
            st.header("Поиск по запросу")
            query = st.text_input("Введите запрос для поиска (например, 'open world shooter'):")
            if query:
                results = search_recommendation(df, query)
                st.write(results)

        with tab2:
            st.header("Рекомендации по товару")
            selected_title = st.selectbox("Выберите товар:", df['Название'].unique())
            if selected_title:
                recommendations = recommend(df, selected_title)
                st.write(recommendations)

