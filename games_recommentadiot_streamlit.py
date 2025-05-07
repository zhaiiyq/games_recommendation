import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Добавляем стили: фон, шрифт, неоновый эффект
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    html, body, .stApp {
        background-color: #0d0d1a;
        background-image: url('https://raw.githubusercontent.com/zhaiiyq/games-recommendation/main/minecraft-cherry.gif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Orbitron', sans-serif;
        color: #e0e0e0;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff;
    }

    .stTextInput > div > div > input {
        background-color: #1e1e2f;
        color: #00ffff;
        border: 1px solid #00ffff;
    }

    .stSelectbox > div > div > div > div {
        background-color: #1e1e2f;
        color: #00ffff;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2f;
        border-radius: 10px 10px 0 0;
        color: #00ffff;
    }

    .stButton > button {
        background-color: #00ffff;
        color: #000000;
        font-weight: bold;
        border-radius: 10px;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: #00cccc;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Загружаем и готовим данные
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
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

# Функция для поиска рекомендаций по запросу
@st.cache_data
def search_recommendation(df, query, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['features'])
    
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    results = df.iloc[top_indices][['Название', 'Жанры', 'Разработчик', 'Общая оценка', 'Описание', 'Цена']].copy()
    results['Сходство'] = cosine_sim[top_indices]
    return results

# Функция рекомендаций по товару
@st.cache_data
def recommend(df, title, num_recommendations=5):
    indices = pd.Series(df.index, index=df['Название']).drop_duplicates()
    if title not in indices:
        return pd.DataFrame()
    
    idx = indices[title]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['features'])
    
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    game_indices = [i[0] for i in sim_scores]
    results = df.iloc[game_indices][['Название', 'Жанры', 'Разработчик', 'Общая оценка', 'Описание', 'Цена']].copy()
    results['Сходство'] = [sim_scores[i][1] for i in range(len(sim_scores))]
    
    return results

# Интерфейс Streamlit
st.markdown("<h1 style='text-align: center;'>🎮 Рекомендательная система для игр</h1>", unsafe_allow_html=True)


# Загрузка файла через Streamlit
uploaded_file = st.file_uploader("Загрузите файл с играми (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    tab1, tab2 = st.tabs(["Поиск", "Рекомендации"])

    with tab1:
        st.markdown("<h2>🔎 Поиск по запросу</h2>", unsafe_allow_html=True)

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


