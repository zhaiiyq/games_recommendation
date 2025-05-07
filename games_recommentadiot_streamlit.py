import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def set_retro_background():
    retro_style = '''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    
    body {
        background-color: #000000;
        font-family: 'Press Start 2P', cursive;
        color: #00FF00;
        text-shadow: 2px 2px 10px rgba(0, 255, 0, 0.8);
    }
    .stApp {
        background-color: rgba(0, 0, 0, 0.9);
        border-radius: 10px;
        padding: 3rem;
        box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.8);
    }
    h1, h2, h3 {
        font-family: 'Press Start 2P', cursive;
        text-align: center;
        color: #00FF00;
    }
    .stButton>button {
        background-color: #00FF00;
        color: #000000;
        border: 2px solid #00FF00;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #000000;
        color: #00FF00;
        border: 2px solid #00FF00;
    }
    .stTextInput>div>input {
        background-color: #222222;
        color: #00FF00;
        border: 2px solid #00FF00;
    }
    </style>
    '''
    st.markdown(retro_style, unsafe_allow_html=True)

# Вставь эту функцию в начале своего кода, чтобы она применялась ко всем компонентам интерфейса.
set_retro_background()


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
st.title("Рекомендательная система для игр")

# Загрузка файла через Streamlit
uploaded_file = st.file_uploader("Загрузите файл с играми (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

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


