import pandas as pd
import numpy as np
import streamlit as st
import joblib
from difflib import get_close_matches

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('cleaned_movies_data.csv')
    embedding = np.load("bert_embedding.npy")
    indices = joblib.load("indices_dict.joblib")

    return df , embedding , indices

df , embedding , indices = load_data()

def recommendation_logic(title,topn=10):
    title_key = str(title).lower().strip()
    if title_key not in indices:
        choices = get_close_matches(title_key,df['title_clean'].tolist(),cutoff=0.6,n=6)
        if len(choices) == 0:
            return f'Sorry we could not find {title} ',None
        return f"Sorry {title} not found , but we have some similar options ",choices
    idx = indices[title_key] 
    query = embedding[idx]

    sim_score = np.dot(embedding,query)
    top_n = np.argsort(sim_score)[::-1]
    top_n = top_n[top_n!=idx]
    top_n = top_n[:topn]
    top_score = sim_score[top_n]

    results = df.iloc[top_n][['title','genres_names','director']].copy()
    results['score'] = top_score
    results = results.reset_index(drop=True)

    return title,results
 
st.set_page_config('Recommender System', layout='centered')
st.title("Recommeder System by kal_ana studio")

st.subheader('*Based on* Bert Model')

user_input = st.text_input('Enter your movie dear')
topn = st.selectbox('select quantity of Recommendations',options=list(range(1,26)))

if st.button('recommend'):
    key , result = recommendation_logic(user_input,topn=topn)
    if result is None:
        st.error(key)
    if isinstance(result,list):
        st.error(key)
        for m in result:
            st.write(m)
    else:
        st.success(key)
        st.table(result)
                
                
