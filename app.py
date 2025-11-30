import streamlit as st
import pandas as pd
import numpy as np
import joblib
from difflib import get_close_matches

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('cleaned_movies_data.csv')
    embedding = np.load('bert_embedding.npy')
    indices_dict = joblib.load('indices_dict.joblib')

    return df,embedding,indices_dict

df , embedding , indices = load_data()

def clean_title(t):
    return str(t).lower().strip()

def recommender(title,topn=10):
    if not isinstance(title,str):
        raise ValueError('Give me valid string')
    
    title_key = clean_title(title)

    if title_key not in indices:
        choices = get_close_matches(title_key,df['title_clean'].tolist(),n=6,cutoff=0.6)
        if len(choices) == 0:
            return f"sorry we could not find {title}", None
        return f"We could not find {title} but here are some realated movies you go check ",choices
    
    idx = indices[title_key]
    query_martix = embedding[idx]

    sim_score = np.dot(embedding,query_martix)
    top_indices = np.argsort(sim_score)[::-1]
    top_indices = top_indices[top_indices!=idx]
    top_indices = top_indices[:topn]
    top_score = sim_score[top_indices]

    results = df.iloc[top_indices][['title','genres_names','director']].copy()
    results['score'] = top_score
    results = results.reset_index(drop=True)

    return f"Recommendations for {title} ",results


st.set_page_config('Recommender System by lelo raat me Studio',layout='centered')
st.title('Bert based movie recommender')

st.write("""
This recommender uses **BERT embeddings** (all-MiniLM-L6-v2) on:
- genres
- keywords
- top cast
- director
- overview

to find semantically similar movies using cosine similarity.
""")

user_title = st.selectbox('choose your movie' , options=df['title'].tolist())
topn = st.slider('How many recommendations u want',min_value=3,max_value=18,value=8)

if st.button('Recommend'):
    if not user_title.strip():
        st.error('Bro give me movie name not your fucking stupidity , wanna suck me ? ')
    else:
        msg,result = recommender(user_title,topn=topn)
        if result is None:
            st.error(msg)
        elif isinstance(result,list):
            st.error(msg)
            st.write("Close Matches")
            for m in result:
                original = df.loc[df['title_clean']==m,'title']
                if not original.empty:
                    st.write("*" + original.iloc[0])
        else:
            '''st.markdown("### This is a Header")
st.markdown("**Bold text**")
st.markdown("*Italic text*")
st.markdown("- Bullet point")'''
            for i, row in result.iterrows():
                with st.container():
                    st.markdown(f"# {row['title']}")
                    if isinstance(row['genres_names'], str):
                        st.write("**Genres:**", " ".join(row['genres_names']))
                    if isinstance(row['director'], str):
                        st.write("**Director:**", row['director'])
                    st.write(f"**Similarity score:** {row['score']:.3f}")
                st.markdown("---")  
            