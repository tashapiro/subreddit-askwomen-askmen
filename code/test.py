import matplotlib.pyplot as plt
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

import plotly.figure_factory as ff
import plotly.express as px

st.title('AskMen or AskWomen')

st.subheader("Let's Predict!")

st.write('Predict whether your question sounds more like a question from the **AskWomen** or **AskMen** subreddit.')

input_var = st.text_input(label="Enter your question here:")

df_stop = pd.read_csv("/Users/tanyashapiro/Desktop/code-files/subreddit/data/text_df.csv")
df_nostop = pd.read_csv("/Users/tanyashapiro/Desktop/code-files/subreddit/data/text_df_nostops.csv")

with open('/Users/tanyashapiro/Desktop/code-files/subreddit/model/rfc_pipe.pkl',mode='rb') as pickle_in:
    pipe = pickle.load(pickle_in)

pred = pipe.predict([input_var])[0]

if input_var == '':
    st.write("")
else:
    st.write(f'You question sounds like it came from **{pred}**.')
    image = Image.open(f'/Users/tanyashapiro/Desktop/code-files/subreddit/images/{pred}.png')
    st.image(image)

    st.subheader("Text Analysis")
    st.write("Below is a  text comparison using words and phrases from your question compared to other questions on AskMen and AskWomen. Select **Remove Fluff Words** to eliminate stop words (e.g. there, what, why). Note: the model factors in stop or 'fluff' words to form a prediction.")
    remove_fluff = st.checkbox('Remove Fluff Words', value=True)

    if remove_fluff:
        text_df = df_nostop
    else:
        text_df = df_stop

    #Get list of words 

    input_text = input_var.replace("?","")
    post_words = input_text.lower().split()
    post_bigrams = [word + ' ' + post_words[index+1] for index, word in enumerate(post_words) if index<len(post_words)-1]
    post_bigrams = set(post_bigrams)
    post_words = list(set(input_text.lower().split()))

    #plot df for words
    plot_df = text_df[text_df['word'].isin(post_words)]
    plot_df = plot_df[['word','askmen_post_count','askwomen_post_count']].rename(columns={'askwomen_post_count':'askwomen','askmen_post_count':'askmen'})
    plot_df = plot_df.melt(id_vars="word").rename(columns={'word':'Word','variable':'Subreddit','value':"Posts"})
    
    #plot df for bigrams
    plot_bi = text_df[text_df['word'].isin(post_bigrams)]
    plot_bi = plot_bi[['word','askmen_post_count','askwomen_post_count']].rename(columns={'askwomen_post_count':'askwomen','askmen_post_count':'askmen'})
    plot_bi = plot_bi.melt(id_vars="word").rename(columns={'word':'Word','variable':'Subreddit','value':"Posts"})
    
    colors = {'askmen': '#4499B9', 'askwomen': '#D42E76'}

    df = px.data.tips()

    fig = px.bar(plot_df, 
                y="Word", 
                x="Posts",
                color='Subreddit', barmode='group',
                color_discrete_map=colors,
                height=400)

    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ))
    
    fig2 = px.bar(plot_bi, 
                y="Word", 
                x="Posts",
                color='Subreddit', barmode='group',
                color_discrete_map=colors,
                height=400)

    fig2.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ))

    st.write("**Word Comparison Match**")
    if plot_df.shape[0]>0:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No word matches, not enough data")

    st.write("**Bigram Comparison Match**")
    if plot_bi.shape[0]>0:
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("No bigram matches, not enough data")

