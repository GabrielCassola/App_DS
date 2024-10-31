import pandas as pd
import seaborn as sns
from sklearn import datasets
import streamlit as st

df = sns.load_dataset('iris')


st.title('Análise do Conjunto de Dados Iris')
st.write(df.head())

st.subheader('Estatísticas Descritivas')
st.write(df.describe())

st.subheader('Gráfico de Dispersão: Sepal Lenght vs Sepal Width')
st.write('Visualizção das características das espécies de Iris')
scatter_plot = sns.scatterplot(data=df, x='sepal_length', y='sepal_width')
st.pyplot(scatter_plot.figure)

st.subheader('Distribuição do comprimento da pétala')
st.write('Distribuição do comprimento da pétala por espécie')
hist_plot = sns.histplot(data=df, x='petal_length', hue='species', multiple='stack')
st.pyplot(hist_plot.figure)


