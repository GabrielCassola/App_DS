# Importando biblitoecas
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st


# Lendo dataset
df = pd.read_csv('datasets/googleplaystore.csv')

# Limpando os dados
df.drop_duplicates(subset=['App', 'Category'], inplace=True)
df = df[df['Android Ver'] != np.nan]
df = df[df['Android Ver'] != 'NaN']
df = df[df['Installs'] != 'Free']
df = df[df['Installs'] != 'Paid']



# Amostra
st.title('Análise do conjunto de dados da Google Play Store')
st.subheader('Amostra do dataset')
st.write(df.sample(5))
