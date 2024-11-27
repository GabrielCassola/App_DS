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

# Convertendo tudo para MB
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))


# Amostra
st.title('Análise do conjunto de dados da Google Play Store')
st.subheader('Amostra do dataset')
st.write(df.sample(5))
