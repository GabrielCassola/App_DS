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

# Número de instalações sendo int e removendo '+'e ','
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: int(x))

# Preço sendo float e removendo '$'
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))

# Removendo valores nulos
df = df.dropna()

# Amostra
st.title('Análise do conjunto de dados da Google Play Store')
st.subheader('Amostra do dataset')
st.write(df.sample(10))


# Análise exploratória de dados
st.subheader('Análise exploratória de dados')
x = df['Rating']
y = df['Size']
z = df['Installs'][df['Installs'] != 0]
p = df['Reviews'][df['Reviews'] != 0]
t = df['Type']
price = df['Price']

df_plot = pd.DataFrame({
    'Rating': x,
    'Size': y,
    'Installs': np.log(z), 
    'Reviews': np.log10(p), 
    'Type': t,
    'Price': price
})

plot = sns.pairplot(df_plot, hue='Type', palette="Set2")
st.pyplot(plot)