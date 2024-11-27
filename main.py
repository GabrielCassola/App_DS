# Importando biblitoecas
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

#Título
st.title('Análise do conjunto de dados da Google Play Store')

# Lendo dataset
df = pd.read_csv('datasets/googleplaystore.csv')

# Limpando os dados
df.drop_duplicates(subset=['App', 'Category'], inplace=True)

# Removendo valores não numéricos em 'Android Ver' e 'Installs'
df = df[df['Android Ver'] != 'NaN']  
df = df[df['Installs'] != 'Free']
df = df[df['Installs'] != 'Paid']

# Convertendo valores para MB
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df['Size'] = pd.to_numeric(df['Size'], errors='coerce') 

# Removendo '+' e ',' e convertendo para inteiro
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '').replace(',', '') if isinstance(x, str) else x)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')  # Convertendo para numérico

# Removendo '$' e convertendo para float
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if isinstance(x, str) else x)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # Convertendo para numérico

# Convertendo para inteiro
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')  # Conversão para numérico

# Removendo valores nulos
df = df.dropna()

# Amostra
st.subheader('Amostra do dataset')
st.write(df.sample(10))


# Análise exploratória de dados
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

st.subheader('Análise exploratória de dados')
aed = sns.pairplot(df_plot, hue='Type', palette="Set2")
aed.tight_layout()
st.pyplot(aed)

st.subheader('Matriz de Correlação')
df_numeric = df.select_dtypes(include=[np.number])
corrmat = df_numeric.corr()

# Criando uma figura separada para o heatmap
fig, ax = plt.subplots(figsize=(10, 8))  # Define o tamanho da figura
sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True), fmt='.2f', ax=ax)
ax.set_title("Matriz de Correlação")
st.pyplot(fig)


st.markdown("#### 1. Relação entre Installs e Reviews")
st.write(
    "Existe uma correlação positiva clara entre o número de instalações e o número de avaliações. "
    "Aplicativos com mais instalações tendem a ter mais avaliações. Isso faz sentido, pois usuários que instalam um aplicativo são potenciais avaliadores.")

# 2. Distribuição do Rating
st.markdown("#### 2. Distribuição do Rating")
st.write("A maioria dos aplicativos tem classificações altas (entre 4 e 5). Isso indica que os usuários geralmente avaliam os aplicativos positivamente. ")

# 3. Impacto do Tipo (Type) nos Preços e Outras Métricas
st.markdown("#### 3. Impacto do Tipo (Type) nos Preços e Outras Métricas")
st.write("A análise mostra que há menos aplicativos pagos, o que indica que a maioria dos aplicativos no Google Play Store são gratuitos. ")

# 4. Tamanho do Aplicativo (Size)
st.markdown("#### 4. Tamanho do Aplicativo (Size)")
st.write("Não há uma relação clara entre o tamanho do aplicativo e outras variáveis como Rating ou Installs.")

# 5. Preço (Price)
st.markdown("#### 5. Preço (Price)")
st.write("O preço dos aplicativos é muito concentrado em valores baixos. No entanto, há alguns outliers significativos, com aplicativos que possuem preços muito altos. ")

# 6. Outliers
st.markdown("#### 6. Outliers")
st.write("Existem valores extremos em variáveis como Price, Reviews e Installs. Esses outliers podem representar casos legítimos ou podem precisar de tratamento nos dados. ")

# 7. Gratuitos x Pagos
st.markdown("#### 7. Gratuitos x Pagos")
st.write("Os aplicativos pagos tendem a ter menos instalações quando comparados aos gratuitos. Isso reflete o comportamento esperado do mercado, onde os usuários preferem experimentar apps gratuitos antes de pagar.")


st.subheader("Distribuição de aplicativos por categoria")
number_of_apps_in_category = df['Category'].value_counts().sort_values(ascending=True)

fig_pie = go.Figure(
    data=[go.Pie(
        labels=number_of_apps_in_category.index,
        values=number_of_apps_in_category.values,
        hoverinfo='label+value'
    )]
)
fig_pie.update_layout(title="Distribuição de aplicativos por categoria")
st.plotly_chart(fig_pie)

# Classificação média dos aplicativos
st.subheader("Classificação média dos aplicativos")
st.write("Analisando se existem aplicativos com classificações muito boas ou muito ruins.")

# Histograma das avaliações (Rating)
fig_hist = px.histogram(
    df, 
    x="Rating", 
    nbins=40, 
    title="Distribuição das avaliações dos aplicativos",
    labels={'Rating': 'Avaliação'}
)
fig_hist.update_layout(xaxis_title="Avaliação", yaxis_title="Quantidade de aplicativos")
st.plotly_chart(fig_hist)

# Média das avaliações
media_rating = np.mean(df['Rating'])
st.write(f"Avaliação média dos aplicativos: **{media_rating:.2f}**")

#Teste ANOVA - Comparando avaliações entre categorias
st.subheader("Comparando avaliações entre categorias (Teste ANOVA)")

# Selecionando categorias com mais de 200 aplicativos
categorias_filtradas = df.groupby('Category').filter(lambda x: len(x) > 200)
categorias = categorias_filtradas['Category'].unique()

# Teste ANOVA
grupos = [categorias_filtradas.loc[categorias_filtradas['Category'] == cat, 'Rating'].dropna() for cat in categorias]
anova_resultado = stats.f_oneway(*grupos)

st.write("Resultado do teste ANOVA:")
st.write(f"Estatística F: **{anova_resultado.statistic:.2f}**, p-valor: **{anova_resultado.pvalue:.2e}**")

if anova_resultado.pvalue < 0.05:
    st.write("**Conclusão:** Existe diferença significativa entre as avaliações médias das categorias.")
else:
    st.write("**Conclusão:** Não há evidências suficientes para afirmar que as avaliações médias diferem entre as categorias.")

# Visualização: Histograma de avaliações por categoria
st.write("**Distribuição das avaliações por categoria:**")
fig_violin = px.violin(
    categorias_filtradas, 
    x="Category", 
    y="Rating", 
    box=True, 
    points="all",
    title="Distribuição das avaliações por categoria",
    labels={'Category': 'Categoria', 'Rating': 'Avaliação'}
)
fig_violin.update_layout(xaxis_title="Categoria", yaxis_title="Avaliação")
st.plotly_chart(fig_violin)

# Melhores categorias com base na avaliação média
st.subheader("Melhores categorias com base na avaliação média")

# Selecionando categorias com pelo menos 200 aplicativos
grupos_rating = df.groupby('Category').filter(lambda x: len(x) >= 200)
media_rating_categorias = grupos_rating.groupby('Category')['Rating'].median().sort_values(ascending=False)

fig_bar = px.bar(
    media_rating_categorias,
    x=media_rating_categorias.index,
    y=media_rating_categorias.values,
    title="Avaliação média por categoria",
    labels={'x': 'Categoria', 'y': 'Avaliação Média'},
    text=media_rating_categorias.values
)
fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_bar.update_layout(
    xaxis_title="Categoria",
    yaxis_title="Avaliação Média",
    showlegend=False
)
st.plotly_chart(fig_bar)

# Adicionando linha com a média geral
media_geral = np.mean(grupos_rating['Rating'])
fig_bar.add_shape(
    type='line',
    x0=-0.5,
    y0=media_geral,
    x1=len(media_rating_categorias) - 0.5,
    y1=media_geral,
    line=dict(color='Red', dash='dashdot')
)
st.write(f"Avaliação média geral dos aplicativos: **{media_geral:.2f}**")


# Impacto do preço na avaliação (Rating)
st.subheader("Impacto do preço na avaliação (Rating)")

paid_apps = df[df.Price > 0]
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x="Price", 
    y="Rating", 
    data=paid_apps, 
    ax=ax
)
ax.set_title("Impacto do preço na avaliação dos aplicativos pagos")
ax.set_xlabel("Preço")
ax.set_ylabel("Avaliação (Rating)")

st.pyplot(fig)
st.write("O gráfico acima mostra a relação entre o preço dos aplicativos pagos e suas avaliações (Rating).")

# Tendências de preço por categoria
st.subheader("Tendências de preço por categoria")
subset_df = df[df.Category.isin([
    'GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE', 
    'LIFESTYLE', 'BUSINESS'
])]
fig, ax = plt.subplots(figsize=(15, 8))
sns.stripplot(
    x="Price", 
    y="Category", 
    data=subset_df[subset_df.Price < 100], 
    jitter=True, 
    linewidth=1, 
    ax=ax
)
ax.set_title("Tendência de preços por categoria")
st.pyplot(fig)
st.write("O gráfico acima mostra como os preços dos aplicativos variam entre diferentes categorias.")

# Análise de aplicativos com preços acima de $100
st.subheader("Aplicativos com preços acima de $100")
high_price_apps = df[['Category', 'App']][df.Price > 100]
st.write(high_price_apps)

# Distribuição de aplicativos pagos e gratuitos por categoria
st.subheader("Distribuição de aplicativos pagos e gratuitos por categoria")
new_df = df.groupby(['Category', 'Type']).agg({'App': 'count'}).reset_index()

outer_group_names = ['GAME', 'FAMILY', 'MEDICAL', 'TOOLS']
outer_group_values = [len(df.App[df.Category == category]) for category in outer_group_names]

a, b, c, d = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples]

inner_group_names = ['Paid', 'Free'] * 4
inner_group_values = []
for category in outer_group_names:
    for t in ['Paid', 'Free']:
        x = new_df[new_df.Category == category]
        try:
            inner_group_values.append(int(x.App[x.Type == t].values[0]))
        except:
            inner_group_values.append(0)

explode = (0.025, 0.025, 0.025, 0.025)

fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('equal')
mypie, texts, _ = ax.pie(
    outer_group_values, 
    radius=1.2, 
    labels=outer_group_names, 
    autopct='%1.1f%%', 
    pctdistance=1.1,
    labeldistance=0.75,  
    explode=explode, 
    colors=[a(0.6), b(0.6), c(0.6), d(0.6)], 
    textprops={'fontsize': 16}
)
plt.setp(mypie, width=0.5, edgecolor='black')

mypie2, _ = ax.pie(
    inner_group_values, 
    radius=1.2 - 0.5, 
    labels=inner_group_names, 
    labeldistance=0.7, 
    textprops={'fontsize': 12}, 
    colors=[a(0.4), a(0.2), b(0.4), b(0.2), c(0.4), c(0.2), d(0.4), d(0.2)]
)
plt.setp(mypie2, width=0.5, edgecolor='black')
plt.margins(0, 0)

st.pyplot(fig)

# Downloads: Aplicativos pagos vs gratuitos
st.subheader("Comparação de downloads entre aplicativos pagos e gratuitos")
trace0 = go.Box(
    y=np.log10(df['Installs'][df.Type == 'Paid']),
    name='Pagos',
    marker=dict(color='rgb(214, 12, 140)')
)
trace1 = go.Box(
    y=np.log10(df['Installs'][df.Type == 'Free']),
    name='Gratuitos',
    marker=dict(color='rgb(0, 128, 128)')
)
layout = go.Layout(
    title="Downloads de aplicativos pagos vs gratuitos (log)",
    yaxis={'title': 'Número de downloads (escala logarítmica)'}
)
data = [trace0, trace1]
fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig)

st.write("""
- Aplicativos gratuitos possuem um número significativamente maior de downloads comparados aos pagos.
- Isso pode refletir a preferência dos usuários por experimentar apps gratuitos antes de comprar versões pagas.
""")


# Análise de sentimentos dos usuários
st.title("Análise de Sentimentos - Avaliações de usuários")
# Carregando os dados
reviews_df = pd.read_csv('datasets/googleplaystore_user_reviews.csv')
merged_df = pd.merge(df, reviews_df, on="App", how="inner")
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])

# Agrupando os dados
grouped_sentiment_category_count = merged_df.groupby(['Category', 'Sentiment']).agg({'App': 'count'}).reset_index()
grouped_sentiment_category_sum = merged_df.groupby(['Category']).agg({'Sentiment': 'count'}).reset_index()
new_df = pd.merge(grouped_sentiment_category_count, grouped_sentiment_category_sum, on=["Category"])
new_df['Sentiment_Normalized'] = new_df.App / new_df.Sentiment_y
new_df = new_df.groupby('Category').filter(lambda x: len(x) == 3)

# Criando as barras para categorias de sentimentos
trace1 = go.Bar(
    x=list(new_df.Category[::3]),
    y=new_df.Sentiment_Normalized[::3],
    name='Negativo',
    marker=dict(color='rgb(209,49,20)')
)

trace2 = go.Bar(
    x=list(new_df.Category[::3]),
    y=new_df.Sentiment_Normalized[1::3],
    name='Neutro',
    marker=dict(color='rgb(49,130,189)')
)

trace3 = go.Bar(
    x=list(new_df.Category[::3]),
    y=new_df.Sentiment_Normalized[2::3],
    name='Positivo',
    marker=dict(color='rgb(49,189,120)')
)

# Configurando layout do gráfico
layout = go.Layout(
    title='Análise de sentimentos',
    barmode='stack',
    xaxis={'tickangle': -45, 'title': 'Categorias'},
    yaxis={'title': 'Proporção de avaliações'}
)

# Gerando o gráfico de barras empilhadas com Plotly
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
st.plotly_chart(fig)

# Boxplot da polaridade dos sentimentos
st.subheader("Distribuição de Polaridade de Sentimentos por Tipo de Aplicativo")
sns.set_style('ticks')
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Type', y='Sentiment_Polarity', data=merged_df, ax=ax)
ax.set_title('Distribuição da polaridade de sentimentos')
st.pyplot(fig)


st.subheader("Wordcloud: Uma visão rápida das avaliações")
nltk.download('stopwords')

# Definindo palavras de parada (stopwords)
stop = stopwords.words('english')
stop = stop + ['app', 'APP', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
               'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']

# Limpando e filtrando avaliações
free_reviews = merged_df.loc[merged_df.Type == 'Free', 'Translated_Review'].dropna().str.replace(r'\s+', ' ', regex=True)
paid_reviews = merged_df.loc[merged_df.Type == 'Paid', 'Translated_Review'].dropna().str.replace(r'\s+', ' ', regex=True)

# Limpando as palavras com stopwords para as avaliações
merged_df['Translated_Review'] = merged_df['Translated_Review'].apply(lambda x: " ".join(x for x in str(x).split(' ') if x not in stop))

# Gerando a WordCloud para avaliações de aplicativos gratuitos
st.subheader("WordCloud para aplicativos gratuitos")
free_reviews_text = ' '.join(free_reviews)
wc = WordCloud(stopwords=stop, background_color="white", max_words=200, colormap="Set2").generate(free_reviews_text)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Gerando a WordCloud para avaliações de aplicativos pagos
st.subheader("WordCloud para aplicativos pagos")
paid_reviews_text = ' '.join(paid_reviews)
wc = WordCloud(stopwords=stop, background_color="white", max_words=200, colormap="Set2").generate(paid_reviews_text)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Conclusão

# Seção: Importância do pré-processamento de Dados]
st.header("Conclusões gerais")
st.subheader("Importância do pré-processamento de dados")
st.write("""
- **Desafios superados:** Dados inconsistentes, valores faltantes e formatos variados.
""")

# Seção: Relevância das análises exploratórias
st.subheader("Relevância das análises exploratórias")
st.write("""
- Identificar padrões nos dados, como a correlação entre número de instalações e avaliações, permite entender o comportamento do mercado e guiar estratégias.
- **Valor para negócios:** Empresas podem usar insights como “aplicativos gratuitos atraem mais downloads” para decidir modelos de monetização.
""")

# Seção: aplicativos pagos vs. gratuitos
st.header("Aplicativos pagos vs. gratuitos")
st.write("""
- Aplicativos gratuitos têm maior alcance, mas os pagos tendem a atrair usuários mais engajados e dispostos a avaliar melhor.
- **Estratégia sugerida:** Modelos híbridos, como aplicativos gratuitos com recursos pagos (freemium), podem maximizar alcance e receita.
""")

# Seção: Impacto das categorias no sucesso do aplicativo
st.header("Impacto das categorias no sucesso do aplicativo")
st.write("""
- Categorias como **Jogos** e **Ferramentas** lideram em downloads e engajamento, mas categorias menores (como **Educação**) apresentam avaliações mais positivas.
- **Oportunidade de mercado:** Investir em nichos com avaliações altas pode criar diferenciação em mercados menos saturados.
""")

# Seção: Sentimentos e feedback dos usuários
st.header("Sentimentos e feedback dos usuários")
st.write("""
- Análise de sentimentos revelou que avaliações geralmente são positivas, mas categorias como **Negócios** têm maior proporção de críticas neutras/negativas.
- **Ação sugerida:** Empresas devem usar essas análises para ajustar funcionalidades e melhorar a experiência do usuário, especialmente em categorias com menor engajamento emocional.
""")

# Seção: Visão estratégica para startups e negócios
st.header("Visão estratégica para startups e negócios")
st.write("""
- **Para desenvolvedores independentes:** Focar em categorias menos exploradas com alta satisfação dos usuários, como **Educação** e **Saúde**.
- **Para grandes empresas:** Ampliar o portfólio em categorias com alta tração e inovar em recursos para manter a competitividade.
""")

