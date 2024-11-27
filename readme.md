# Análise de Dados da Google Play Store

Este projeto é uma aplicação interativa desenvolvida em **Streamlit** para analisar dados da Google Play Store. A aplicação explora diferentes aspectos, como avaliações de aplicativos, categorias, sentimentos dos usuários e tendências de mercado, com o objetivo de extrair insights úteis e estratégicos.

---

## 📊 **Principais Funcionalidades**
- **Visualização de Dados**:
  - Gráficos interativos como histogramas, heatmaps e scatterplots.
  - Nuvens de palavras para destacar termos frequentes em avaliações.
- **Análise Exploratória**:
  - Correlação entre variáveis, como número de instalações e avaliações.
  - Comparação entre aplicativos pagos e gratuitos.
- **Análise Estatística**:
  - Teste ANOVA para identificar diferenças significativas entre categorias.
  - Identificação de outliers e padrões de mercado.
- **Sentimento dos Usuários**:
  - Proporção de avaliações positivas, negativas e neutras por categoria.
  - WordCloud para termos mais usados em aplicativos pagos e gratuitos.

---

## 📁 **Estrutura do Projeto**
- **`main.py`**: Arquivo principal que contém o código da aplicação Streamlit.
- **`datasets/`**: Diretório contendo os arquivos de dados:
  - `googleplaystore.csv`: Dados principais sobre os aplicativos.
  - `googleplaystore_user_reviews.csv`: Avaliações traduzidas e sentimentos dos usuários.
- **`README.md`**: Documentação do projeto.

---

## 🛠️ **Pré-requisitos**
1. **Python 3.8+** instalado.
2. Instale as dependências necessárias executando:
   ```bash
   pip install -r requirements.txt