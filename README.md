# Previsão de Preços de Imóveis com Redes MLP 🏠

Este projeto desenvolve uma solução fim-a-fim de Machine Learning para prever o preço de venda de residências. O grande diferencial desta abordagem foi realizar a **fusão de dados** de características físicas dos imóveis com indicadores socioeconômicos e demográficos da região baseados no código postal (*zipcode*).

A modelagem preditiva foi feita utilizando uma rede neural artificial **Multi-Layer Perceptron (MLP)** profunda com TensorFlow e Keras.

## 📁 Estrutura do Repositório

O projeto está estruturado da seguinte forma:
```text
├── data/              # Datasets originais (.csv)
├── plots/             # Gráficos gerados durante a análise (Heatmap, Boxplots, Loss)
├── src/               # Código-fonte (Notebook Jupyter com o pipeline)
│   └── real_estate_mlp.ipynb
├── reports/           # Relatório técnico detalhado formatado em LaTeX e PDF
└── README.md          # Documentação principal do projeto

```

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Python 3**
* **Pandas & NumPy:** Manipulação e fusão dos dados
* **Seaborn & Matplotlib:** Análise exploratória e visualização de gráficos
* **Scikit-Learn:** Pré-processamento, divisão de dados e escalonamento
* **TensorFlow & Keras:** Construção e treinamento da rede neural MLP
* **LaTeX:** Redação do relatório técnico profissional

## 📈 Etapas do Projeto & Decisões Técnicas

### 1. Fusão de Dados & Engenharia de Features

* União das bases `kc_house_data.csv` e `zipcode_demographics.csv` usando a chave comum `zipcode`.
* Eliminação de variáveis irrelevantes para predições futuras (`id` e `date`).
* Criação da feature binária `was_renovated` a partir do ano de reforma para simplificar o aprendizado da rede neural.

### 2. Análise Exploratória (EDA) & Filtros

* **Outliers:** Identificação e remoção de uma inconsistência grave na coluna `bedrooms` (imóvel registrado incorretamente com mais de 30 quartos).
* **Seleção de Atributos:** Aplicação de um filtro estatístico baseado no coeficiente de **Spearman** com corte de $|\rho| \ge 0.3$.
* *Exceção de Negócio 1:* A variável `view` (0.29 de correlação) foi mantida pelo forte impacto comercial na avaliação de imóveis.
* *Exceção de Negócio 2:* A longitude (`long`) foi mantida para fazer o par geométrico com a latitude (`lat`), preservando a informação de geolocalização exata.


* Retirada das colunas com a contagem absoluta de escolaridade (`edctn_*_qty`) deixando apenas as respectivas variáveis percentuais.

### 3. Preparação dos Dados & Pipeline

* **Transformação Logarítmica:** Aplicada à variável alvo `price` e variáveis financeiras para reduzir a assimetria e ajudar na convergência do gradiente.
* **Prevenção de Data Leakage:** Separação estrita em 80% treino e 20% teste. O ajuste do `StandardScaler` foi feito *apenas* na base de treino.

### 4. Arquitetura da Rede MLP

O modelo conta com uma arquitetura de funil extrator de características:

* **Camada de Entrada:** 17 inputs
* **Camadas Ocultas:** 4 camadas densas (256 ➡️ 128 ➡️ 64 ➡️ 32 neurônios) com ativação **ReLU**.
* **Camada de Saída:** 1 neurônio com ativação **Linear** (Regressão).
* **Otimizador:** Adam ($\alpha = 10^{-3}$) minimizando o Erro Quadrático Médio (**MSE**).
* **Regularização:** Uso de **Early Stopping** com paciência de 15 épocas para evitar *overfitting*.

## 🚀 Como Executar o Projeto

Você pode executar este projeto de duas formas:

### Opção 1: Localmente (Jupyter / VS Code)

1. Clone este repositório:
```bash
git clone https://github.com/F-Loschi/real-estate-mlp-predictor.git

```


2. Abra o ambiente do Jupyter Notebook ou VS Code na raiz do projeto.
3. Execute a primeira célula do arquivo `src/atividadeExtra.ipynb` para instalar as dependências necessárias automaticamente via `!pip install`.
4. Execute as demais células para acompanhar a análise e o treino da rede MLP.

### Opção 2: Google Colab (Sem instalar nada na máquina)

1. Faça o upload do arquivo `src/atividadeExtra.ipynb` para o seu Google Drive.
2. Abra o arquivo com o Google Colab.
3. Faça o upload dos arquivos `.csv` da pasta `data/` para o ambiente temporário do Colab e execute as células normalmente.

## 📄 Relatório Técnico

O relatório científico completo, contendo as justificativas matemáticas detalhadas, a tabela da arquitetura e as análises das curvas de convergência, encontra-se na pasta `reports/` compilado a partir do código LaTeX.
