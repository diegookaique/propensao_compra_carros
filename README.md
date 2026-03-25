# Previsão de Propensão de Compra de Carros com XGBoost

## Visão Geral do Projeto

Este projeto tem como objetivo desenvolver e implementar um modelo de Machine Learning (XGBoost) para prever a propensão de clientes comprarem um carro. A análise se baseia em dados de clientes, incluindo idade e salário anual, para identificar padrões que indicam uma maior probabilidade de compra.

## Objetivo

O principal objetivo é construir um modelo preditivo robusto e eficaz que possa auxiliar empresas a otimizar suas estratégias de marketing e vendas, direcionando recursos para clientes com maior probabilidade de conversão.

## Conjunto de Dados

O dataset utilizado, `CARRO_CLIENTES.csv`, contém as seguintes colunas:

*   `id`: Identificador único do cliente.
*   `genero`: Gênero do cliente (Masculino/Feminino).
*   `idade`: Idade do cliente.
*   `salario`: Salário anual do cliente.
*   `comprou`: Variável alvo, indicando se o cliente comprou um carro (1) ou não (0).

## Metodologia

A metodologia seguida neste projeto inclui as seguintes etapas:

1.  **Carregamento e Exploração de Dados (EDA):**
    *   Carregamento do dataset `CARRO_CLIENTES.csv` usando Pandas.
    *   Análise inicial das primeiras linhas (`.head()`), informações gerais (`.info()`), estatísticas descritivas (`.describe()`).
    *   Verificação de valores ausentes (`.isnull().sum()`) e duplicatas (`.duplicated().sum()`).
    *   Análise de outliers utilizando box plots para as colunas numéricas.

2.  **Pré-processamento de Dados:**
    *   Tradução dos nomes das colunas para português para facilitar a compreensão (`User ID` para `id`, `Gender` para `genero`, etc.).
    *   Exclusão da coluna `id`, que não é relevante para o modelo preditivo.
    *   Codificação da variável categórica `genero` utilizando `LabelEncoder`.

3.  **Análise de Correlação e Visualização:**
    *   Cálculo e visualização da matriz de correlação para identificar relações entre as variáveis, especialmente com a variável alvo `comprou`.
    *   Visualizações específicas (box plots e histogramas) para entender a distribuição da `idade` e do `salario` em relação à decisão de compra.

4.  **Divisão dos Dados:**
    *   Separação das features (`X`) e da variável alvo (`Y`).
    *   Divisão dos dados em conjuntos de treino e teste (80% treino, 20% teste) usando `train_test_split` do `sklearn.model_selection`.

5.  **Treinamento do Modelo XGBoost:**
    *   Inicialização e treinamento de um classificador XGBoost (`xgb.XGBClassifier`).
    *   O objetivo do modelo foi definido como `binary:logistic`, apropriado para problemas de classificação binária.

6.  **Avaliação do Modelo:**
    *   Realização de previsões (`Y_pred`) e obtenção das probabilidades (`Y_pred_prob`) no conjunto de teste.
    *   Avaliação do desempenho do modelo utilizando métricas como:
        *   **Acurácia** (`accuracy_score`)
        *   **Relatório de Classificação** (`classification_report`) com Precisão, Recall e F1-Score para ambas as classes.
        *   **Matriz de Confusão** (`confusion_matrix`).

7.  **Análise de Importância das Features:**
    *   Extração e análise das features mais importantes do modelo XGBoost (`model_xgboost.get_booster().get_score(importance_type='gain')`).

## Resultados e Conclusões

### Desempenho do Modelo

O modelo XGBoost demonstrou um **desempenho robusto**, alcançando uma **Acurácia de 0.905** no conjunto de teste. Os principais pontos da avaliação foram:

*   **Classe 0 (Não Comprou):** Alta **Recall (0.95)** e boa **Precisão (0.89)**.
*   **Classe 1 (Comprou):** Excelente **Precisão (0.93)** e bom **Recall (0.85)**. Esta alta precisão para a classe positiva é crucial para otimizar campanhas, minimizando o contato com clientes que não comprariam.

**Matriz de Confusão:**
*   106 Verdadeiros Negativos
*   75 Verdadeiros Positivos
*   6 Falsos Positivos
*   13 Falsos Negativos

Estes indicadores mostram que o modelo é **altamente confiável e eficaz** na identificação de potenciais compradores, com uma baixa taxa de erros significativos.

### Importância das Características

A análise de importância das features, tanto pela matriz de correlação quanto pelo modelo XGBoost, confirmou que a **'idade'** é a característica mais influente na decisão de compra (importância de 1.78), seguida pelo **'salário'** (importância de 1.16). Isso sugere que, embora ambos sejam importantes, a idade do cliente é um fator preditivo mais forte para a compra de carros.

**Em resumo, o modelo XGBoost construído é uma ferramenta valiosa e eficaz para identificar clientes com alta propensão à compra de carros, baseando-se principalmente na idade e, em menor grau, no salário anual. Sua alta precisão e recall para a classe de interesse (compra) o tornam ideal para otimizar estratégias de marketing e vendas.**

## Como Executar o Projeto

Para replicar e executar este projeto, siga os passos abaixo:

1.  **Clone o Repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <nome_do_repositorio>
    ```

2.  **Crie e Ative um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: .\venv\Scripts\activate
    ```

3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    (Certifique-se de criar um arquivo `requirements.txt` com as bibliotecas usadas: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`)

4.  **Execute o Notebook:**
    Abra o arquivo `.ipynb` (por exemplo, `nome_do_seu_notebook.ipynb`) em um ambiente como Jupyter Notebook, JupyterLab ou Google Colab e execute as células sequencialmente.

## Dependências

As principais bibliotecas utilizadas neste projeto são:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `xgboost`
*   `matplotlib`
*   `seaborn`

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

---

📎 **Projeto desenvolvido por:** Diego Kaique

🔗 **LinkedIn:** [https://www.linkedin.com/in/diego-kaique-9ba3697b]

📧 **Contato:** [kaique_0208@hotmail.com]
