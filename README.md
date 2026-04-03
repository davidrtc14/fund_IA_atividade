# 🍷 KNN — Wine Quality Classification

Atividade Prática da disciplina de **Inteligência Artificial**, aplicando o algoritmo **K-Nearest Neighbors (KNN)** para classificação de qualidade de vinhos.

---

## 📋 Sobre a Atividade

O objetivo é aplicar o fluxo completo de um projeto de Ciência de Dados — desde o pré-processamento até a tunagem de hiperparâmetros — utilizando o algoritmo KNN com diferentes métricas de distância.

---

## 📁 Estrutura do Repositório

```
├── atividade_knn_wine.py       # Script principal
├── winequalityN.csv            # Dataset (baixar do Kaggle)
├── relatorio_knn.docx          # Relatório com análise completa
└── README.md
```

> **Atenção:** o arquivo `winequalityN.csv` não está incluído no repositório. Faça o download pelo link abaixo.

---

## 🗂️ Dataset

- **Nome:** Wine Quality
- **Fonte:** [Kaggle — rajyellow46/wine-quality](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
- **Amostras:** 6.497 (1.599 tintos + 4.898 brancos)
- **Atributos:** 12 físico-químicos + 1 coluna alvo (`quality`)

### Classes utilizadas

| Classe | Notas originais | Amostras |
|--------|----------------|----------|
| Baixa  | 3, 4, 5        | ~2.384   |
| Média  | 6, 7           | ~3.915   |
| Alta   | 8, 9           | ~198     |

---

## ⚙️ Pré-processamento Aplicado

| Técnica | Motivo |
|--------|--------|
| Imputação pela mediana | 38 valores nulos em 7 colunas |
| One-Hot Encoding | Coluna `type` era categórica (red/white) |
| StandardScaler | KNN é sensível à escala dos atributos |
| Divisão estratificada 80/20 | Manter proporção das classes no treino e teste |

---

## 🔬 Experimentos

O script testa **60 combinações** (K de 1 a 15 × 4 distâncias):

| Distância | Parâmetro scikit-learn |
|-----------|----------------------|
| Euclidiana | `metric='euclidean'` |
| Manhattan  | `metric='manhattan'` |
| Chebyshev  | `metric='chebyshev'` |
| Minkowski  | `metric='minkowski', p=3` |

---

## 📊 Resultados

| Distância   | Melhor K | Acurácia |
|-------------|----------|----------|
| Euclidiana  | 1        | 0.7477   |
| Manhattan   | 1        | 0.7546   |
| Chebyshev   | 1        | 0.7369   |
| Minkowski   | 1        | 0.7438   |

**Melhor combinação: Manhattan com K=1 (acurácia: 0.7546)**

### Análise de Overfitting (K=1 vs K=15)

| Distância  | K=1    | K=15   | Diferença |
|------------|--------|--------|-----------|
| Euclidiana | 0.7477 | 0.7038 | +0.0438   |
| Manhattan  | 0.7546 | 0.7008 | +0.0538   |
| Chebyshev  | 0.7369 | 0.7023 | +0.0346   |
| Minkowski  | 0.7438 | 0.7131 | +0.0308   |

A diferença positiva consistente em todas as métricas indica **sinais de overfitting com K=1**.

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/davidrtc14/fund_IA_atividade.git
cd seu-repositorio
```

### 2. Instale as dependências
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Baixe o dataset
Acesse [este link](https://www.kaggle.com/datasets/rajyellow46/wine-quality), baixe o arquivo `winequalityN.csv` e coloque na pasta do projeto.

### 4. Execute o script
```bash
python atividade_knn_wine.py
```

### 5. Imagens geradas
Após a execução, os seguintes arquivos serão salvos na mesma pasta:

| Arquivo | Conteúdo |
|--------|---------|
| `eda_distribuicao_notas.png` | Distribuição das notas originais |
| `eda_distribuicao_classes.png` | Distribuição das classes criadas |
| `preprocessing_normalizacao.png` | Comparação antes/depois da normalização |
| `grafico_knn_acuracia.png` | Acurácia × K para cada distância |
| `matriz_confusao.png` | Matriz de confusão da melhor combinação |

---

## 🛠️ Dependências

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 👥 Integrantes

- David Ramalho Teixeira de Carvalho - RGM: 34262407
