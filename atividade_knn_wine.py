# =============================================================
# ATIVIDADE PRÁTICA — Inteligência Artificial
# Algoritmo KNN — Wine Quality Dataset
# =============================================================
# Dataset: Wine Quality (Red + White combinados)
# Link: https://www.kaggle.com/datasets/rajyellow46/wine-quality
# Target: qualidade agrupada em 3 classes
#   Baixa  = notas 3, 4, 5
#   Média  = notas 6, 7
#   Alta   = notas 8, 9
# =============================================================

# ============================================================
# 1. IMPORTAÇÕES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# ============================================================
# 2. CARREGAMENTO DO DATASET
# ============================================================


df = pd.read_csv('winequalityN.csv')

print("=== INFORMAÇÕES GERAIS ===")
print(f"Shape: {df.shape}")
print(f"\nColunas: {list(df.columns)}")
print(f"\nTipos:\n{df.dtypes}")
print(f"\nPrimeiras linhas:\n{df.head()}")

# ============================================================
# 3. ANÁLISE EXPLORATÓRIA (EDA)
# ============================================================
print("\n=== ANÁLISE EXPLORATÓRIA ===")
print(df.describe())

print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

print(f"\nDistribuição da coluna 'quality':\n{df['quality'].value_counts().sort_index()}")

print(f"\nDistribuição por tipo de vinho:\n{df['type'].value_counts()}")

# Criação de Gráfico: distribuição das notas originais
plt.figure(figsize=(8, 4))
df['quality'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Distribuição das Notas de Qualidade (original)')
plt.xlabel('Nota')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.savefig('eda_distribuicao_notas.png', dpi=150)
plt.show()

# ============================================================
# 4. PRÉ-PROCESSAMENTO
# ============================================================

# ─── 4.1 Criar a coluna alvo (target) ────────────────────────
# Agrupa notas em 3 classes interpretáveis
def classificar_qualidade(nota):
    if nota <= 5:
        return 'Baixa'
    elif nota <= 7:
        return 'Média'
    else:
        return 'Alta'

df['quality_class'] = df['quality'].apply(classificar_qualidade)

print("\n=== Distribuição das classes criadas ===")
print(df['quality_class'].value_counts())

# Criação de Gráfico: distribuição das classes criadas
plt.figure(figsize=(6, 4))
df['quality_class'].value_counts().plot(kind='bar', color=['#4CAF50','#FFC107','#F44336'], edgecolor='black')
plt.title('Distribuição das Classes (após agrupamento)')
plt.xlabel('Classe')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.savefig('eda_distribuicao_classes.png', dpi=150)
plt.show()

# ─── 4.2 Separar features e target ───────────────────────────
# Remove colunas desnecessárias para o modelo
X = df.drop(columns=['quality', 'quality_class'])
y = df['quality_class']

# ─── 4.3 Codificação de variável categórica (One-Hot Encoding) ─
# Justificativa: a coluna 'type' é categórica (red/white).
# O KNN calcula distâncias numéricas, portanto precisamos converter.
# Usamos get_dummies (One-Hot) para evitar ordenação implícita.
print("\n=== PRÉ-PROCESSAMENTO: Codificação ===")
print("Antes do encoding — colunas categóricas:", X.select_dtypes(include='object').columns.tolist())

X = pd.get_dummies(X, columns=['type'], drop_first=True)
# Resultado: cria coluna 'type_white' (1=branco, 0=tinto)

print("Após encoding — shape:", X.shape)
print("Colunas:", list(X.columns))

# ─── 4.4 Tratamento de Missing Values ────────────────────────
# Justificativa: mesmo que este dataset dificilmente tenha nulos,
# é obrigatório verificar e tratar para garantir robustez.
print("\n=== PRÉ-PROCESSAMENTO: Missing Values ===")
print("Nulos antes da imputação:\n", X.isnull().sum())

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("Nulos após a imputação:\n", X_imputed.isnull().sum())

# ─── 4.5 Normalização (StandardScaler) ───────────────────────
# Justificativa: KNN é sensível a escala. Atributos como
# 'total sulfur dioxide' (0–300) dominariam a distância
# sobre atributos como 'pH' (2.8–4.0). A padronização
# (média=0, desvio=1) elimina esse problema.
print("\n=== PRÉ-PROCESSAMENTO: Normalização ===")
print("Estatísticas ANTES da normalização:")
print(X_imputed[['alcohol', 'total sulfur dioxide', 'pH']].describe().round(3))

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

print("\nEstatísticas APÓS a normalização (média≈0, desvio≈1):")
print(X_scaled[['alcohol', 'total sulfur dioxide', 'pH']].describe().round(3))

# Criação de Gráfico: comparação antes/depois da normalização para 'alcohol'
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(X_imputed['alcohol'], bins=30, color='steelblue', edgecolor='black')
axes[0].set_title('alcohol — ANTES da normalização')
axes[0].set_xlabel('Valor original')

axes[1].hist(X_scaled['alcohol'], bins=30, color='darkorange', edgecolor='black')
axes[1].set_title('alcohol — APÓS normalização')
axes[1].set_xlabel('Valor padronizado (z-score)')

plt.tight_layout()
plt.savefig('preprocessing_normalizacao.png', dpi=150)
plt.show()

# ─── 4.6 Codificação do target ────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nClasses codificadas:", dict(zip(le.classes_, le.transform(le.classes_))))

# ============================================================
# 5. DIVISÃO TREINO / TESTE (80/20 estratificada)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded   # isso mantém proporção das classes
)

print(f"\n=== DIVISÃO TREINO/TESTE ===")
print(f"Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"Teste:  {X_test.shape[0]} amostras ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")

# Verifica balanceamento nas partições
classes, contagens_treino = np.unique(y_train, return_counts=True)
_, contagens_teste = np.unique(y_test, return_counts=True)
print("\nProporção das classes no treino:", dict(zip(le.classes_, (contagens_treino / len(y_train) * 100).round(1))))
print("Proporção das classes no teste: ", dict(zip(le.classes_, (contagens_teste  / len(y_test)  * 100).round(1))))

# ============================================================
# 6. EXPERIMENTAÇÃO KNN
# ============================================================
k_range = range(1, 16)

distancias = {
    'Euclidiana': {'metric': 'euclidean'},
    'Manhattan':  {'metric': 'manhattan'},
    'Chebyshev':  {'metric': 'chebyshev'},
    'Minkowski':  {'metric': 'minkowski', 'p': 3},
}

resultados = {}

print("\n=== RODANDO KNN ===")
for nome, params in distancias.items():
    acuracias = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, **params)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acuracias.append(acc)
    resultados[nome] = acuracias
    melhor_k   = list(k_range)[np.argmax(acuracias)]
    melhor_acc = max(acuracias)
    print(f"{nome:12s} → melhor K={melhor_k:2d} | acurácia={melhor_acc:.4f}")

# ============================================================
# 7. GRÁFICO DE DESEMPENHO
# ============================================================
cores = {
    'Euclidiana': '#1f77b4',
    'Manhattan':  '#ff7f0e',
    'Chebyshev':  '#2ca02c',
    'Minkowski':  '#d62728',
}

plt.figure(figsize=(12, 6))
for nome, acuracias in resultados.items():
    plt.plot(list(k_range), acuracias, marker='o', label=nome, color=cores[nome], linewidth=2)

# Criação de Gráfico: Gráfico de desempenho/acurácia

plt.title('Acurácia × Valor de K por Métrica de Distância\n(Wine Quality — 3 classes)', fontsize=13)
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.xticks(list(k_range))
plt.legend(title='Métrica de Distância')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('grafico_knn_acuracia.png', dpi=150)
plt.show()

# ============================================================
# 8. MELHOR COMBINAÇÃO — RELATÓRIO FINAL
# ============================================================
print("\n=== MELHOR COMBINAÇÃO (K + Distância) ===")
melhor_global_acc  = 0
melhor_global_k    = None
melhor_global_dist = None

for nome, acuracias in resultados.items():
    melhor_k   = list(k_range)[np.argmax(acuracias)]
    melhor_acc = max(acuracias)
    print(f"  {nome:12s} → K={melhor_k:2d} | acurácia={melhor_acc:.4f}")
    if melhor_acc > melhor_global_acc:
        melhor_global_acc  = melhor_acc
        melhor_global_k    = melhor_k
        melhor_global_dist = nome

print(f"\n>>> VENCEDOR: distância={melhor_global_dist} | K={melhor_global_k} | acurácia={melhor_global_acc:.4f}")

# ─── Matriz de confusão da melhor configuração ───────────────
params_melhor = distancias[melhor_global_dist]
knn_final = KNeighborsClassifier(n_neighbors=melhor_global_k, **params_melhor)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)

print(f"\n=== RELATÓRIO DETALHADO — {melhor_global_dist} K={melhor_global_k} ===")
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

# Criação de Gráfico: Matriz de confusão da melhor configuração

cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Matriz de Confusão\n{melhor_global_dist} | K={melhor_global_k}')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig('matriz_confusao.png', dpi=150)
plt.show()

# ─── Análise de overfitting: K=1 vs K alto ───────────────────
print("\n=== ANÁLISE DE OVERFITTING ===")
for nome, acuracias in resultados.items():
    acc_k1    = acuracias[0]
    acc_k_max = acuracias[-1]
    print(f"{nome:12s} | K=1: {acc_k1:.4f} | K=15: {acc_k_max:.4f} | diff: {acc_k1 - acc_k_max:+.4f}")

# =============================================================
# FIM DO SCRIPT
# =============================================================
# Arquivos gerados:
#   eda_distribuicao_notas.png
#   eda_distribuicao_classes.png
#   preprocessing_normalizacao.png
#   grafico_knn_acuracia.png
#   matriz_confusao.png
# =============================================================
