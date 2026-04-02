import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore")

# Estilo dos gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)


# =============================================================================
# ETAPA 1 — COLETA E CARREGAMENTO DOS DADOS
# =============================================================================
print("=" * 60)
print("ETAPA 1 — CARREGAMENTO DOS DADOS")
print("=" * 60)

URLS = [
    "https://raw.githubusercontent.com/selva86/datasets/master/student-mat.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv",
]

df = None

for url in URLS:
    try:
        df = pd.read_csv(url, sep=";")
        print(f"✅ Dataset carregado da URL com sucesso!")
        break
    except Exception:
        continue


if df is None:
    try:
        df = pd.read_csv("student-mat.csv", sep=";")
        print("✅ Dataset carregado do arquivo local!")
    except FileNotFoundError:
        pass


if df is None:
    print("⚠️  Dataset não encontrado online nem localmente.")
    print("✅ Gerando dataset sintético com as mesmas variáveis do Student Performance Dataset...\n")

    np.random.seed(42)
    n = 395

    Medu = np.random.randint(0, 5, n)
    Fedu = np.random.randint(0, 5, n)
    studytime = np.random.randint(1, 5, n)
    failures = np.random.choice([0, 1, 2, 3], n, p=[0.67, 0.20, 0.08, 0.05])
    absences = np.random.randint(0, 30, n)
    G1 = np.clip(np.random.normal(11, 3, n) + studytime * 0.5 - failures * 1.5, 0, 20).astype(int)
    G2 = np.clip(G1 + np.random.normal(0, 1.5, n), 0, 20).astype(int)
    G3 = np.clip(G2 + np.random.normal(0, 1.2, n) - absences * 0.05, 0, 20).astype(int)

    df = pd.DataFrame({
        "school":     np.random.choice(["GP", "MS"], n),
        "sex":        np.random.choice(["M", "F"], n),
        "age":        np.random.randint(15, 22, n),
        "address":    np.random.choice(["U", "R"], n),
        "famsize":    np.random.choice(["LE3", "GT3"], n),
        "Pstatus":    np.random.choice(["T", "A"], n),
        "Medu":       Medu,
        "Fedu":       Fedu,
        "Mjob":       np.random.choice(["teacher","health","services","at_home","other"], n),
        "Fjob":       np.random.choice(["teacher","health","services","at_home","other"], n),
        "reason":     np.random.choice(["home","reputation","course","other"], n),
        "guardian":   np.random.choice(["mother","father","other"], n),
        "traveltime": np.random.randint(1, 5, n),
        "studytime":  studytime,
        "failures":   failures,
        "schoolsup":  np.random.choice(["yes","no"], n),
        "famsup":     np.random.choice(["yes","no"], n),
        "paid":       np.random.choice(["yes","no"], n),
        "activities": np.random.choice(["yes","no"], n),
        "nursery":    np.random.choice(["yes","no"], n),
        "higher":     np.random.choice(["yes","no"], n, p=[0.85, 0.15]),
        "internet":   np.random.choice(["yes","no"], n, p=[0.80, 0.20]),
        "romantic":   np.random.choice(["yes","no"], n),
        "famrel":     np.random.randint(1, 6, n),
        "freetime":   np.random.randint(1, 6, n),
        "goout":      np.random.randint(1, 6, n),
        "Dalc":       np.random.randint(1, 6, n),
        "Walc":       np.random.randint(1, 6, n),
        "health":     np.random.randint(1, 6, n),
        "absences":   absences,
        "G1":         G1,
        "G2":         G2,
        "G3":         G3,
    })

print(f"   Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
print("\nPrimeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())


# =============================================================================
# ETAPA 2 — PRÉ-PROCESSAMENTO E LIMPEZA DOS DADOS
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 2 — PRÉ-PROCESSAMENTO")
print("=" * 60)

# Verificação de valores nulos
print("\nValores nulos por coluna:")
print(df.isnull().sum())

# Criação da variável-alvo binária
# Aprovado (1): nota final G3 >= 10 | Reprovado (0): G3 < 10
df["aprovado"] = (df["G3"] >= 10).astype(int)
print(f"\nDistribuição da variável-alvo:")
print(df["aprovado"].value_counts())
print(f"  Aprovados: {df['aprovado'].sum()} ({df['aprovado'].mean()*100:.1f}%)")
print(f"  Reprovados: {(df['aprovado'] == 0).sum()} ({(1 - df['aprovado'].mean())*100:.1f}%)")

# Codificação de variáveis categóricas binárias (yes/no → 1/0)
colunas_binarias = ["schoolsup", "famsup", "paid", "activities",
                    "nursery", "higher", "internet", "romantic"]
le = LabelEncoder()
for col in colunas_binarias:
    df[col] = le.fit_transform(df[col])

# One-Hot Encoding para variáveis categóricas com múltiplas categorias
colunas_categoricas = ["school", "sex", "address", "famsize",
                       "Pstatus", "Mjob", "Fjob", "reason", "guardian"]
df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

print(f"\nDataset após encoding: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Definição das features (X) e alvo (y)

colunas_excluir = ["G3", "aprovado"]
X = df.drop(columns=colunas_excluir)
y = df["aprovado"]

print(f"\nFeatures utilizadas ({X.shape[1]}):")
print(list(X.columns))

# Normalização das variáveis numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# =============================================================================
# ETAPA 3 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 3 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análise Exploratória de Dados", fontsize=16, fontweight="bold")

# Gráfico 1: Distribuição da nota final G3
axes[0, 0].hist(df["G3"] if "G3" in df.columns else
                df.filter(like="G3").iloc[:, 0],
                bins=20, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Distribuição da Nota Final (G3)")
axes[0, 0].set_xlabel("Nota Final")
axes[0, 0].set_ylabel("Frequência")

# Gráfico 2: Aprovados vs Reprovados
contagem = y.value_counts()
axes[0, 1].bar(["Reprovados", "Aprovados"], contagem.values,
               color=["salmon", "mediumseagreen"], edgecolor="white")
axes[0, 1].set_title("Aprovados vs Reprovados")
axes[0, 1].set_ylabel("Quantidade")
for i, v in enumerate(contagem.values):
    axes[0, 1].text(i, v + 2, str(v), ha="center", fontweight="bold")

# Gráfico 3: Horas de estudo vs Aprovação
estudo_aprovacao = df.groupby(["studytime", "aprovado"]).size().unstack(fill_value=0)
estudo_aprovacao.plot(kind="bar", ax=axes[1, 0],
                      color=["salmon", "mediumseagreen"], edgecolor="white")
axes[1, 0].set_title("Horas de Estudo vs Aprovação")
axes[1, 0].set_xlabel("Horas de Estudo (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
axes[1, 0].set_ylabel("Quantidade")
axes[1, 0].legend(["Reprovado", "Aprovado"])
axes[1, 0].tick_params(axis="x", rotation=0)

# Gráfico 4: Reprovações anteriores vs Nota final
if "failures" in X.columns:
    sns.boxplot(data=df, x="failures", y="G3" if "G3" not in df.columns
                else "G3", ax=axes[1, 1], palette="muted")
    axes[1, 1].set_title("Reprovações Anteriores vs Nota Final")
    axes[1, 1].set_xlabel("Número de Reprovações Anteriores")
    axes[1, 1].set_ylabel("Nota Final (G3)")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Gráfico EDA salvo como 'eda_plots.png'")

# Mapa de correlação das variáveis numéricas principais
variaveis_numericas = ["age", "Medu", "Fedu", "traveltime", "studytime",
                       "failures", "famrel", "freetime", "goout",
                       "Dalc", "Walc", "health", "absences", "G1", "G2", "aprovado"]
variaveis_presentes = [v for v in variaveis_numericas if v in df.columns]

plt.figure(figsize=(12, 9))
corr = df[variaveis_presentes].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Mapa de Correlação entre Variáveis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlacao.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Mapa de correlação salvo como 'correlacao.png'")


# =============================================================================
# ETAPA 4 — DESENVOLVIMENTO DO MODELO PREDITIVO
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 4 — DESENVOLVIMENTO DO MODELO PREDITIVO")
print("=" * 60)

# Divisão treino/teste (80/20) com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nDivisão dos dados:")
print(f"  Treino: {X_train.shape[0]} amostras")
print(f"  Teste:  {X_test.shape[0]} amostras")

# Definição dos modelos
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árvore de Decisão":   DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

resultados = {}

print("\nTreinando e avaliando modelos...\n")

for nome, modelo in modelos.items():
    # Treinamento
    modelo.fit(X_train, y_train)

    # Predição
    y_pred = modelo.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring="f1")
    f1_cv = cv_scores.mean()

    resultados[nome] = {
        "modelo": modelo,
        "y_pred": y_pred,
        "acuracia": acc,
        "f1_cv": f1_cv
    }

    print(f"{'─'*50}")
    print(f"🤖 {nome}")
    print(f"   Acurácia no teste : {acc:.4f} ({acc*100:.1f}%)")
    print(f"   F1-Score (CV-5)   : {f1_cv:.4f} ({f1_cv*100:.1f}%)")
    print(f"\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Reprovado", "Aprovado"]))


# =============================================================================
# ETAPA 5 — AVALIAÇÃO E COMPARAÇÃO DOS MODELOS
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 5 — AVALIAÇÃO E COMPARAÇÃO DOS MODELOS")
print("=" * 60)

# Gráfico de comparação de acurácia
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Matrizes de Confusão por Modelo", fontsize=14, fontweight="bold")

for ax, (nome, res) in zip(axes, resultados.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Reprovado", "Aprovado"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{nome}\nAcurácia: {res['acuracia']*100:.1f}%")

plt.tight_layout()
plt.savefig("matrizes_confusao.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Matrizes de confusão salvas como 'matrizes_confusao.png'")

# Comparativo geral
print("\n📊 Comparativo Final dos Modelos:")
print(f"{'Modelo':<25} {'Acurácia':>10} {'F1-Score (CV)':>15}")
print("─" * 52)
melhor_modelo = max(resultados.items(), key=lambda x: x[1]["f1_cv"])
for nome, res in resultados.items():
    star = " ⭐" if nome == melhor_modelo[0] else ""
    print(f"{nome:<25} {res['acuracia']*100:>9.1f}% {res['f1_cv']*100:>14.1f}%{star}")

print(f"\n✅ Melhor modelo: {melhor_modelo[0]}")


# =============================================================================
# ETAPA 6 — IMPORTÂNCIA DAS FEATURES (Random Forest)
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 6 — IMPORTÂNCIA DAS FEATURES")
print("=" * 60)

rf_model = resultados["Random Forest"]["modelo"]
importancias = pd.Series(rf_model.feature_importances_, index=X.columns)
importancias_top = importancias.nlargest(15).sort_values()

plt.figure(figsize=(10, 7))
importancias_top.plot(kind="barh", color="steelblue", edgecolor="white")
plt.title("Top 15 Features Mais Importantes (Random Forest)",
          fontsize=13, fontweight="bold")
plt.xlabel("Importância")
plt.tight_layout()
plt.savefig("importancia_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Gráfico de importância salvo como 'importancia_features.png'")

print("\nTop 10 features mais importantes:")
for feat, imp in importancias.nlargest(10).items():
    print(f"  {feat:<30} {imp:.4f}")


# =============================================================================
# ETAPA 7 — PREDIÇÃO PARA NOVOS ESTUDANTES
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 7 — EXEMPLO DE PREDIÇÃO")
print("=" * 60)

# Exemplo de predição com o melhor modelo
modelo_final = melhor_modelo[1]["modelo"]

# Usa as primeiras 5 amostras do conjunto de teste como exemplo
amostras_exemplo = X_test.iloc[:5]
predicoes = modelo_final.predict(amostras_exemplo)
probabilidades = modelo_final.predict_proba(amostras_exemplo)

print("\nPredições para 5 estudantes do conjunto de teste:")
print(f"{'#':<5} {'Predição':<15} {'Prob. Reprovado':>17} {'Prob. Aprovado':>16} {'Real':>8}")
print("─" * 65)
for i, (pred, prob, real) in enumerate(zip(predicoes, probabilidades, y_test.iloc[:5])):
    status = "Aprovado" if pred == 1 else "Reprovado"
    real_status = "Aprovado" if real == 1 else "Reprovado"
    acerto = "✅" if pred == real else "❌"
    print(f"{i+1:<5} {status:<15} {prob[0]*100:>16.1f}% {prob[1]*100:>15.1f}% {real_status:>8} {acerto}")

print("\n" + "=" * 60)
print("✅ PROJETO CONCLUÍDO COM SUCESSO!")
print("Arquivos gerados:")
print("  📊 eda_plots.png")
print("  📊 correlacao.png")
print("  📊 matrizes_confusao.png")
print("  📊 importancia_features.png")
print("=" * 60)