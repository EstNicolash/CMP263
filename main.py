#!/usr/bin/env python
# coding: utf-8

# # Predição de Doença Cardíaca com Aprendizado de Máquina
# 
# **Dataset:** UCI Heart Disease – Cleveland
# 
# **Tarefa:** Classificação binária (0 = ausência, 1 = presença de doença cardíaca)
# 
# **Objetivo:** Comparar algoritmos de AM, otimizar hiperparâmetros, avaliar com métricas clínicas e interpretar o melhor modelo com SHAP.
# 
# **Reprodutibilidade:** seeds fixas, versões de bibliotecas registradas, notebook disponível.

# In[ ]:


# =====================================================================
# 0. CONFIGURAÇÃO INICIAL
# =====================================================================
# Importa as bibliotecas necessárias para análise de dados, visualização,
# modelagem e interpretabilidade.
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------
# Define uma seed global para que todos os processos aleatórios
# (divisão de dados, inicialização de modelos, etc.) sejam reproduzíveis.
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------
# Scikit-learn: pré-processamento, modelagem e métricas
# ---------------------------------------------------------------
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             ConfusionMatrixDisplay)

# ---------------------------------------------------------------
# Algoritmos de classificação
# ---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# ---------------------------------------------------------------
# Interpretabilidade
# ---------------------------------------------------------------
import shap

# ---------------------------------------------------------------
# Visualização: estilo dos gráficos
# ---------------------------------------------------------------
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# ---------------------------------------------------------------
# Versões das bibliotecas (para reprodutibilidade)
# ---------------------------------------------------------------
import sklearn, xgboost
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgboost.__version__}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")


# ## 1. Carregamento dos Dados
# 
# Baixa o dataset Cleveland do repositório UCI e separa o atributo alvo.

# In[ ]:


# =====================================================================
# 1. CARREGAMENTO DOS DADOS
# =====================================================================
# O dataset é baixado diretamente da URL pública do UCI.

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Nomes das 14 colunas (13 atributos + 1 alvo)
colunas = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

# Lê o CSV; '?' é convertido para NaN (valor ausente)
df = pd.read_csv(url, names=colunas, na_values='?')

# Visualiza as primeiras linhas para conferir o carregamento
df.head()


# In[ ]:


# =====================================================================
# BINARIZAÇÃO DO ATRIBUTO ALVO
# =====================================================================
# O alvo original varia de 0 a 4 (gravidade da doença).
# Convertemos para binário: 0 = ausência, 1 = presença (1 a 4).

df['target'] = (df['num'] > 0).astype(int)

# Remove a coluna original 'num'
df = df.drop(columns=['num'])

# Verifica o formato final e a distribuição das classes
print("Formato do dataset:", df.shape)
print("\nDistribuição do alvo:")
print(df['target'].value_counts())
print("\nProporção das classes:")
print(df['target'].value_counts(normalize=True))


# ## 2. Análise Exploratória dos Dados (EDA)
# 
# - Tipos de atributos
# - Valores faltantes
# - Distribuição do alvo
# - Estatísticas descritivas
# - Histogramas e boxplots
# - Correlações

# In[ ]:


# =====================================================================
# 2. ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# =====================================================================

# Informações gerais: tipos de dados e valores não nulos
print("Informações gerais do dataset:")
df.info()

# Estatísticas descritivas das variáveis numéricas
print("\nEstatísticas descritivas:")
df.describe().T


# In[ ]:


# =====================================================================
# VERIFICAÇÃO DE VALORES FALTANTES
# =====================================================================
# Conta os valores ausentes por atributo e calcula o percentual.

missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df)) * 100

# Tabela resumo apenas para atributos com valores faltantes
missing_df = pd.DataFrame({
    'Valores Faltantes': missing,
    'Percentual (%)': missing_pct
})
missing_df[missing_df['Valores Faltantes'] > 0]


# In[ ]:


# =====================================================================
# DISTRIBUIÇÃO DO ATRIBUTO ALVO
# =====================================================================
# Gráfico de barras mostrando o balanceamento entre as classes.

plt.figure(figsize=(7, 5))
sns.countplot(x='target', data=df)
plt.title('Distribuição do Alvo (0 = Ausência | 1 = Presença)')
plt.xlabel('Classe')
plt.ylabel('Número de Pacientes')
plt.xticks([0, 1], ['Sem doença', 'Com doença'])
plt.tight_layout()
plt.close()


# In[ ]:


# =====================================================================
# HISTOGRAMAS DAS VARIÁVEIS NUMÉRICAS
# =====================================================================
# Mostra a distribuição de cada variável numérica.

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

df[num_cols].hist(bins=20, figsize=(14, 10), edgecolor='black')
plt.suptitle('Histogramas das Variáveis Numéricas', fontsize=14, y=1.02)
plt.tight_layout()
plt.close()


# In[ ]:


# =====================================================================
# BOXPLOTS DAS VARIÁVEIS NUMÉRICAS POR CLASSE
# =====================================================================
# Compara a distribuição de cada variável entre pacientes com e sem doença.

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(num_cols):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('Classe')
    axes[i].set_ylabel('')
    axes[i].set_xticklabels(['Sem doença', 'Com doença'])

plt.suptitle('Distribuição das Variáveis Numéricas por Classe', fontsize=14, y=1.05)
plt.tight_layout()
plt.close()


# In[ ]:


# =====================================================================
# MATRIZ DE CORRELAÇÃO
# =====================================================================
# Calcula as correlações entre todas as variáveis (Spearman).
# Serve para identificar relações lineares e possíveis redundâncias.

plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Matriz de Correlação (Spearman)', fontsize=13)
plt.tight_layout()
plt.close()


# ## 3. Pré-processamento
# 
# - Separação entre X (atributos) e y (alvo)
# - Divisão treino/teste estratificada (80/20)
# - Pipeline de pré-processamento:
#   - Numéricas: imputação pela mediana + padronização
#   - Categóricas: imputação pela moda + one-hot encoding

# In[ ]:


# =====================================================================
# 3. PRÉ-PROCESSAMENTO
# =====================================================================

# Separa atributos (X) e alvo (y)
X = df.drop(columns=['target'])
y = df['target']

# Divide em treino (80%) e teste (20%), mantendo a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
print(f"\nProporção no treino:\n{y_train.value_counts(normalize=True)}")
print(f"\nProporção no teste:\n{y_test.value_counts(normalize=True)}")


# In[ ]:


# =====================================================================
# DEFINIÇÃO DAS COLUNAS E DO PRÉ-PROCESSADOR
# =====================================================================
# O ColumnTransformer permite aplicar transformações diferentes
# para colunas numéricas e categóricas, de forma encapsulada.

num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Pipeline para variáveis numéricas: imputa pela mediana e padroniza
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para variáveis categóricas: imputa pela moda e aplica one-hot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina os dois pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

preprocessor


# ## 4. Spot-checking
# 
# Comparação inicial de 6 algoritmos com vieses indutivos diferentes,
# usando validação cruzada estratificada de 5 folds.
# Métricas: AUC-ROC, Recall, F1, Acurácia, Precisão.

# In[ ]:


# =====================================================================
# 4. SPOT-CHECKING
# =====================================================================

# Dicionário com os 6 algoritmos a serem comparados
modelos = {
    'Regressão Logística': LogisticRegression(max_iter=1000, random_state=SEED),
    'Random Forest': RandomForestClassifier(random_state=SEED),
    'XGBoost': XGBClassifier(random_state=SEED, eval_metric='logloss'),
    'SVM (RBF)': SVC(probability=True, random_state=SEED),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Estratégia de validação cruzada: 5 folds estratificados
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Métricas a serem calculadas em cada fold
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'f1': 'f1',
    'accuracy': 'accuracy',
    'precision': 'precision'
}


# In[ ]:


# =====================================================================
# EXECUÇÃO DO SPOT-CHECKING
# =====================================================================
# Para cada algoritmo, treina e avalia via validação cruzada.
# A mesma seed é usada em todos os modelos para garantir comparabilidade.

resultados = []

for nome, modelo in modelos.items():
    # Pipeline que combina pré-processamento + classificador
    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('clf', modelo)
    ])

    # Avaliação com validação cruzada
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=scoring, return_train_score=False
    )

    # Armazena as médias dos resultados
    resultados.append({
        'Modelo': nome,
        'AUC-ROC': scores['test_roc_auc'].mean(),
        'Recall': scores['test_recall'].mean(),
        'F1': scores['test_f1'].mean(),
        'Acurácia': scores['test_accuracy'].mean(),
        'Precisão': scores['test_precision'].mean(),
        'AUC-ROC std': scores['test_roc_auc'].std()
    })

# Cria DataFrame ordenado por AUC-ROC (decrescente)
df_resultados = pd.DataFrame(resultados).sort_values('AUC-ROC', ascending=False)
df_resultados


# In[ ]:


# =====================================================================
# GRÁFICO COMPARATIVO DO SPOT-CHECKING
# =====================================================================
# Mostra lado a lado as principais métricas para cada algoritmo.

df_resultados.set_index('Modelo')[['AUC-ROC', 'Recall', 'F1', 'Acurácia']].plot(
    kind='bar', figsize=(12, 6), edgecolor='black'
)
plt.title('Spot-checking: Comparação de Modelos', fontsize=13)
plt.ylabel('Valor da Métrica')
plt.xlabel('Algoritmo')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.close()


# ## 5. Otimização de Hiperparâmetros
# 
# Otimização dos 3 melhores modelos com GridSearchCV.
# Mantém-se a mesma validação cruzada e seed para comparabilidade.

# In[ ]:


# =====================================================================
# 5. OTIMIZAÇÃO DE HIPERPARÂMETROS - RANDOM FOREST
# =====================================================================

param_grid_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

pipe_rf = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=SEED))
])

grid_rf = GridSearchCV(
    pipe_rf, param_grid_rf, cv=cv,
    scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_rf.fit(X_train, y_train)

print("Melhores parâmetros RF:", grid_rf.best_params_)
print(f"Melhor AUC-ROC (CV): {grid_rf.best_score_:.4f}")


# In[ ]:


# =====================================================================
# 5.2 OTIMIZAÇÃO DE HIPERPARÂMETROS - XGBOOST
# =====================================================================

param_grid_xgb = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0]
}

pipe_xgb = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss'))
])

grid_xgb = GridSearchCV(
    pipe_xgb, param_grid_xgb, cv=cv,
    scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_xgb.fit(X_train, y_train)

print("Melhores parâmetros XGB:", grid_xgb.best_params_)
print(f"Melhor AUC-ROC (CV): {grid_xgb.best_score_:.4f}")


# In[ ]:


# =====================================================================
# 5.3 OTIMIZAÇÃO DE HIPERPARÂMETROS - REGRESSÃO LOGÍSTICA
# =====================================================================

param_grid_logreg = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear'],
    'clf__class_weight': [None, 'balanced']
}

pipe_logreg_opt = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, random_state=SEED))
])

grid_logreg = GridSearchCV(
    pipe_logreg_opt, param_grid_logreg, cv=cv,
    scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_logreg.fit(X_train, y_train)

print("Melhores parâmetros LogReg:", grid_logreg.best_params_)
print(f"Melhor AUC-ROC (CV): {grid_logreg.best_score_:.4f}")


# In[ ]:


# =====================================================================
# SELEÇÃO DO MELHOR MODELO
# =====================================================================
# O melhor modelo é escolhido com base no AUC-ROC de validação cruzada.
# O conjunto de teste NÃO é usado nesta etapa.

candidatos = {
    'Regressão Logística': grid_logreg,
    'Random Forest': grid_rf,
    'XGBoost': grid_xgb
}

melhor_nome = max(candidatos, key=lambda k: candidatos[k].best_score_)
best_model = candidatos[melhor_nome].best_estimator_

print(f"Melhor modelo: {melhor_nome}")
print(f"AUC-ROC (CV): {candidatos[melhor_nome].best_score_:.4f}")


# ## 6. Avaliação Final
# 
# Avaliação do melhor modelo no conjunto de teste.

# In[ ]:


# =====================================================================
# 6. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# =====================================================================

# Predições: classe prevista e probabilidade da classe positiva
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Cálculo das métricas
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred):.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred,
                            target_names=['Sem doença', 'Com doença']))


# In[ ]:


# =====================================================================
# MATRIZ DE CONFUSÃO
# =====================================================================
# Mostra a distribuição dos acertos e erros do modelo.

plt.figure(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Sem doença', 'Com doença'],
    cmap='Blues',
    colorbar=False
)
plt.title('Matriz de Confusão - Modelo Final', fontsize=13)
plt.tight_layout()
plt.close()



# In[ ]:


# =====================================================================
# CURVA ROC DO MODELO FINAL
# =====================================================================

fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_valor = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', linewidth=2,
         label=f'AUC = {auc_valor:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC - Modelo Final', fontsize=13)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.close()


# In[ ]:


# =====================================================================
# CURVA ROC COMPARATIVA ENTRE OS PRINCIPAIS MODELOS
# =====================================================================
# Treina LogReg e SVM adicionais para a comparação gráfica.

pipe_logreg = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=SEED))
])
pipe_logreg.fit(X_train, y_train)

pipe_svm = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', SVC(probability=True, random_state=SEED))
])
pipe_svm.fit(X_train, y_train)

# Dicionário com os 4 modelos para comparação
modelos_roc = {
    'Regressão Logística': pipe_logreg,
    'Random Forest': grid_rf.best_estimator_,
    'SVM (RBF)': pipe_svm,
    'XGBoost': grid_xgb.best_estimator_
}

plt.figure(figsize=(9, 7))
for nome, modelo in modelos_roc.items():
    y_proba_i = modelo.predict_proba(X_test)[:, 1]
    fpr_i, tpr_i, _ = roc_curve(y_test, y_proba_i)
    auc_i = roc_auc_score(y_test, y_proba_i)
    plt.plot(fpr_i, tpr_i, linewidth=2,
             label=f'{nome} (AUC = {auc_i:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC - Comparação de Modelos', fontsize=13)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.close()


# ## 7. Interpretabilidade
# 
# Aplicação do SHAP para identificar os atributos mais relevantes.
# O explainer é escolhido de acordo com o tipo do modelo.

# In[ ]:


# =====================================================================
# 7. INTERPRETABILIDADE - PREPARAÇÃO
# =====================================================================

# Extrai o pré-processador e o classificador do pipeline
prep = best_model.named_steps['prep']
clf = best_model.named_steps['clf']

# Transforma os conjuntos de treino e teste usando o pré-processador
X_train_prep = prep.transform(X_train)
X_test_prep = prep.transform(X_test)

# Obtém os nomes das features após o one-hot encoding
feature_names = prep.get_feature_names_out()
print(f"Número de features após pré-processamento: {len(feature_names)}")


# In[ ]:


# =====================================================================
# 7.1 SHAP - VERSÃO ADAPTATIVA
# =====================================================================
# Escolhe o explainer correto de acordo com o tipo de modelo.

tipo_modelo = type(clf).__name__
print(f"Tipo do classificador: {tipo_modelo}")

if tipo_modelo in ['RandomForestClassifier', 'XGBClassifier',
                   'GradientBoostingClassifier', 'DecisionTreeClassifier',
                   'ExtraTreesClassifier']:
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_prep)
elif tipo_modelo in ['LogisticRegression', 'LinearSVC', 'Ridge', 'Lasso']:
    explainer = shap.LinearExplainer(clf, X_train_prep)
    shap_values = explainer.shap_values(X_test_prep)
else:
    explainer = shap.KernelExplainer(clf.predict_proba, X_train_prep[:100])
    shap_values = explainer.shap_values(X_test_prep[:100])

print(f"Tipo: {type(shap_values)}")
print(f"Shape: {np.shape(shap_values)}")

# Normaliza para array 2D (amostras x features) da classe positiva
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values_pos = shap_values[:, :, 1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
    shap_values_pos = shap_values
else:
    raise ValueError(f"Formato não reconhecido: {np.shape(shap_values)}")

print(f"Shape shap_values_pos: {shap_values_pos.shape}")
print(f"Shape X_test_prep: {X_test_prep.shape}")


# In[ ]:


# =====================================================================
# 7.2 SHAP - SUMMARY PLOT (BEESWARM)
# =====================================================================
# Mostra o impacto de cada atributo nas predições.

shap.summary_plot(shap_values_pos, X_test_prep, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()


# In[ ]:


# =====================================================================
# 7.3 SHAP - IMPORTÂNCIA GLOBAL
# =====================================================================
# Mostra a importância média absoluta de cada atributo.

shap.summary_plot(shap_values_pos, X_test_prep,
                  feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig( FIGURES_DIR / 'shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()


# In[ ]:


# =====================================================================
# 7.4 ATRIBUTO MAIS IMPORTANTE SEGUNDO SHAP
# =====================================================================

mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
top_feature_idx = np.argmax(mean_abs_shap)
top_feature_name = feature_names[top_feature_idx]
print(f"Atributo mais importante: {top_feature_name}")


# In[ ]:


# =====================================================================
# 7.5 SHAP - DEPENDENCE PLOT
# =====================================================================
# Mostra como o valor do atributo mais importante afeta a predição.

shap.dependence_plot(top_feature_idx, shap_values_pos, X_test_prep,
                     feature_names=feature_names, show=False)
plt.tight_layout()
plt.close()


# ## 8. Importância por Permutação
# 
# Método complementar ao SHAP para validar a relevância dos atributos.

# In[ ]:


# =====================================================================
# 8. IMPORTÂNCIA POR PERMUTAÇÃO
# =====================================================================
# Mede quanto o desempenho cai quando cada atributo é embaralhado.

from sklearn.inspection import permutation_importance

result = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=30, random_state=SEED, scoring='roc_auc'
)

importancias = pd.DataFrame({
    'Atributo': X_test.columns,
    'Importância Média': result.importances_mean,
    'Desvio Padrão': result.importances_std
}).sort_values('Importância Média', ascending=False).reset_index(drop=True)

print(importancias.head(10))


# In[ ]:


# =====================================================================
# GRÁFICO DA IMPORTÂNCIA POR PERMUTAÇÃO
# =====================================================================

top10 = importancias.head(10).iloc[::-1]  # Inverte para ordenar no gráfico

plt.figure(figsize=(10, 7))
plt.barh(top10['Atributo'], top10['Importância Média'],
         xerr=top10['Desvio Padrão'], color='steelblue', edgecolor='black')
plt.xlabel('Importância por Permutação (Δ AUC-ROC)')
plt.title('Top 10 Atributos - Importância por Permutação', fontsize=13)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'permutation_importance.png', dpi=150, bbox_inches='tight')
plt.close()


# ## 9. Análise de Threshold
# 
# Avaliação de como recall, precisão e F1 variam conforme o ponto de corte.

# In[ ]:


# =====================================================================
# 9. ANÁLISE DE THRESHOLD
# =====================================================================
# Em contexto clínico, o limiar de decisão pode ser ajustado conforme
# o objetivo (triagem vs. confirmação diagnóstica).

thresholds = np.arange(0.1, 1.0, 0.05)
resultados_thr = []

for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    resultados_thr.append({
        'Threshold': thr,
        'Recall': recall_score(y_test, y_pred_thr),
        'Precisão': precision_score(y_test, y_pred_thr),
        'F1': f1_score(y_test, y_pred_thr)
    })

df_thr = pd.DataFrame(resultados_thr)
print(df_thr)


# In[ ]:


# =====================================================================
# GRÁFICO DE MÉTRICAS VS THRESHOLD
# =====================================================================

plt.figure(figsize=(10, 6))
plt.plot(df_thr['Threshold'], df_thr['Recall'], label='Recall',
         color='steelblue', linewidth=2)
plt.plot(df_thr['Threshold'], df_thr['Precisão'], label='Precisão',
         color='darkorange', linewidth=2)
plt.plot(df_thr['Threshold'], df_thr['F1'], label='F1',
         color='green', linewidth=2)
plt.axvline(0.5, color='gray', linestyle='--',
            label='Threshold padrão (0.5)')
plt.xlabel('Threshold de Decisão')
plt.ylabel('Valor da Métrica')
plt.title('Métricas vs Threshold de Decisão', fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.close()


# ## 10. Reprodutibilidade
# 
# - Seeds fixas em todas as etapas
# - Versões de bibliotecas registradas
# - Modelo final salvo
# - Resultados exportados em CSV

# In[ ]:


# =====================================================================
# 10. SALVAMENTO DE MODELO E RESULTADOS
# =====================================================================

import json
import joblib
import pandas as pd

# Define os caminhos a partir da raiz do projeto (um nível acima de notebooks/)
# ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

MODELS_DIR = ROOT_DIR / "models"
PARAMS_DIR = MODELS_DIR / "params"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
# FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Cria os diretórios caso ainda não existam
for path in [MODELS_DIR, PARAMS_DIR, METRICS_DIR, FIGURES_DIR]:
  path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Métricas e Tabelas (reports/metrics/)
# ---------------------------------------------------------------------
metricas_finais = {
    "Modelo Escolhido": melhor_nome,
    "AUC-ROC CV": candidatos[melhor_nome].best_score_,
    "AUC-ROC Teste": roc_auc_score(y_test, y_proba),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "Acurácia": accuracy_score(y_test, y_pred),
    "Precisão": precision_score(y_test, y_pred),
}

pd.DataFrame([metricas_finais]).to_csv(
    METRICS_DIR / "metricas_finais.csv", index=False
)
df_resultados.to_csv(
    METRICS_DIR / "resultados_spot_checking.csv", index=False
)
importancias.to_csv(
    METRICS_DIR / "permutation_importance.csv", index=False
)

# ---------------------------------------------------------------------
# 2. Modelo Treinado (models/)
# ---------------------------------------------------------------------
joblib.dump(best_model, MODELS_DIR / "melhor_modelo_heart_disease.pkl")

# ---------------------------------------------------------------------
# 3. Hiperparâmetros Ótimos (models/params/)
# ---------------------------------------------------------------------
with open(PARAMS_DIR / "best_params_rf.json", "w", encoding="utf-8") as f:
  json.dump(grid_rf.best_params_, f, indent=2, default=str)

with open(PARAMS_DIR / "best_params_xgb.json", "w", encoding="utf-8") as f:
  json.dump(grid_xgb.best_params_, f, indent=2, default=str)

with open(PARAMS_DIR / "best_params_logreg.json", "w", encoding="utf-8") as f:
  json.dump(grid_logreg.best_params_, f, indent=2, default=str)

print(f"Artefatos salvos com sucesso em:\n - {MODELS_DIR}\n - {METRICS_DIR}")


# In[ ]:


# =====================================================================
# RESUMO FINAL
# =====================================================================

print("=" * 60)
print("RESUMO FINAL DO TRABALHO")
print("=" * 60)
print(f"\nMelhor modelo: {melhor_nome}")
print(f"\nMétricas no conjunto de teste:")
for k, v in metricas_finais.items():
    if k != 'Modelo Escolhido':
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

print("\nArquivos salvos:")
for arq in [
    'shap_summary.png',
    'shap_bar.png',
    'permutation_importance.png',
    'metricas_finais.csv',
    'resultados_spot_checking.csv',
    'permutation_importance.csv',
    'melhor_modelo_heart_disease.pkl',
    'best_params_rf.json',
    'best_params_xgb.json',
    'best_params_logreg.json'
]:
    print(f"  - {arq}")
