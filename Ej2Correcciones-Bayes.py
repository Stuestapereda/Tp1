import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import re
import string
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from itertools import chain

# Descargar recursos de NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')

# 1. Cargar el conjunto de datos con el delimitador correcto
df = pd.read_csv('noticias_argentinas.csv', sep=';', usecols=['titular', 'categoria'], encoding='utf-8')

# 2. Limpieza de datos: Eliminar filas con 'categoria' faltante
df = df.dropna(subset=['categoria'])

# 3. Filtrar las categorías seleccionadas
selected_categories = ['Noticias destacadas', 'Nacional', 'Deportes', 'Economia']
df = df[df['categoria'].isin(selected_categories)]

# 4. Balancear el conjunto de datos si es necesario
category_counts = df['categoria'].value_counts()
min_count = category_counts.min()
df_balanced = df.groupby('categoria').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

# 5. Preprocesamiento de texto mejorado
stop_words = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    # Eliminar espacios extra
    text = text.strip()
    # Tokenización
    tokens = text.split()
    # Eliminar stopwords y lematizar
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

df_balanced['tokens'] = df_balanced['titular'].apply(preprocess)

# 6. Generar N-gramas (Bigramas)
from nltk import ngrams

def generate_ngrams(tokens, n=2):
    return ['_'.join(gram) for gram in ngrams(tokens, n)]

df_balanced['bigrams'] = df_balanced['tokens'].apply(lambda tokens: generate_ngrams(tokens, n=2))
df_balanced['tokens'] = df_balanced['tokens'] + df_balanced['bigrams']

# 7. División en conjunto de entrenamiento y prueba
X = df_balanced['tokens']
y = df_balanced['categoria']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 8. Implementación del clasificador Ingenuo de Bayes optimizado
class NaiveBayesClassifier:
    def __init__(self):
        self.prior = {}
        self.likelihood = {}
        self.vocab = set()
        self.classes = []
        self.class_word_counts = {}
        self.vocab_size = 0

    def train(self, X, y):
        self.classes = np.unique(y)
        class_counts = y.value_counts().to_dict()
        total_count = len(y)
        
        # Calcular probabilidades a priori
        self.prior = {cls: np.log(count / total_count) for cls, count in class_counts.items()}
        
        # Calcular frecuencias de palabras por clase
        self.likelihood = {cls: defaultdict(int) for cls in self.classes}
        self.class_word_counts = {cls: 0 for cls in self.classes}
        
        for tokens, cls in zip(X, y):
            word_counts = Counter(tokens)
            for word, count in word_counts.items():
                self.likelihood[cls][word] += count
                self.class_word_counts[cls] += count
                self.vocab.add(word)
        
        self.vocab_size = len(self.vocab)

    def predict(self, tokens):
        class_scores = {}
        token_counts = Counter(tokens)
        for cls in self.classes:
            log_prob = self.prior[cls]
            for token, count in token_counts.items():
                token_freq = self.likelihood[cls].get(token, 0)
                log_prob += count * np.log((token_freq + 1) / (self.class_word_counts[cls] + self.vocab_size))
            class_scores[cls] = log_prob
        return max(class_scores, key=class_scores.get)

    def predict_batch(self, X):
        return [self.predict(tokens) for tokens in X]

    def get_probabilities(self, tokens):
        class_scores = {}
        token_counts = Counter(tokens)
        for cls in self.classes:
            log_prob = self.prior[cls]
            for token, count in token_counts.items():
                token_freq = self.likelihood[cls].get(token, 0)
                log_prob += count * np.log((token_freq + 1) / (self.class_word_counts[cls] + self.vocab_size))
            class_scores[cls] = log_prob
        # Convertir log-probabilidades a probabilidades
        max_log = max(class_scores.values())
        exp_scores = {cls: np.exp(score - max_log) for cls, score in class_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {cls: score / total for cls, score in exp_scores.items()}
        return probabilities

# 9. Entrenar el clasificador
nb_classifier = NaiveBayesClassifier()
nb_classifier.train(X_train, y_train)

# 10. Predecir en el conjunto de prueba
y_pred = nb_classifier.predict_batch(X_test)

# 11. Construir la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred, labels=selected_categories)
conf_matrix_df = pd.DataFrame(conf_matrix, index=selected_categories, columns=selected_categories)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, cmap='YlOrRd', fmt='d')
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 12. Calcular medidas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Visualizar las métricas generales
metrics = ['Accuracy', 'Precisión', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
sns.barplot(x=metrics, y=values, palette='viridis')
plt.title('Métricas de Evaluación General')
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.savefig('general_metrics.png')
plt.close()

# 13. Calcular métricas por clase
def compute_metrics_per_class(y_true, y_pred, classes):
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        
        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls) if (precision_cls + recall_cls) >0 else 0
        accuracy_cls = (tp + tn) / (tp + tn + fp + fn)
        
        metrics[cls] = {
            'Accuracy': accuracy_cls,
            'Precisión': precision_cls,
            'Recall': recall_cls,
            'F1-Score': f1_cls
        }
    return metrics

metrics_per_class = compute_metrics_per_class(y_test.values, y_pred, selected_categories)

# Visualizar métricas por clase
metrics_df = pd.DataFrame(metrics_per_class).T
plt.figure(figsize=(12, 8))
sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.4f')
plt.title('Métricas por Clase')
plt.tight_layout()
plt.savefig('metrics_per_class.png')
plt.close()

# 14. Calcular la curva ROC y analizarla
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Binarizar las etiquetas
y_test_binarized = label_binarize(y_test, classes=selected_categories)
n_classes = y_test_binarized.shape[1]

# Obtener las probabilidades para cada clase
def get_probabilities_batch(classifier, X):
    probs = []
    for tokens in X:
        probabilities = classifier.get_probabilities(tokens)
        probs.append([probabilities.get(cls, 0) for cls in selected_categories])
    return np.array(probs)

y_scores = get_probabilities_batch(nb_classifier, X_test)
roc_auc = roc_auc_score(y_test_binarized, y_scores, average='weighted', multi_class='ovr')

# Calcular ROC para cada clase
fpr = dict()
tpr = dict()
roc_auc_per_class = dict()
for i, cls in enumerate(selected_categories):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
    roc_auc_per_class[cls] = auc(fpr[i], tpr[i])

# Plotear la curva ROC
plt.figure(figsize=(10, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_categories)))
for i, (cls, color) in enumerate(zip(selected_categories, colors)):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve de {cls} (AUC = {roc_auc_per_class[cls]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Clase')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

print("Se han generado las siguientes imágenes:")
print("1. confusion_matrix.png - Matriz de Confusión")
print("2. general_metrics.png - Métricas de Evaluación General")
print("3. metrics_per_class.png - Métricas por Clase")
print("4. roc_curves.png - Curvas ROC por Clase")
print(f"\nÁrea Bajo la Curva (AUC) Promedio: {roc_auc:.4f}")
