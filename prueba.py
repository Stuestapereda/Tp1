import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

# Cargar datos
data = "Noticias_argentinas.csv"
categorias_seleccionadas = ['Internacional', 'Deportes', 'Salud', 'Ciencia y Tecnologia', 'Nacional', 'Economia']

df = pd.read_csv(data)
df_filtrado = df[["titular", "fuente", "categoria"]]
df_filtrado = df_filtrado[df_filtrado["categoria"].isin(categorias_seleccionadas)]
df_filtrado = df_filtrado.dropna()

# Dividir datos en entrenamiento y prueba
entrenamiento, prueba = train_test_split(df_filtrado, test_size=0.3, random_state=42)

# Crear un modelo Naive Bayes con vectorización
modelo = make_pipeline(CountVectorizer(stop_words='spanish'), MultinomialNB())

# Entrenar el modelo
X_train = entrenamiento['titular']
y_train = entrenamiento['categoria']
modelo.fit(X_train, y_train)

# Predecir en los datos de prueba
X_test = prueba['titular']
y_test = prueba['categoria']
predicciones = modelo.predict(X_test)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, predicciones)
precision = precision_score(y_test, predicciones, average='macro')
recall = recall_score(y_test, predicciones, average='macro')
f1 = f1_score(y_test, predicciones, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
