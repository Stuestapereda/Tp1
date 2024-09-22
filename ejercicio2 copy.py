import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
nltk.download('punkt_tab')
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

#Preprocesamiento
data = "Noticias_argentinas.csv"

categorias_seleccionadas = [
    'Internacional', 'Deportes', 'Salud', 
    'Ciencia y Tecnologia', 'Nacional', 'Economia'
]

df = pd.read_csv(data)
df_filtrado=df[["titular","fuente","categoria"]]
df_filtrado=df_filtrado[df_filtrado["categoria"].isin(categorias_seleccionadas)]

# Eliminar filas con valores nulos en las columnas seleccionadas
df_filtrado = df_filtrado.dropna()

# Dividir los datos en conjunto de entrenamiento (70%) y de prueba (30%)
entrenamiento, prueba = train_test_split(df_filtrado, test_size=0.3, random_state=42)

#Análisis de palabras clave'

# Cargar las stopwords en español
stop_words = set(stopwords.words('spanish'))

palabra_clave={}
for i,categoria in enumerate(categorias_seleccionadas):
    df_cat_1=entrenamiento[entrenamiento["categoria"]==categoria]
    todos_los_titulares = " ".join(df_cat_1['titular'])

    # Tokenizar y filtrar palabras
    palabras = word_tokenize(todos_los_titulares.lower())
    palabras_filtradas = [palabra for palabra in palabras if palabra.isalpha() and palabra not in stop_words]
    

    # Contar las palabras más comunes
    contador_palabras = Counter(palabras_filtradas)

    # Obtener las 50 palabras más comunes
    palabra_clave[categoria] = dict(contador_palabras.most_common(50))

    

#Calculo de probabilidades
#Aplicar suavisado laplasiano
todas_palabras={}
freq=0 #total de palabras
for categoria in palabra_clave:
    for palabra in palabra_clave[categoria]:
        if palabra not in todas_palabras:
            todas_palabras[palabra]=palabra_clave[categoria][palabra]
        else:
            todas_palabras[palabra]+=palabra_clave[categoria][palabra]
        freq+=palabra_clave[categoria][palabra]
    
for word in todas_palabras:
    for categoria in palabra_clave:
        if word in palabra_clave[categoria]:
            palabra_clave[categoria][word]+=1
        else:
            palabra_clave[categoria][word]=1 #Para evitar la influencia de palabras no observadas

priori={}
for i,categoria in enumerate(categorias_seleccionadas):
    df_cat_1=entrenamiento[entrenamiento["categoria"]==categoria]
    priori[categoria]=len(df_cat_1)/len(entrenamiento)

"""
evidencia={}
freq=0
for categoria in palabra_clave:
    for palabra in palabra_clave[categoria]:
        if palabra not in evidencia:
            evidencia[palabra]=palabra_clave[categoria][palabra]
        else:
            evidencia[palabra]+=palabra_clave[categoria][palabra]
        freq+=palabra_clave[categoria][palabra]

for palabra in evidencia:
    evidencia[palabra]=evidencia[palabra]/freq
"""

verosimilitud={}
for categoria in palabra_clave:
    total_palabras_categoria = sum(palabra_clave[categoria].values()) 
    cat={}
    for palabra in palabra_clave[categoria]:
        cat[palabra]=palabra_clave[categoria][palabra]/total_palabras_categoria
    verosimilitud[categoria]=cat


#Comprobando
probabilidades = []
predicion=[]
for index, fila in prueba.iterrows():
    titulo = fila['titular'].lower()
    titulo_filtrado = titulo.split()
    palabras_filtradas = [palabra for palabra in titulo_filtrado if palabra not in stop_words and len(palabra) > 1]

    prob_cate={}
    for categoria in palabra_clave:
        prob=np.log(priori[categoria])

        for palabra in palabra_clave[categoria]:
            if palabra in palabras_filtradas:
                prob+=np.log(verosimilitud[categoria][palabra])
            else:
                continue #Ignoramos las palabras que no estan.
                #prob+=np.log(1-verosimilitud[categoria][palabra])

        prob_cate[categoria]=prob

    predic = max(prob_cate, key=prob_cate.get)
    predicion.append(predic)
    probabilidades.append([prob_cate[cat] for cat in categorias_seleccionadas])

probabilidades = np.array(probabilidades)

# Agregar las predicciones al conjunto de prueba
prueba['prediccion'] = predicion

"""
Binarización en un problema multiclase
En un problema de clasificación multiclase, 
para calcular la curva ROC para cada clase, 
debes tratar cada categoría como un problema binario: "¿Esta muestra pertenece a esta categoría o no?".

"""
binarizado = label_binarize(prueba['categoria'], classes=categorias_seleccionadas)

# Crear la curva ROC para cada categoría
fpr = {} #false positive    
tpr = {} #true positive
roc_auc = {} #area under the curve

plt.figure()
for i, categoria in enumerate(categorias_seleccionadas):
    # Calcular FPR, TPR y AUC para cada categoría
    fpr[categoria], tpr[categoria], thresholds = roc_curve(binarizado[:, i], probabilidades[:, i])
    roc_auc[categoria] = auc(fpr[categoria], tpr[categoria])

    # Graficar la curva ROC
    plt.scatter(fpr[categoria], tpr[categoria], label=f'ROC {categoria} (AUC = {roc_auc[categoria]:.2f})')
    print(f"Para la categoría {categoria} se usaron {len(thresholds)} umbrales")
plt.show()

"""
# Crear la matriz de confusión
etiquetas = categorias_seleccionadas
matriz_confusion = confusion_matrix(prueba['categoria'], prueba['prediccion'], labels=etiquetas)

# Mejorar la visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
plt.imshow(matriz_confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(len(etiquetas))
plt.xticks(tick_marks, etiquetas, rotation=45, ha="right")
plt.yticks(tick_marks, etiquetas)

# Anotar la matriz de confusión
fmt = 'd'
thresh = matriz_confusion.max() / 2.
for i, j in np.ndindex(matriz_confusion.shape):
    plt.text(j, i, format(matriz_confusion[i, j], fmt),
             ha="center", va="center",
             color="white" if matriz_confusion[i, j] > thresh else "black")

plt.ylabel('Categoría Real')
plt.xlabel('Categoría Predicha')
plt.tight_layout()
plt.savefig('matriz_confusion_mejorada2.png', bbox_inches='tight')

# Calcular las métricas de evaluación por categoría
metricas_por_categoria = {}

for i, etiqueta in enumerate(etiquetas):
    tp = matriz_confusion[i, i]
    fp = matriz_confusion[:, i].sum() - tp
    fn = matriz_confusion[i, :].sum() - tp
    tn = matriz_confusion.sum() - (tp + fp + fn)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metricas_por_categoria[etiqueta] = {
        'Precisión': precision,
        'Recall': recall,
        'F1-score': f1
    }

# Crear un DataFrame para mostrar las métricas por categoría
df_metricas_por_categoria = pd.DataFrame(metricas_por_categoria).T

# Mejorar la visualización de las métricas por categoría
plt.figure(figsize=(10, 6))
plt.table(cellText=df_metricas_por_categoria.values.round(2), 
          colLabels=df_metricas_por_categoria.columns, 
          rowLabels=df_metricas_por_categoria.index, 
          cellLoc='center', 
          rowColours=["#f2f2f2"] * len(df_metricas_por_categoria), 
          colColours=["#f2f2f2"] * len(df_metricas_por_categoria.columns),
          loc='center', 
          colWidths=[0.2] * len(df_metricas_por_categoria.columns))
plt.axis('off')
plt.title('Métricas de Evaluación por Categoría', fontsize=16)
plt.savefig('metricas_por_categoria2.png', bbox_inches='tight')

# Calcular las métricas de evaluación globales
accuracy = accuracy_score(prueba['categoria'], prueba['prediccion'])
precision_global = precision_score(prueba['categoria'], prueba['prediccion'], average='weighted', zero_division=0)
recall_global = recall_score(prueba['categoria'], prueba['prediccion'], average='weighted', zero_division=0)
f1_global = f1_score(prueba['categoria'], prueba['prediccion'], average='weighted', zero_division=0)

# Crear una figura para las métricas de evaluación globales y guardarlas como imagen
plt.figure(figsize=(6, 4))
plt.text(0.1, 0.8, f'Accuracy: {accuracy:.2f}', fontsize=14)
plt.text(0.1, 0.6, f'Precisión: {precision_global:.2f}', fontsize=14)
plt.text(0.1, 0.4, f'Recall: {recall_global:.2f}', fontsize=14)
plt.text(0.1, 0.2, f'F1-score: {f1_global:.2f}', fontsize=14)
plt.axis('off')
plt.title('Métricas de Evaluación Globales', fontsize=16)
plt.savefig('metricas_evaluacion_global2.png', bbox_inches='tight')

# Binarizar las etiquetas de las categorías para el cálculo de la curva ROC
categorias_binarizadas = label_binarize(prueba['categoria'], classes=etiquetas)
predicciones_binarizadas = label_binarize(prueba['prediccion'], classes=etiquetas)

# Calcular la curva ROC y el AUC para cada categoría
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(etiquetas)):
    fpr[i], tpr[i], _ = roc_curve(categorias_binarizadas[:, i], predicciones_binarizadas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar la curva ROC para cada categoría y guardarla como imagen
plt.figure(figsize=(10, 8))
for i, label in enumerate(etiquetas):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC de {label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
plt.title('Curva ROC para cada categoría', fontsize=16)
plt.legend(loc='lower right', fontsize=10)
plt.savefig('curva_roc2.png', bbox_inches='tight')
plt.close()
"""