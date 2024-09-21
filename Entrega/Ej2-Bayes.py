import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar el archivo Excel
file_path = 'Noticias_argentinas.xlsx'
datos_noticias = pd.read_excel(file_path)

# Seleccionar las columnas relevantes: 'titular' y 'categoria'
datos_noticias_relevantes = datos_noticias[['titular', 'categoria']]

# Lista completa de categorías a incluir
categorias_seleccionadas = [
    'Internacional', 'Deportes', 'Salud', 
    'Ciencia y Tecnologia', 'Nacional', 'Economia'
]

# Filtrar solo las categorías seleccionadas
datos_filtrados = datos_noticias_relevantes[datos_noticias_relevantes['categoria'].isin(categorias_seleccionadas)]

# Eliminar filas con valores nulos en las columnas seleccionadas
datos_filtrados = datos_filtrados.dropna()

# Dividir los datos en conjunto de entrenamiento (70%) y de prueba (30%)
entrenamiento, prueba = train_test_split(datos_filtrados, test_size=0.3, random_state=42)

# Lista de palabras clave actualizadas para la clasificación
palabras_clave = ['política local',
 'legislatura',
 'gobernación',
 'ministro',
 'legislativo',
 'provincial',
 'municipal',
 'presupuesto',
 'reforma política',
 'partido político',
 'sindicatos',
 'huelga nacional',
 'tarifas',
 'ajuste fiscal',
 'deuda pública',
 'justicia argentina',
 'corte suprema',
 'ANSES',
 'AFIP',
 'diplomacia',
 'embajada',
 'tratado internacional',
 'conflicto global',
 'crisis internacional',
 'elecciones extranjeras',
 'G7',
 'ONU',
 'OTAN',
 'guerra',
 'Brexit',
 'cambio climático global',
 'sanciones',
 'comercio internacional',
 'migración',
 'embajador',
 'cumbre internacional',
 'epidemia',
 'pandemia',
 'COVID-19',
 'hospitalización',
 'emergencia sanitaria',
 'campaña de vacunación',
 'OMS',
 'enfermedad rara',
 'brote',
 'investigación médica',
 'tratamiento innovador',
 'medicamento',
 'farmacias',
 'salud mental',
 'nutrición',
 'obesidad',
 'sistema inmunológico',
 'enfermedad crónica',
 'enfermedades infecciosas',
 'deflación',
 'crecimiento económico',
 'inversión extranjera',
 'PIB',
 'balanza comercial',
 'mercados financieros',
 'bolsa de valores',
 'reservas internacionales',
 'banco central',
 'política monetaria',
 'deuda externa',
 'deuda interna',
 'fondos de inversión',
 'economía local',
 'microeconomía',
 'macroeconomía',
 'dólar',
 'gobierno',
 'millones',
 'banco',
 'economía',
 'mercado',
 'inflación',
 'crisis',
 'FMI',
 'deuda',
 'recesión',
 'tarifas',
 'ajuste',
 'subsidios',
 'tasa de interés',
 'tipo de cambio',
 'exportaciones',
 'importaciones',
 'PBI',
 'reservas',
 'fondo monetario',
 'cepo',
 'Lebacs',
 'riesgo país',
 'macri',
 'cristina',
 'fernández',
 'kirchner',
 'trump',
 'venezuela',
 'alberto',
 'elección',
 'justicia',
 'congreso',
 'senado',
 'diputados',
 'peronismo',
 'kirchnerismo',
 'cambiemos',
 'Frente de Todos',
 'campaña electoral',
 'voto',
 'reforma',
 'tribunal',
 'corrupción',
 'juicio',
 'denuncia',
 'causa',
 'fiscalía',
 'Juntos por el Cambio',
 'ministerio',
 'whatsapp',
 'tierra',
 'android',
 'usuarios',
 'celular',
 'iphone',
 'google',
 'facebook',
 'redes sociales',
 'ciberseguridad',
 'inteligencia artificial',
 'criptomonedas',
 'bitcoin',
 'fintech',
 'app',
 'actualización',
 'internet',
 'software',
 'datos',
 '5G',
 'fibra óptica',
 'startup',
 'innovación',
 'NASA',
 'asteroide',
 'tecnología',
 'ciencia',
 'espacio',
 'boca',
 'river',
 'final',
 'copa',
 'superliga',
 'fútbol',
 'mundial',
 'messi',
 'selección',
 'torneo',
 'partido',
 'gol',
 'liga',
 'equipo',
 'hinchas',
 'técnico',
 'jugador',
 'club',
 'estadio',
 'AFA',
 'sanción',
 'campeonato',
 'baloncesto',
 'tenis',
 'rugby',
 'olimpíadas',
 'medalla',
 'hantavirus',
 'riesgo',
 'muerte',
 'cáncer',
 'salud',
 'vacuna',
 'diabetes',
 'enfermedad',
 'sarampión',
 'prevención',
 'campaña',
 'hospitales',
 'ministerio de salud',
 'sanidad',
 'epidemia',
 'contagio',
 'virus',
 'medicina',
 'tratamiento',
 'pediatría',
 'consulta',
 'síntomas',
 'mortalidad',
 'sistema de salud',
 'donación',
 'órganos',
 'emergencia sanitaria',
 'china',
 'venezuela',
 'trump',
 'g20',
 'brexit',
 'macron',
 'bolsonaro',
 'estados unidos',
 'méxico',
 'brasil',
 'crisis humanitaria',
 'refugiados',
 'cumbre',
 'conflicto',
 'sanciones',
 'elecciones',
 'guerra comercial',
 'ONU',
 'acuerdo',
 'tratado',
 'diplomacia',
 'relaciones internacionales',
 'derechos humanos',
 'ayuda humanitaria',
 'inmigración',
 'intervención',
 'armas',
 'video',
 'campaña',
 'caso',
 'mundo',
 'incendio',
 'tormenta',
 'femicidio',
 'violencia',
 'accidente',
 'tránsito',
 'robo',
 'crimen',
 'investigación',
 'denuncia',
 'teatro',
 'cine',
 'música',
 'festival',
 'cultura',
 'espectáculo',
 'redes sociales',
 'influencer',
 'famosos',
 'celebridad',
 'evento',
 'moda',
 'desfile',
 'cumbre del g20',
 'boca-river',
 'final de libertadores',
 'elecciones 2019',
 'crisis económica',
 'FMI',
 'venezuela',
 'incendio',
 'emergencia',
 'inundación',
 'terremoto',
 'escándalo',
 'caso judicial',
 'asesinato',
 'tragedia',
 'manifestación',
 'huelga',
 'protesta',
 'reforma',
 'golpe',
 'renuncia',
 'ataque',
 'terrorismo',
 'operación policial',
 'rescate',
 'allanamiento']

# Calcular la probabilidad de aparición de cada categoría en el conjunto de entrenamiento
probabilidad_categoria = entrenamiento['categoria'].value_counts(normalize=True).to_dict()
print(probabilidad_categoria)
# Inicializar un diccionario para almacenar las probabilidades de aparición de las palabras relevantes por categoría
probabilidad_palabras_por_categoria = {categoria: {} for categoria in categorias_seleccionadas}

# Calcular las probabilidades a posteriori para cada palabra y categoría
for categoria in categorias_seleccionadas:
    titulos_categoria = entrenamiento[entrenamiento['categoria'] == categoria]['titular'].str.lower()
    total_palabras_categoria = sum(titulos_categoria.apply(lambda x: len(x.split())))
    
    for palabra in palabras_clave:
        frecuencia_palabra = titulos_categoria.apply(lambda x: palabra in x).sum()
        probabilidad_palabra = (frecuencia_palabra + 1) / (total_palabras_categoria + len(palabras_clave))
        probabilidad_palabras_por_categoria[categoria][palabra] = probabilidad_palabra

# Inicializar una lista para almacenar las predicciones
predicciones = []

# Iterar sobre cada título en el conjunto de prueba para hacer predicciones
for index, fila in prueba.iterrows():
    titulo = fila['titular'].lower()
    probabilidades_categoria = {}

    for categoria in probabilidad_categoria:
        probabilidad = np.log(probabilidad_categoria[categoria])
        
        for palabra in palabras_clave:
            if palabra in titulo:
                probabilidad += np.log(probabilidad_palabras_por_categoria[categoria][palabra])
            else:
                probabilidad += np.log(1 - probabilidad_palabras_por_categoria[categoria][palabra])
        
        probabilidades_categoria[categoria] = probabilidad
    
    prediccion = max(probabilidades_categoria, key=probabilidades_categoria.get)
    predicciones.append(prediccion)

# Agregar las predicciones al conjunto de prueba
prueba['prediccion'] = predicciones

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
plt.savefig('matriz_confusion_mejorada.png', bbox_inches='tight')

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
plt.savefig('metricas_por_categoria.png', bbox_inches='tight')

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
plt.savefig('metricas_evaluacion_global.png', bbox_inches='tight')

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
plt.savefig('curva_roc.png', bbox_inches='tight')
plt.close()
