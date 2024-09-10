import csv

# Definición de nombres de atributos y clases
ATRIBUTOS = ['scones', 'cerveza', 'whisky', 'avena', 'futbol']
CLASES = {'I': 'Inglesa', 'E': 'Escocesa'}

# Función para cargar los datos desde un archivo CSV
def cargar_datos(nombre_archivo):
    datos = []  # Lista para almacenar los datos cargados
    with open(nombre_archivo, 'r') as archivo:  # Abrir el archivo en modo lectura
        lector = csv.reader(archivo, delimiter=';')  # Crear un lector CSV con el delimitador ';'
        next(lector)  # Saltar la fila de encabezados
        for fila in lector:
            # Convertir cada valor a entero excepto el último (que es la clase) y añadir a la lista de datos
            datos.append([int(valor) for valor in fila[:-1]] + [fila[-1]])
    return datos

# Función para calcular las probabilidades a priori y las probabilidades condicionales
def calcular_probabilidades(datos):
    total_ejemplos = len(datos)  # Total de ejemplos en los datos
    conteo_clases = {'I': 0, 'E': 0}  # Contador para cada clase
    conteo_atributos = {('I', i, v): 0 for i in range(len(ATRIBUTOS)) for v in [0, 1]}  # Contador para atributos en clase I
    conteo_atributos.update({('E', i, v): 0 for i in range(len(ATRIBUTOS)) for v in [0, 1]})  # Contador para atributos en clase E

    # Contar ejemplos por clase y atributos
    for ejemplo in datos:
        clase = ejemplo[-1]  # Obtener la clase del ejemplo (último valor de la lista)
        conteo_clases[clase] += 1  # Incrementar el conteo de la clase
        for i, valor in enumerate(ejemplo[:-1]):
            conteo_atributos[(clase, i, valor)] += 1  # Incrementar el conteo del atributo en la clase correspondiente

    # Calcular probabilidades
    prob_clases = {c: conteo_clases[c] / total_ejemplos for c in conteo_clases}  # Probabilidad a priori de cada clase
    probabilidades = {(c, i, v): (conteo_atributos[(c, i, v)] + 1) / (conteo_clases[c] + 2) for c in conteo_clases for i in range(len(ATRIBUTOS)) for v in [0, 1]}

    return prob_clases, probabilidades

# Función para clasificar un nuevo ejemplo con normalización de probabilidades
def clasificar_con_normalizacion(ejemplo, prob_clases, probabilidades):
    probabilidades_conjuntas = {}  # Para almacenar las probabilidades conjuntas de cada clase
    probabilidad_evidencia = 0  # Para almacenar la suma de las probabilidades conjuntas (P(evidencia))

    # Calcular la probabilidad conjunta para cada clase dada el ejemplo
    for clase in prob_clases:
        probabilidad_conjunta = prob_clases[clase]  # Iniciar con la probabilidad a priori de la clase
        for i, valor in enumerate(ejemplo):
            probabilidad_conjunta *= probabilidades[(clase, i, valor)]  # Multiplicar por la probabilidad condicional
        probabilidades_conjuntas[clase] = probabilidad_conjunta  # Guardar la probabilidad conjunta
        probabilidad_evidencia += probabilidad_conjunta  # Sumar para la probabilidad de la evidencia

    # Normalizar las probabilidades conjuntas para que sumen 1 (probabilidad posterior)
    probabilidades_posteriores = {clase: prob_conjunta / probabilidad_evidencia for clase, prob_conjunta in probabilidades_conjuntas.items()}

    # Encontrar la clase con la mayor probabilidad posterior
    mejor_clase = max(probabilidades_posteriores, key=probabilidades_posteriores.get)
    mejor_probabilidad = probabilidades_posteriores[mejor_clase]

    return mejor_clase, mejor_probabilidad, probabilidades_posteriores

# Función para mostrar un cálculo detallado de la clasificación con normalización
def mostrar_calculo_detallado_con_normalizacion(ejemplo, prob_clases, probabilidades):
    print("## Cálculo detallado para el ejemplo:")
    for i, valor in enumerate(ejemplo):
        print(f"  {ATRIBUTOS[i]}: {'Sí' if valor == 1 else 'No'}")

    probabilidades_conjuntas = {}
    probabilidad_evidencia = 0

    for clase, nombre_clase in CLASES.items():
        probabilidad_conjunta = prob_clases[clase]
        print(f"\nPara la nacionalidad {nombre_clase}:")
        print(f"P({nombre_clase}) = {probabilidad_conjunta:.6f}")

        for i, valor in enumerate(ejemplo):
            p = probabilidades[(clase, i, valor)]
            print(f"P({ATRIBUTOS[i]}={'Sí' if valor == 1 else 'No'} | {nombre_clase}) = {p:.6f}")
            probabilidad_conjunta *= p

        probabilidades_conjuntas[clase] = probabilidad_conjunta
        probabilidad_evidencia += probabilidad_conjunta
        print(f"Probabilidad conjunta para {nombre_clase}: {probabilidad_conjunta:.6f}")

    print(f"\nProbabilidad de la evidencia (P(evidencia)): {probabilidad_evidencia:.6f}")

    probabilidades_posteriores = {clase: prob_conjunta / probabilidad_evidencia for clase, prob_conjunta in probabilidades_conjuntas.items()}

    for clase, prob_posterior in probabilidades_posteriores.items():
        print(f"Probabilidad posterior para {CLASES[clase]}: {prob_posterior:.6f}")

# Cargar los datos desde el archivo CSV
datos = cargar_datos('PreferenciasBritanicos.csv')

# Calcular las probabilidades a priori y las probabilidades condicionales
prob_clases, probabilidades = calcular_probabilidades(datos)

# Definir ejemplos a clasificar
x1 = [1, 0, 1, 1, 0]  # Ejemplo 1
x2 = [0, 1, 1, 0, 1]  # Ejemplo 2

# Clasificar y mostrar detalles para el primer ejemplo
print("# Clasificación de ejemplos con normalización")
print("\n## Ejemplo x1:")
mostrar_calculo_detallado_con_normalizacion(x1, prob_clases, probabilidades)
clase_x1, prob_x1, probabilidades_x1 = clasificar_con_normalizacion(x1, prob_clases, probabilidades)
print(f"\nClasificación final para x1: {CLASES[clase_x1]} con probabilidad {prob_x1:.6f}")
print(f"Probabilidades finales: {probabilidades_x1}")

# Clasificar y mostrar detalles para el segundo ejemplo
print("\n## Ejemplo x2:")
mostrar_calculo_detallado_con_normalizacion(x2, prob_clases, probabilidades)
clase_x2, prob_x2, probabilidades_x2 = clasificar_con_normalizacion(x2, prob_clases, probabilidades)
print(f"\nClasificación final para x2: {CLASES[clase_x2]} con probabilidad {prob_x2:.6f}")
print(f"Probabilidades finales: {probabilidades_x2}")
