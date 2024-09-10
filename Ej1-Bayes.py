import csv

# Definición de nombres de atributos y clases
# Los atributos representan las preferencias y las clases representan la nacionalidad
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

# Función para clasificar un nuevo ejemplo basado en las probabilidades calculadas
def clasificar(ejemplo, prob_clases, probabilidades):
    mejor_clase = None
    mejor_probabilidad = -1

    # Calcular la probabilidad para cada clase dada el ejemplo
    for clase in prob_clases:
        probabilidad = prob_clases[clase]  # Iniciar con la probabilidad a priori de la clase
        for i, valor in enumerate(ejemplo):
            probabilidad *= probabilidades[(clase, i, valor)]  # Multiplicar por la probabilidad condicional
        # Comparar con la mejor probabilidad encontrada hasta ahora
        if probabilidad > mejor_probabilidad:
            mejor_probabilidad = probabilidad
            mejor_clase = clase

    return mejor_clase, mejor_probabilidad

# Función para mostrar un cálculo detallado de la clasificación
def mostrar_calculo_detallado(ejemplo, prob_clases, probabilidades):
    print("## Cálculo detallado para el ejemplo:")
    for i, valor in enumerate(ejemplo):
        print(f"  {ATRIBUTOS[i]}: {'Sí' if valor == 1 else 'No'}")

    for clase, nombre_clase in CLASES.items():
        probabilidad = prob_clases[clase]
        print(f"\nPara la nacionalidad {nombre_clase}:")
        print(f"P({nombre_clase}) = {probabilidad:.6f}")

        for i, valor in enumerate(ejemplo):
            p = probabilidades[(clase, i, valor)]
            print(f"P({ATRIBUTOS[i]}={'Sí' if valor == 1 else 'No'} | {nombre_clase}) = {p:.6f}")
            probabilidad *= p

        print(f"Probabilidad final para {nombre_clase}: {probabilidad:.6f}")

# Cargar los datos desde el archivo CSV
datos = cargar_datos('PreferenciasBritanicos.csv')

# Calcular las probabilidades a priori y las probabilidades condicionales
prob_clases, probabilidades = calcular_probabilidades(datos)

# Definir ejemplos a clasificar
x1 = [1, 0, 1, 1, 0]  # Ejemplo 1
x2 = [0, 1, 1, 0, 1]  # Ejemplo 2

# Clasificar y mostrar detalles para el primer ejemplo
print("# Clasificación de ejemplos")
print("\n## Ejemplo x1:")
mostrar_calculo_detallado(x1, prob_clases, probabilidades)
clase_x1, prob_x1 = clasificar(x1, prob_clases, probabilidades)
print(f"\nClasificación final para x1: {CLASES[clase_x1]} con probabilidad {prob_x1:.6f}")

# Clasificar y mostrar detalles para el segundo ejemplo
print("\n## Ejemplo x2:")
mostrar_calculo_detallado(x2, prob_clases, probabilidades)
clase_x2, prob_x2 = clasificar(x2, prob_clases, probabilidades)
print(f"\nClasificación final para x2: {CLASES[clase_x2]} con probabilidad {prob_x2:.6f}")
