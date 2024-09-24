import pandas as pd
from collections import defaultdict

# Suponiendo que df es el DataFrame utilizado en ambos archivos
df = pd.read_csv('binary.csv')

# Función de red_bayesiana.py
def calcular_frecuencias_red_bayesiana(df):
    admit_rgg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    laplace = 1  # Suponiendo un valor de suavizado de Laplace

    A = [0, 1]
    GRE = ["high", 'low']
    GPA = ["high", 'low']
    rank = [e for e in range(1, 5)]

    for a in A:
        for gre in GRE:
            for gpa in GPA:
                for r in rank:
                    admit_rgg[a][gre][gpa][r] = laplace

    for f, row in df.iterrows():
        admit2 = row['admit']
        gpa2 = row['gpa']
        gre2 = row['gre']
        rank2 = row['rank']
        admit_rgg[admit2][gre2][gpa2][rank2] += 1

    return admit_rgg

# Función de CorreccionEJ3.py
def calcular_frecuencias_correccion_ej3(data, variable, padres):
    frecuencias = defaultdict(int)
    for valor_variable in data[variable].unique():
        for combinacion_padres in data[padres].drop_duplicates().values:
            filtro = data[padres] == list(combinacion_padres)
            filtro = filtro.all(axis=1)
            conteo_variable = len(data[(data[variable] == valor_variable) & filtro])
            frecuencias[(valor_variable, tuple(combinacion_padres))] = conteo_variable
    return frecuencias

# Comparación de frecuencias
frecuencias_rb = calcular_frecuencias_red_bayesiana(df)
frecuencias_cej3 = calcular_frecuencias_correccion_ej3(df, 'admit', ['gre', 'gpa', 'rank'])

# Convertir frecuencias_rb a un formato comparable
frecuencias_rb_convertidas = {}
for a in frecuencias_rb:
    for gre in frecuencias_rb[a]:
        for gpa in frecuencias_rb[a][gre]:
            for r in frecuencias_rb[a][gre][gpa]:
                clave = (a, (gre, gpa, r))
                frecuencias_rb_convertidas[clave] = frecuencias_rb[a][gre][gpa][r]

# Comparar frecuencias
diferencias_frecuencias = []
for key in frecuencias_rb_convertidas:
    if key in frecuencias_cej3:
        if frecuencias_rb_convertidas[key] != frecuencias_cej3[key]:
            diferencias_frecuencias.append((key, frecuencias_rb_convertidas[key], frecuencias_cej3[key]))
    else:
        diferencias_frecuencias.append((key, frecuencias_rb_convertidas[key], 'No encontrado en CEJ3'))

for key in frecuencias_cej3:
    if key not in frecuencias_rb_convertidas:
        diferencias_frecuencias.append((key, 'No encontrado en RB', frecuencias_cej3[key]))

# Imprimir diferencias de frecuencias
if diferencias_frecuencias:
    print("Diferencias en frecuencias encontradas:")
    for dif in diferencias_frecuencias:
        print(f"Clave: {dif[0]}, Red Bayesiana: {dif[1]}, CorreccionEJ3: {dif[2]}")
else:
    print("No se encontraron diferencias en frecuencias.")

# Calcular probabilidades condicionales
def calcular_probabilidades(frecuencias, total_frecuencias):
    probabilidades = {}
    for key in frecuencias:
        total = total_frecuencias[key[1]]
        if total > 0:
            probabilidades[key] = frecuencias[key] / total
        else:
            probabilidades[key] = 0
    return probabilidades

# Calcular total de frecuencias para cada combinación de padres
total_frecuencias_rb = defaultdict(int)
total_frecuencias_cej3 = defaultdict(int)
for key in frecuencias_rb_convertidas:
    total_frecuencias_rb[key[1]] += frecuencias_rb_convertidas[key]
for key in frecuencias_cej3:
    total_frecuencias_cej3[key[1]] += frecuencias_cej3[key]

# Calcular probabilidades
probabilidades_rb = calcular_probabilidades(frecuencias_rb_convertidas, total_frecuencias_rb)
probabilidades_cej3 = calcular_probabilidades(frecuencias_cej3, total_frecuencias_cej3)

# Comparar probabilidades
diferencias_probabilidades = []
for key in probabilidades_rb:
    if key in probabilidades_cej3:
        if probabilidades_rb[key] != probabilidades_cej3[key]:
            diferencias_probabilidades.append((key, probabilidades_rb[key], probabilidades_cej3[key]))
    else:
        diferencias_probabilidades.append((key, probabilidades_rb[key], 'No encontrado en CEJ3'))

for key in probabilidades_cej3:
    if key not in probabilidades_rb:
        diferencias_probabilidades.append((key, 'No encontrado en RB', probabilidades_cej3[key]))

# Imprimir diferencias de probabilidades
if diferencias_probabilidades:
    print("Diferencias en probabilidades encontradas:")
    for dif in diferencias_probabilidades:
        print(f"Clave: {dif[0]}, Red Bayesiana: {dif[1]}, CorreccionEJ3: {dif[2]}")
else:
    print("No se encontraron diferencias en probabilidades.")