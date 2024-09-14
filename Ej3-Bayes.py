import pandas as pd
from collections import defaultdict, Counter

# -------------------------------
# 1. Carga y Preprocesamiento de Datos
# -------------------------------

# Cargar el dataset
data = pd.read_csv('binary.csv')

# Discretizar las variables GRE y GPA
def discretizar_gre(gre):
    return 'high' if gre >= 500 else 'low'

def discretizar_gpa(gpa):
    return 'high' if gpa >= 3 else 'low'

data['GRE_discretized'] = data['gre'].apply(discretizar_gre)
data['GPA_discretized'] = data['gpa'].apply(discretizar_gpa)

# Eliminar filas con valores nulos en las columnas relevantes
data = data.dropna(subset=['rank', 'GRE_discretized', 'GPA_discretized', 'admit'])

# -------------------------------
# 2. Cálculo de Probabilidades Condicionales y Marginales
# -------------------------------

# Crear contadores para las probabilidades
# Contador para P(admit | rank)
admit_given_rank = defaultdict(Counter)
# Contador para P(GPA | rank)
gpa_given_rank = defaultdict(Counter)
# Contador para P(GRE | rank)
gre_given_rank = defaultdict(Counter)
# Contador para P(rank)
rank_counts = Counter(data['rank'])
# Contador total
total_count = len(data)

# Rellenar los contadores
for _, row in data.iterrows():
    rank = row['rank']
    admit = row['admit']
    gpa = row['GPA_discretized']
    gre = row['GRE_discretized']
    
    admit_given_rank[rank][admit] += 1
    gpa_given_rank[rank][gpa] += 1
    gre_given_rank[rank][gre] += 1

# Función para calcular probabilidades condicionales con suavizado de Laplace
def calcular_probabilidad(contador, categoria, valor, laplace=1):
    """
    Calcula la probabilidad condicional P(valor | categoria) con suavizado de Laplace.

    Args:
        contador (defaultdict): Contador de frecuencias para la categoría.
        categoria (int): Valor de la categoría (rank).
        valor (str o int): Valor de la variable condicionante (GPA, GRE, admit).
        laplace (int, optional): Parámetro de suavizado. Default es 1.

    Returns:
        float: Probabilidad condicional calculada.
    """
    return (contador[categoria][valor] + laplace) / (contador[categoria].total() + laplace * len(contador[categoria]))

# Calcular probabilidades marginales P(rank)
p_rank = {rank: count / total_count for rank, count in rank_counts.items()}

# Calcular P(admit | rank), P(GPA | rank), P(GRE | rank)
p_admit_given_rank = defaultdict(dict)
p_gpa_given_rank = defaultdict(dict)
p_gre_given_rank = defaultdict(dict)

for rank in rank_counts:
    # P(admit=1 | rank)
    p_admit_given_rank[rank][1] = calcular_probabilidad(admit_given_rank, rank, 1)
    # P(admit=0 | rank)
    p_admit_given_rank[rank][0] = calcular_probabilidad(admit_given_rank, rank, 0)
    
    # P(GPA='high' | rank)
    p_gpa_given_rank[rank]['high'] = calcular_probabilidad(gpa_given_rank, rank, 'high')
    # P(GPA='low' | rank)
    p_gpa_given_rank[rank]['low'] = calcular_probabilidad(gpa_given_rank, rank, 'low')
    
    # P(GRE='high' | rank)
    p_gre_given_rank[rank]['high'] = calcular_probabilidad(gre_given_rank, rank, 'high')
    # P(GRE='low' | rank)
    p_gre_given_rank[rank]['low'] = calcular_probabilidad(gre_given_rank, rank, 'low')

# -------------------------------
# 3. Cálculo de P(admit | rank, GPA, GRE)
# -------------------------------

def calcular_probabilidad_admit(rank, gpa, gre, admit_value):
    """
    Calcula la probabilidad no normalizada P(admit=admit_value | rank, GPA, GRE).

    Args:
        rank (int): Rango de la escuela secundaria.
        gpa (str): Categoría de GPA ('high' o 'low').
        gre (str): Categoría de GRE ('high' o 'low').
        admit_value (int): Valor de admisión (0 o 1).

    Returns:
        float: Probabilidad no normalizada.
    """
    p_admit = p_admit_given_rank[rank][admit_value]
    p_gpa = p_gpa_given_rank[rank][gpa]
    p_gre = p_gre_given_rank[rank][gre]
    return p_admit * p_gpa * p_gre

def calcular_probabilidad_admit_normalizada(rank, gpa, gre):
    """
    Calcula la probabilidad normalizada P(admit | rank, GPA, GRE).

    Args:
        rank (int): Rango de la escuela secundaria.
        gpa (str): Categoría de GPA ('high' o 'low').
        gre (str): Categoría de GRE ('high' o 'low').

    Returns:
        dict: Diccionario con las probabilidades normalizadas para admit=1 y admit=0.
    """
    p_admit_1 = calcular_probabilidad_admit(rank, gpa, gre, 1)
    p_admit_0 = calcular_probabilidad_admit(rank, gpa, gre, 0)
    total = p_admit_1 + p_admit_0
    if total == 0:
        return {'admit_1': 0, 'admit_0': 0}
    return {'admit_1': p_admit_1 / total, 'admit_0': p_admit_0 / total}

# -------------------------------
# 4. Cálculo de las Probabilidades Solicitadas
# -------------------------------

# a) Probabilidad de que una persona que proviene de una escuela con rank=1 y tiene GPA='high' y GRE='low' no haya sido admitida
prob_no_admit_rank_1 = calcular_probabilidad_admit_normalizada(rank=1, gpa='high', gre='low')['admit_0']
print(f"Probabilidad de que una persona con rank=1, GPA='high' y GRE='low' no haya sido admitida: {prob_no_admit_rank_1:.4f}")

# b) Probabilidad de que una persona que fue a una escuela de rank=2, tiene GRE=450 (low) y GPA=3.5 (high) sea admitida
# Discretización ya aplicada: GRE=450 -> 'low', GPA=3.5 -> 'high'
prob_admit_rank_2 = calcular_probabilidad_admit_normalizada(rank=2, gpa='high', gre='low')['admit_1']
print(f"Probabilidad de que una persona con rank=2, GRE='low' y GPA='high' sea admitida: {prob_admit_rank_2:.4f}")

# -------------------------------
# 5. Funciones Adicionales (Opcional)
# -------------------------------

def calcular_probabilidad_general_admit(admit_value):
    """
    Calcula la probabilidad general de admisión P(admit=admit_value).

    Args:
        admit_value (int): Valor de admisión (0 o 1).

    Returns:
        float: Probabilidad general de admisión.
    """
    admit_counts = admit_given_rank[1] + admit_given_rank[2] + admit_given_rank[3] + admit_given_rank[4]
    total_admit = admit_counts[admit_value]
    return total_admit / total_count

# Ejemplo de uso de la función adicional
p_admit_1_general = calcular_probabilidad_general_admit(1)
p_admit_0_general = calcular_probabilidad_general_admit(0)
print(f"P(admit=1) general: {p_admit_1_general:.4f}")
print(f"P(admit=0) general: {p_admit_0_general:.4f}")
