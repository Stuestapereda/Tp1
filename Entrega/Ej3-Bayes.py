import pandas as pd

# Cargar el dataset
data = pd.read_csv('binary.csv')

# Discretizar las variables
def discretizar_gre(gre):
    return 'high' if gre >= 500 else 'low'

def discretizar_gpa(gpa):
    return 'high' if gpa >= 3 else 'low'

data['GRE'] = data['gre'].apply(discretizar_gre)
data['GPA'] = data['gpa'].apply(discretizar_gpa)

# Calcular las tablas de probabilidad condicionales
def calcular_probabilidad_condicional_admit(rank, gpa, gre, admit_value, data):
    total = len(data[(data['rank'] == rank) & (data['GPA'] == gpa) & (data['GRE'] == gre)])
    admit_count = len(data[(data['rank'] == rank) & (data['GPA'] == gpa) & (data['GRE'] == gre) & (data['admit'] == admit_value)])
    return admit_count / total if total > 0 else 0

def calcular_probabilidad_gpa_given_rank(rank, gpa, data):
    total = len(data[data['rank'] == rank])
    gpa_count = len(data[(data['rank'] == rank) & (data['GPA'] == gpa)])
    return gpa_count / total if total > 0 else 0

def calcular_probabilidad_gre_given_rank(rank, gre, data):
    total = len(data[data['rank'] == rank])
    gre_count = len(data[(data['rank'] == rank) & (data['GRE'] == gre)])
    return gre_count / total if total > 0 else 0

def calcular_probabilidad_rank(rank, data):
    total = len(data)
    rank_count = len(data[data['rank'] == rank])
    return rank_count / total if total > 0 else 0

# Aplicar el teorema de factorización
def calcular_probabilidad_conjunta(rank, gpa, gre, admit_value, data):
    p_admit_given_others = calcular_probabilidad_condicional_admit(rank, gpa, gre, admit_value, data)
    p_gpa_given_rank = calcular_probabilidad_gpa_given_rank(rank, gpa, data)
    p_gre_given_rank = calcular_probabilidad_gre_given_rank(rank, gre, data)
    p_rank = calcular_probabilidad_rank(rank, data)
    
    # Aplicamos la factorización: P(admit, rank, GPA, GRE) = P(admit | rank, GPA, GRE) * P(GPA | rank) * P(GRE | rank) * P(rank)
    return p_admit_given_others * p_gpa_given_rank * p_gre_given_rank * p_rank

 #Probabilidad de P(rank)
p_rank_1 = calcular_probabilidad_rank(1, data)
p_rank_2 = calcular_probabilidad_rank(2, data)
p_rank_3 = calcular_probabilidad_rank(3, data)
p_rank_4 = calcular_probabilidad_rank(4, data)

# Probabilidad de P(GPA | rank)
p_gpa_given_rank_1 = calcular_probabilidad_gpa_given_rank(1, 'high', data)
p_gpa_given_rank_2 = calcular_probabilidad_gpa_given_rank(2, 'high', data)
p_gpa_given_rank_3 = calcular_probabilidad_gpa_given_rank(3, 'high', data)
p_gpa_given_rank_4 = calcular_probabilidad_gpa_given_rank(4, 'high', data)

# Probabilidad de P(GPA, rank)
p_gpa_rank_1 = p_gpa_given_rank_1 * p_rank_1
p_gpa_rank_2 = p_gpa_given_rank_2 * p_rank_2
p_gpa_rank_3 = p_gpa_given_rank_3 * p_rank_3
p_gpa_rank_4 = p_gpa_given_rank_4 * p_rank_4

# Probabilidad de P(GRE | rank)
p_gre_given_rank_1 = calcular_probabilidad_gre_given_rank(1, 'high', data)
p_gre_given_rank_2 = calcular_probabilidad_gre_given_rank(2, 'high', data)
p_gre_given_rank_3 = calcular_probabilidad_gre_given_rank(3, 'high', data)
p_gre_given_rank_4 = calcular_probabilidad_gre_given_rank(4, 'high', data)

# Probabilidad de P(admit | rank, GPA, GRE)
p_admit_given_rank_gpa_gre_1 = calcular_probabilidad_condicional_admit(1, 'high', 'low', 1, data)
p_admit_given_rank_gpa_gre_2 = calcular_probabilidad_condicional_admit(2, 'high', 'low', 1, data)
p_admit_given_rank_gpa_gre_3 = calcular_probabilidad_condicional_admit(3, 'high', 'low', 1, data)
p_admit_given_rank_gpa_gre_4 = calcular_probabilidad_condicional_admit(4, 'high', 'low', 1, data)

# Probabilidad de P(admit, rank, GPA, GRE)
p_admit_rank_gpa_gre_1 = calcular_probabilidad_conjunta(1, 'high', 'low', 1, data)
p_admit_rank_gpa_gre_2 = calcular_probabilidad_conjunta(2, 'high', 'low', 1, data)
p_admit_rank_gpa_gre_3 = calcular_probabilidad_conjunta(3, 'high', 'low', 1, data)
p_admit_rank_gpa_gre_4 = calcular_probabilidad_conjunta(4, 'high', 'low', 1, data)

# Mostrar resultados
print(f"P(rank=1) = {p_rank_1}")
print(f"P(GPA | rank=1) = {p_gpa_given_rank_1}")
print(f"P(GPA, rank=1) = {p_gpa_rank_1}")
print(f"P(GRE | rank=1) = {p_gre_given_rank_1}")
print(f"P(admit | rank=1, GPA='high', GRE='low') = {p_admit_given_rank_gpa_gre_1}")
print(f"P(admit, rank=1, GPA='high', GRE='low') = {p_admit_rank_gpa_gre_1}")

# Repetir para rank=2, rank=3, rank=4

# Probabilidad solicitada: Una persona con rank 1 no ha sido admitida
prob_no_admit_rank_1 = calcular_probabilidad_conjunta(1, 'high', 'low', 0, data)
print(f"Probabilidad de que una persona con rank 1 no haya sido admitida: {prob_no_admit_rank_1}")

# Probabilidad solicitada: Una persona con rank 2, GRE=450, y GPA=3.5 sea admitida
prob_admit_rank_2_gre_gpa = calcular_probabilidad_conjunta(2, 'high', 'low', 1, data)
print(f"Probabilidad de que una persona con rank 2, GRE=450, y GPA=3.5 sea admitida: {prob_admit_rank_2_gre_gpa}")




