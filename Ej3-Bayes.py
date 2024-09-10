import csv

# Lectura de datos
data = []
with open('binary.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append({
            'admit': int(row['admit']),
            'gre': int(row['gre']),
            'gpa': float(row['gpa']),
            'rank': int(row['rank'])
        })

# Discretización de GRE y GPA
for row in data:
    row['gre'] = 'high' if row['gre'] >= 500 else 'low'
    row['gpa'] = 'high' if row['gpa'] >= 3 else 'low'

# Función para calcular la probabilidad condicional
def calcular_probabilidad(admit_val, rank_val, data):
    admit_count = sum(1 for row in data if row['admit'] == admit_val and row['rank'] == rank_val)
    rank_count = sum(1 for row in data if row['rank'] == rank_val)
    return admit_count / rank_count if rank_count > 0 else 0

# Cálculo de P(admit = 0 | rank = 1)
prob_no_admit_rank_1 = calcular_probabilidad(0, 1, data)
print(f"Probabilidad de que no sea admitido dado rank = 1: {prob_no_admit_rank_1:.4f}")

# Cálculo de P(admit = 1 | rank = 2, GRE < 500, GPA >= 3)
def calcular_probabilidad_condicional(admit_val, rank_val, gre_val, gpa_val, data):
    admit_count = sum(1 for row in data if row['admit'] == admit_val and row['rank'] == rank_val and row['gre'] == gre_val and row['gpa'] == gpa_val)
    total_count = sum(1 for row in data if row['rank'] == rank_val and row['gre'] == gre_val and row['gpa'] == gpa_val)
    return admit_count / total_count if total_count > 0 else 0

prob_admit_condicional = calcular_probabilidad_condicional(1, 2, 'low', 'high', data)
print(f"Probabilidad de ser admitido dado rank = 2, GRE < 500 y GPA >= 3: {prob_admit_condicional:.4f}")