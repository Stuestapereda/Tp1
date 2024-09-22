import pandas as pd

# Paso 1 y 2: Carga de datos y preprocesamiento
data = pd.read_csv("binary.csv")

# Discretización de GRE y GPA
data['GRE_discretized'] = data['gre'].apply(lambda x: 'GRE_>=500' if x >= 500 else 'GRE_<500')
data['GPA_discretized'] = data['gpa'].apply(lambda x: 'GPA_>=3' if x >= 3 else 'GPA_<3')

# Paso 3: Definición de la estructura de la red (implícita en los cálculos posteriores)

# Paso 4: Cálculo de probabilidades 
def calcular_probabilidad_marginal(data, variable):
  """Calcula la probabilidad marginal de una variable."""
  conteo = data[variable].value_counts()
  total = len(data)
  probabilidades = conteo / total
  return probabilidades

def calcular_probabilidad_condicional(data, variable, padres):
  """Calcula la probabilidad condicional P(variable | padres)."""
  probabilidades = {}
  for valor_variable in data[variable].unique():
    for combinacion_padres in data[padres].drop_duplicates().values:
      filtro = data[padres] == list(combinacion_padres)
      filtro = filtro.all(axis=1)
      conteo_total = len(data[filtro])
      conteo_variable = len(data[(data[variable] == valor_variable) & filtro])
      
      if conteo_total > 0:
        probabilidad = conteo_variable / conteo_total
      else:
        probabilidad = 0  # Manejar casos con conteo cero

      probabilidades[
          (valor_variable, tuple(combinacion_padres))
      ] = probabilidad
  return probabilidades

# Probabilidades marginales
P_rank = calcular_probabilidad_marginal(data, 'rank')
P_GRE = calcular_probabilidad_marginal(data, 'GRE_discretized')
P_GPA = calcular_probabilidad_marginal(data, 'GPA_discretized')

# Probabilidad condicional de admisión
P_admit = calcular_probabilidad_condicional(
    data, 'admit', ['rank', 'GRE_discretized', 'GPA_discretized']
)
# Paso 5: Cálculo de probabilidades específicas
def calcular_probabilidad_admit(rank, gre=None, gpa=None):
  """Calcula P(admit | rank, gre, gpa) usando las CPTs."""
  if gre is not None:
    gre_disc = 'GRE_>=500' if gre >= 500 else 'GRE_<500'
  if gpa is not None:
    gpa_disc = 'GPA_>=3' if gpa >= 3 else 'GPA_<3'

  if gre is None and gpa is None:
    prob_admit_0 = sum(
        P_admit[(0, (rank, gre_cat, gpa_cat))] * P_GRE[gre_cat] * P_GPA[gpa_cat]
        for gre_cat in P_GRE.index
        for gpa_cat in P_GPA.index
    )
    prob_admit_1 = sum(
        P_admit[(1, (rank, gre_cat, gpa_cat))] * P_GRE[gre_cat] * P_GPA[gpa_cat]
        for gre_cat in P_GRE.index
        for gpa_cat in P_GPA.index
    )
  elif gre is not None and gpa is not None:
    prob_admit_0 = P_admit[(0, (rank, gre_disc, gpa_disc))]
    prob_admit_1 = P_admit[(1, (rank, gre_disc, gpa_disc))]
  
  return {0: prob_admit_0, 1: prob_admit_1}

# Ejemplos de cálculo
prob_admit_0_rank1 = calcular_probabilidad_admit(rank=1)[0]
prob_admit_1_rank2_gre450_gpa35 = calcular_probabilidad_admit(rank=2, gre=450, gpa=3.5)[1]

print(f"P(admit=0 | rank=1): {prob_admit_0_rank1}")
print(f"P(admit=1 | rank=2, GRE=450, GPA=3.5): {prob_admit_1_rank2_gre450_gpa35}")

# Paso 6: Validación (se deja como ejercicio al lector)

# Paso 7: Interpretación y ajuste (se deja como ejercicio al lector)