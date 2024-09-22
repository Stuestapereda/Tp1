import numpy as np
import pandas as pd
from collections import defaultdict, Counter
print()
"""
Calculo de probabilidad Aplicando el teorema de la factorización de la probabilidad conjunta
P(A,R,GRE,GPA) = P(A|R,GRE,GPA)*P(GRE|R)*P(GPA|R)*P(R)

Teorema de bayes para el calculo de probabilidad condicional
P(A|R,GRE,GPA) = P(R,GRE,GPA|A) * P(A) / P(R,GRE,GPA) 
"""

df = pd.read_csv("binary.csv")

df["gre"]=['high' if row >= 500 else 'low' for row in df["gre"]]
df["gpa"]=['high' if row >= 3 else 'low' for row in df["gpa"]]

laplace=0
"""
Vamos a armar
                               A = 0            |           A = 1
GRE=0 y GPA=0 y R=1    P(A=0|GRE=0,GPA=0,R=1)      P(A=1|GRE=0,GPA=0,R=1)
GRE=1 y GPA=0 y R=1    P(A=0|GRE=1GPA=0,R=1)       P(A=1|GRE=1,GPA=0,R=1)
GRE=1 y GPA=1 y R=1    P(A=0|GRE=1,GPA=1,R=1)      P(A=1|GRE=1,GPA=1,R=1)
GRE=0 y GPA=0 y R=2    P(A=0|GRE=0,GPA=0,R=2)      P(A=1|GRE=0,GPA=0,R=2)
GRE=1 y GPA=0 y R=2    P(A=0|GRE=1GPA=0,R=2)       P(A=1|GRE=1,GPA=0,R=2)
GRE=1 y GPA=1 y R=2    P(A=0|GRE=1,GPA=1,R=2)      P(A=1|GRE=1,GPA=1,R=2)
GRE=0 y GPA=0 y R=3    P(A=0|GRE=0,GPA=0,R=3)      P(A=1|GRE=0,GPA=0,R=3)
GRE=1 y GPA=0 y R=3    P(A=0|GRE=1GPA=0,R=3)       P(A=1|GRE=1,GPA=0,R=3)
GRE=1 y GPA=1 y R=3    P(A=0|GRE=1,GPA=1,R=3)      P(A=1|GRE=1,GPA=1,R=3)
GRE=0 y GPA=0 y R=4    P(A=0|GRE=0,GPA=0,R=4)      P(A=1|GRE=0,GPA=0,R=4)
GRE=1 y GPA=0 y R=4    P(A=0|GRE=1GPA=0,R=4)       P(A=1|GRE=1,GPA=0,R=4)
GRE=1 y GPA=1 y R=4    P(A=0|GRE=1,GPA=1,R=4)      P(A=1|GRE=1,GPA=1,R=4)
"""
admit_rgg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

#Suavizado landa
A=[0,1]
GRE=["high",'low']
GPA=["high",'low']
rank=[e for e in range(1,5)]

for a in A:
    for gre in GRE:
        for gpa in GPA:
            for r in rank:
                admit_rgg[a][gre][gpa][r] = laplace

#Calculo de frecuencias
for f, row in df.iterrows():
    admit2 = row['admit']
    gpa2 = row['gpa']
    gre2 = row['gre']
    rank2 = row['rank']

    admit_rgg[admit2][gre2][gpa2][rank2]+=1

#Calculo de probabilidades
print()
count=0
for r in rank:
    for gpa in GPA:
        for gre in GRE:
            total = sum(admit_rgg[a][gre][gpa][r] for a in A)     
            if total > 0:  
                for a in A:
                    admit_rgg[a][gre][gpa][r] = admit_rgg[a][gre][gpa][r] / total
                    print(f"P(A={a}|GRE={gre},GPA={gpa},rank={r}={admit_rgg[a][gre][gpa][r]})")
                    filtro = (df['gre'] == gre) & (df['gpa'] == gpa) & (df['rank'] == r) & (df['admit'] == a)
                    conteo = df[filtro].shape[0]
                    count+=conteo
                    filtro = (df['gre'] == gre) & (df['gpa'] == gpa) & (df['rank'] == r)
                    conteo2 = df[filtro].shape[0]
                    print(f"Hay {conteo} registros de {conteo2} que cumplen con los valores especificados")
print(f"total={count}")

                
"""
Vamos a armar
P(GRE|R)
                   GRE=0      |    GRE= 1
        R = 1   P(GRE=0|R=1)  | P(GRE=1|R=1)
        R = 2   P(GRE=0|R=2)  | P(GRE=1|R=2)
        R = 3   P(GRE=0|R=3)  | P(GRE=1|R=3)
        R = 4   P(GRE=0|R=4)  | P(GRE=1|R=4)
"""
gre_rank = defaultdict(lambda: defaultdict(int))
#Suavizado landa
for gre in GRE:
    for r in rank:
        gre_rank[gre][r] = laplace
#Calculo de frecuencias
for f, row in df.iterrows():
    admit2 = row['admit']
    gre2 = row['gre']
    gpa2 = row['gpa']
    rank2 = row['rank']

    gre_rank[gre2][rank2]+=1
#Calculo de probabilidades
print()
count=0
for r in rank:
    total = sum(gre_rank[gre][r] for gre in GRE)
    if total > 0:  
        for gre in GRE:
            count+=gre_rank[gre][r]
            gre_rank[gre][r] = gre_rank[gre][r] / total
            print(f"P(GRE={gre}|rank={r}={gre_rank[gre][r]})")
            filtro = (df['gre'] == gre) & (df['rank'] == r)
            conteo = df[filtro].shape[0]
            filtro = (df['rank'] == r)
            conteo2 = df[filtro].shape[0]
            print(f"Hay {conteo} registros de {conteo2} que cumplen con los valores especificados")
print(f"total={count}")

"""
Vamos a armar
P(GPA|R)
                   GPA=0      |    GPA = 1
        R = 1   P(GPA=0|R=1)  | P(GPA=1|R=1)
        R = 2   P(GPA=0|R=2)  | P(GPA=1|R=2)
        R = 3   P(GPA=0|R=3)  | P(GPA=1|R=3)
        R = 4   P(GPA=0|R=4)  | P(GPA=1|R=4)
"""
gpa_rank = defaultdict(lambda: defaultdict(int))
#Suavizado landa
for gpa in GPA:
        for r in rank:
            gpa_rank[gpa][r] = laplace
#Calculo de frecuencias
for f, row in df.iterrows():
    admit2 = row['admit']
    gpa2 = row['gpa']
    gre2 = row['gre']
    rank2 = row['rank']

    gpa_rank[gpa2][rank2]+=1
#Calculo de probabilidades
print()
count=0
for r in rank:
    total = sum(gpa_rank[gpa][r] for gpa in GPA)
    if total > 0:  
        for gpa in GPA:
            count+=gpa_rank[gpa][r]
            gpa_rank[gpa][r] = gpa_rank[gpa][r] / total
            print(f"P(GPA={gpa}|rank={r})={gpa_rank[gpa][r]})")
            filtro = (df['gpa'] == gpa) & (df['rank'] == r)
            conteo = df[filtro].shape[0]
            filtro = (df['rank'] == r)
            conteo2 = df[filtro].shape[0]
            print(f"Hay {conteo} registros de {conteo2} que cumplen con los valores especificados")
print(f"total={count}")


"""
Vamos a armar
P(R)
"""

rank_dict = defaultdict()
#Suavizado landa
for r in rank:
    rank_dict[r] = 0

#Calculo de frecuencias
for f, row in df.iterrows():
    admit2 = row['admit']
    gpa2 = row['gpa']
    gre2 = row['gre']
    rank2 = row['rank']

    rank_dict[rank2]+=1

#Calculo de probabilidades
print()
total = sum(rank_dict[r] for r in rank)
if total > 0:  
    for r in rank:
        rank_dict[r] = rank_dict[r] / total
print(rank_dict)
def calculo_de_probabilidades_conjunta(r,a,gre,gpa):
    return rank_dict[r]*gpa_rank[gpa][r]*gre_rank[gre][r]*admit_rgg[a][gre][gpa][r]
    

# -------------------------------
# 4. Cálculo de las Probabilidades Solicitadas
# -------------------------------
print()
# a) Calcular la probabilidad de que una persona que proviene de una escuela con rango 1 no haya sido admitida en la universidad

prob = 0
r = 1
a = 0

# Para calcular P(Admit=0,rank=1)
for v1 in GPA:
    for v2 in GRE:
        print(f"Probabilidad de P(A={a},r={r},gre={v2},gpa={v1})=(P(r)={rank_dict[r]}*P(GPA|r)={gpa_rank[v1][r]}*P(GRE|r)={gre_rank[v2][r]}*P(A|r,gre,gpa))={admit_rgg[a][v2][v1][r]}={rank_dict[r]*gpa_rank[v1][r]*gre_rank[v2][r]*admit_rgg[a][v2][v1][r]}")
        filtro = (df['gre'] == v2) & (df['gpa'] == v1) & (df['rank'] == r) & (df['admit'] == a)
        conteo = df[filtro].shape[0]
        conteo2 = df.shape[0]
        print(f"Hay {conteo} registros de {conteo2} que cumplen con los valores especificados")
        prob+=rank_dict[r]*gpa_rank[v1][r]*gre_rank[v2][r]*admit_rgg[a][v2][v1][r]
print(f"Probabilidad de que una persona con rank={r} no haya sido admitida: {prob}")

filtro = (df['rank'] == r) & (df['admit'] == a)
conteo = df[filtro].shape[0]
conteo2 = df.shape[0]

# Imprimir el resultado
print(f"Hay {conteo} registros de {conteo2} que cumplen con los valores especificados y no han sido adminitidos")

print()
# b) Probabilidad de que una persona que proviene de una escuela con rank=2 y tiene GPA='high' y GRE='low' no haya sido admitida
r=2
gpa = "high"
gre = "low"
a = 0
print(f"Probabilidad de que una persona con rank={r}, GPA={gpa} y GRE={gre} no haya sido admitida:")
print(f"P(A,R,GRE,GPA)={calculo_de_probabilidades_conjunta(r,a,gre,gpa)}")
print(f"P(A|R,GRE,GPA)={admit_rgg[a][gre][gpa][r]}")

filtro = (df['gre'] == gre) & (df['gpa'] == gpa) & (df['rank'] == r) & (df['admit'] == a)
conteo = df[filtro].shape[0]
filtro = (df['gre'] == gre) & (df['gpa'] == gpa) & (df['rank'] == r)
conteo2 = df[filtro].shape[0]

# Imprimir el resultado
print(f"Hay {conteo} registros de {conteo2} que cumplen con los valores especificados y no han sido adminitidos")

