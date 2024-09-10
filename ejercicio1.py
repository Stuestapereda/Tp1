import csv
import numpy as np

def Extraer_data(data):
    BD_1=[] #lista para calcular la probabilidad a priori y la evidencia
    BD_2=[[],[]] #lista para calcular la verosimilitud, BD_2[0] atributos para clase I, BD_2[1] atributos para clase E

    with open(data, 'r') as archivo:
        lector_csv = csv.reader(archivo) # Abrir el archivo en modo lectura
        next(lector_csv) #Saltearse la primera fila
        
        for fila in lector_csv: #Itero en cada fila del archivo
            for i,elemento in enumerate(fila): #Itero en cada elemento de cada fila
                #Para BD_1
                if len(BD_1)==0:
                    BD_1=[[] for e in range(len(fila))] #Llenar el vector con listas vacias en función del número de atributos, cada lista dentro de BD_1 corresponde a las anotacioens de un atributo
                
                if elemento!="I" and elemento!="E":
                    elemento=int(elemento) #Convertir a int aquellos elementos que no sean letras
                BD_1[i].append(elemento)

                #Para BD_2
                if fila[-1]=="I":
                    if len(BD_2[0])==0:
                        BD_2[0]=[[] for e in range(len(fila[:-1]))] #Llenar el vector con listas vacias en función del número de atributos, cada lista dentro de BD_2[0] corresponde a las anotacioens de un atributo para la clase I
                    
                    if elemento!="I" and elemento!="E":
                        BD_2[0][i].append(int(elemento)) #Convertir a int aquellos elementos que no sean letras

                else:
                    if len(BD_2[1])==0:
                        BD_2[1]=[[] for e in range(len(fila[:-1]))] #Llenar el vector con listas vacias en función del número de atributos, cada lista dentro de BD_2[1] corresponde a las anotacioens de un atributo para la clase E
                    
                    if elemento!="I" and elemento!="E":
                        BD_2[1][i].append(int(elemento)) #Convertir a int aquellos elementos que no sean letras
    return BD_1,BD_2

Data='PreferenciasBritanicos.csv'
BD_1,BD_2 = Extraer_data(Data)

def calcular_probablidades(BD_1,BD_2):
    #Creamos listas donde almaceneremos las probabilidades
    priori=[] #priori[0]=P(I) y priori[1]=P(E)
    evidencia=[]  #Evidencia[i]=P_i(atributo_i==1) MISMO ORDEN QUE EN EL DATASET
    verosimilitud=[[] for e in range(len(BD_2))] #Versomilitud[0][i] = P(atributo_i==1|I) y Versomilitud[1][i] = P(atributo_i==1|E)

    #Para calcular a priori y evidencia
    #Se agrega 1 valor a cada atributo y se vidie entre 2 (correcioin de laplace)
    for i,atributos in enumerate(BD_1):
        if i!=len(BD_1)-1:
            conteo=sum([elemento for elemento in atributos if elemento==1])
            total=len(atributos)
            p=(conteo+2)/(total+4) #Calculando la probabilidad de que atributo_i==1
            evidencia.append(p)
        else:
            conteo=sum([1 for elemento in atributos if elemento=="I"]) #Calculando la probabilidad de que la nacionanalidad sea I
            total=len(atributos)
            p=(conteo+2)/(total+4)
            priori.append(p)
            priori.append(1-p) #Calculando la probabilidad de que la nacionanalidad sea E

    #Para calcular la verosimulitud
    for i,clase in enumerate(BD_2):
        for e,atributos in enumerate(clase):
            conteo=sum([elemento for elemento in atributos if elemento==1])
            total=len(atributos)
            p=(conteo+1)/(total+2) #Calculando la probabilidad de que atributo_e==1 dado la clase i 
            verosimilitud[i].append(p)

    return evidencia, priori, verosimilitud

evidencia, priori, verosimilitud = calcular_probablidades(BD_1,BD_2)

def clasificar(lista,evidencia, priori, verosimilitud):

    #Calculo de probabilidad de evidencia
    prob_evi=1
    for i,elemento in enumerate(lista):
        if elemento==1:
            prob_evi*=evidencia[i]
        else:
            prob_evi*=(1-evidencia[i])

    
    prob_ver= [1 for e in range(len(priori))]
    #Calculo de probabilidad de verosimilitud
    for i  in range(len(prob_ver)):
        for e,elemento in enumerate(verosimilitud[i]):
            if lista[e]==1:
                prob_ver[i]*=elemento
            else:
                prob_ver[i]*=(1-elemento)
    
    #Calculo de probabilidad de clases
    clases = [1 for e in range(len(priori))]
    for i in range(len(clases)):
        if prob_evi != 0:
            clases[i] = (prob_ver[i] * priori[i]) / prob_evi
        else:
            clases[i] = 0  # O algún valor que refleje el escenario

    
    print(sum(clases))

    if clases[0]>clases[1]:
        prob=clases[0]
        nacionalidad="Inglesa"
    else:
        prob=clases[1]
        nacionalidad="Escosesa"

    return prob,nacionalidad


for i,elemento in enumerate(BD_1):
    if i==len(BD_1)-1:
        print(f"Clases: {elemento}")
    else:
        print(f"Atributos_{i+1}: {elemento}")
    

for i,clase in enumerate(BD_2):
    for e,atributo in enumerate(clase):
        print(f"Clase_{i+1}_atributo{e+1}: {atributo}")


for i,p in enumerate(evidencia):
    print(f"probabilidad_de_atributo_{i+1}:{p}")


for i,p in enumerate(priori):
    print(f"probabilidad_de_clase_{i+1}:{p}")

for i,clase in enumerate(verosimilitud):
    for e,atributo in enumerate(clase):
        print(f"Clase_{i+1}_atributo_{e+1}: {verosimilitud[i][e]}")

p,n = clasificar([1,0,1,1,0],evidencia, priori, verosimilitud)
print(f"La probliadad de que la nacionalidad sea {n} es de {p*100}%")

p,n = clasificar([0,1,1,0,1],evidencia, priori, verosimilitud)
print(f"La probliadad de que la nacionalidad sea {n} es de {p*100}%")