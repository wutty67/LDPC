import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import keyboard
import time
from pyldpc import make_ldpc, decode, get_message, encode
from belief_propagation import TannerGraph, bsc_llr

######################################################################################################
#                                     FUNCIONES                                                      #
######################################################################################################


######################################################################################################
#                           IMPRIME MATRIZ GENERADORA                                                 #
######################################################################################################


def Generadora():
    print ("La matriz Generadora es ",n," x ",k)
    print ("Longitud del Código :",LongitudCodigo)
    print("Bits de Datos : ", k)
    print ("Matriz Generadora : ")
    print (G)
    EsperaTecla()


######################################################################################################
#                           IMPRIME MATRIZ VERIFICADORA                                              #
######################################################################################################

def Verificadora():        
    print ("La matriz Verificadora es ",m," x ",r)
    print ("Matriz Verificadora : ")
    print (H)
    EsperaTecla()

######################################################################################################
#                           GRAFO DE TANNERI                                                         #
######################################################################################################

def GrafoTanner():
    print ("Para continuar con la ejecución del programa, debe de cerrar la pantalla gráfica")
    tg = TannerGraph.from_biadjacency_matrix(H, channel_model=bsc_llr(0.1))
    g = tg.to_nx()
    fig = plt.figure()
    top = nx.bipartite.sets(g)[0]
    labels = {node: d["label"] for node,d in g.nodes(data=True)}
    nx.draw_networkx(g,
                 with_labels=True,
                 node_color=[d["color"] for d in g.nodes.values()],
                 pos=nx.bipartite_layout(g, top),
                 labels=labels)
    
    
    nx.draw_networkx(g,
                 with_labels=True,
                 node_color=[d["color"] for d in g.nodes.values()],
                 pos=nx.bipartite_layout(g, top),
                 labels=labels)   
    
    plt.title=('Matriz Regular  ({},{},{})'.format(n,k,n-k))
    plt.show()

def EsperaTecla():
    print("Presione 'Esc' para continuar...")
    key=" "
    while key != 'esc':
        key = keyboard.read_key()
    os.system("cls")

def actualiza(iteraciones, cont,actual,total,incremento=500):
    ultimo = actual
    actual = time.time()
    diferencia = actual-ultimo
    return total + diferencia

def Representa_Tiempo_frente_Iteraciones():

    print("Se comparan tiempos y BER entre dos iteraciones distintas del algoritmo BP")
    iteraciones1=input("Número de Iteraciones (1) :")
    iteraciones2=input("Número de Iteraciones (2) :")
    iteraciones1=int(iteraciones1)
    iteraciones2=int(iteraciones2)

    NumeroMensajes = int(1000)
    incremento = NumeroMensajes/4

    v = np.random.randint(2, size=(NumeroMensajes,k))
    min_snr=0
    max_snr=10
    snrs = np.arange(min_snr,max_snr,0.5)
    tiempos1 = np.array(())
    tiempos2 = np.array(())
    errores1 = np.array(())
    errores2 = np.array(())
    for snr in snrs:
        Cantidad_errores=0
        total1 = 0
        actual = time.time()
        for cont in range(NumeroMensajes):
            v_i = v[cont]
            y = encode(G, v_i, snr)
            d = decode(H, y, snr, iteraciones1)
            x = get_message(G, d)
            if abs(x-v_i).sum() != 0 :
                Cantidad_errores+=1;

            if (cont+1) % incremento == 0:
                total1 = actualiza(iteraciones1,cont,actual,total1,incremento)
        errores=float(Cantidad_errores/NumeroMensajes)        
        print('ITERACIONES {} SNR: {:04.2f} BER: {:02.2f} Tiempo de Cálculo: {:03.2f}s'.format(iteraciones1,snr,errores,total1))
        tiempos1=np.append(tiempos1,total1)
        errores1=np.append(errores1,errores)
  
    for snr in snrs:
        print
        total2 = 0
        actual = time.time()
        Cantidad_errores=0
        for cont in range(NumeroMensajes):
            v_i = v[cont]
            y = encode(G, v_i, snr)
            d = decode(H, y, snr, round(iteraciones2))
            x = get_message(G, d)
            if abs(x-v_i).sum() != 0 :
                Cantidad_errores+=1;
            if (cont+1) % incremento == 0:
                total2 = actualiza(iteraciones2, cont,actual,total2,incremento)
        errores=float(Cantidad_errores/NumeroMensajes)        
        print('ITERACIONES {} SNR: {:04.2f} BER: {:02.2f} Tiempo de Cálculo: {:03.2f}s'.format(iteraciones2,snr,errores,total2))
        tiempos2=np.append(tiempos2,total2)
        errores2=np.append(errores2,errores)
    


    # Dibujar
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))

    ax1.set_title("Tiempos de Cálculo y BER para "+str(iteraciones1)+" y "+str(iteraciones2)+" Iteraciones")
    ax1.plot(snrs, tiempos1, color="indianred", label = str(iteraciones1)+" Iteraciones")
    ax1.plot(snrs, tiempos2, color="darkblue", label = str(iteraciones2)+" Iteraciones")
    ax1.set_ylabel("Tiempo de decodificación en segundos")
    ax1.legend([ str(iteraciones1)+" Iteraciones", str(iteraciones2)+" Iteraciones"],loc="upper right", fontsize="x-large")
    ax2.plot(snrs, errores1, color="indianred", label = str(iteraciones1)+" Iteraciones")
    ax2.plot(snrs, errores2, color="darkblue", label = str(iteraciones2)+" Iteraciones")
    ax2.set_ylabel("BER")
    ax2.set_xlabel ("SNR")
    plt.show()

def Representa_BER_frente_Ruidos():
    iteraciones=input ("Iteraciones : ")
    iteraciones=int(iteraciones)
    inicio=time.time()
    errores = []
    snrs = np.linspace(-2, 10, 25)
    v = np.arange(k) % 2  # fixed k bits message
    intentos = 50  # number of transmissions with different noise
    V = np.tile(v, (intentos, 1)).T  # stack v in columns

    anterior=time.time();
    for snr in snrs:
        y = encode(G, V, snr, seed=seed)
        D = decode(H, y, snr, iteraciones)
        actual=time.time()
        tiempo=actual-anterior
        error = 0.
        for i in range(intentos):
            x = get_message(G, D[:, i])
            error += abs(v - x).sum() / (k * intentos)
        errores.append(error)
        print("SNR  ",round(snr,3),"  en ", round(tiempo,3), "Segundos con",round(error*k*intentos)," errores")
        anterior=actual

    plt.figure()
    plt.plot(snrs, errores, color="indianred")
    plt.ylabel("Bit error rate")
    plt.xlabel("SNR")
    final=time.time()
    tiempo=format(final-inicio,".2f")
    plt.title("Número de Iteraciones "+str(iteraciones)+ " en "+tiempo+" Segundos")
    plt.show()


######################################################################################################
#                                             PROGRAMA PRINCIPAL                                      #
#######################################################################################################



######################################################################################################
#                                    INTRODUCCION DE PARÁMETROS                                      #
#######################################################################################################
print ("INICIANDO")
# Se piden datos al usuario
os.system("cls")
print ("                             SIMULACIÓN DE LDPC CON MATRICES REGULARES")
print("")
print ("Se pedirán 3 parámetros :")
print ("     1.- Longitud del código, número de bits del código")
print ("     2.- Número de unos por columna de la matriz verificadora (Wc)")
print ("     3.- Número de unos por fila de la matriz verificadora (Wr)")
print("")
print ("  - Si el número de bits de código es mayor de 15 la visualización del grafo de Tanner será confusa por el exceso de líneas, ")
print ("  - Wc debe de ser mayor que 2 y menor del tamaño del código")
print ("  - Wr debe ser mayor que Wc y divisor del tamaño del código")
print ("  - Entradas aceptables son 15,2,3  15,2,5 15,4,5  16,4,8 .........")
print()
print()


# Se genera Semilla aleatoria
seed = np.random.RandomState(42)
LongitudCodigo=0
Wc=0
Wr=0

while LongitudCodigo<=4:
    LongitudCodigo = input ("Número de bits de Código (mayor o igual que 4): ")
    LongitudCodigo=int(LongitudCodigo)

print()    
while (Wc<2) or (Wc>LongitudCodigo):
    Wc = input ("Wc -> Unos por columna, debe ser menor o igual al Número de bits de código y mayor o igual 2) :")
    Wc=int(Wc)

permitidos=[];                   #Cáculo de los valores permitidos para Wr
for cont in range (Wc+1,LongitudCodigo):  
    if LongitudCodigo//cont == LongitudCodigo/cont:
        permitidos.append(cont)

print()    
while (Wr<Wc) or (LongitudCodigo//Wr != LongitudCodigo/Wr):
    print ()
    print() 
    cad="["
    for c in permitidos:
        cad+=str(c)+" "   
    cad+="]"    
    cad="Wr -> Unos por fila (Mayor que unos por columna y divisor de Número de bits de código) Valores permitidos "+cad+" : "
        
    Wr = input (cad)
    Wr=int(Wr)

######################################################################################################
#                               CONSTRUCCION DE LAS MATRICES                                          #
#######################################################################################################

H, G = make_ldpc(LongitudCodigo, Wc, Wr, seed=seed, systematic=True, sparse=True)
n, k = G.shape
m, r=H.shape


######################################################################################################
#                               ESCRITURA DE PARAMETROS EN DISCO                                     #
######################################################################################################

fichero=input ("Nombre del fichero LDPC (Se almacenarán las matrices y parámetros)")
f=open(fichero+".ldpc","w")
f.write("Wr -->"+str(Wr)+'\n')
f.write("Wc -->"+str(Wc)+'\n')
f.write("n  -->"+str(n)+'\n')
f.write("k  -->"+str(k)+'\n')
f.write("m  -->"+str(m)+'\n')
f.write("R  -->"+str(r)+'\n')
f.write('G  -->'+"\n")

for i in range (n):
    for j in range (k):
        f.write(str(G[i,j]))
    f.write("\n")

f.write('H  -->'+"\n")
for i in range (m):
    for j in range (r):
        f.write(str(H[i,j]))
    f.write("\n")

f.close()



######################################################################################################
#                                          MENU                                                      #
#######################################################################################################

os.system("cls")
opcion=" "
while (opcion!="0"):             # Se muestra el menú hasta que se pulse 0 --> Salir
    os.system("cls")
    print ("1 - Datos de la Matriz Generadora")
    print ("2 - Datos de la Matriz Verificadora")
    print ("3 - Gráfico de Tanner")
    print ("4 - Simulación de ruido")
    print ("5.- Tiempo y BER por Iteraciones de BP")
    print ("0 - Salir")
    opcion=keyboard.read_key(True)

    if (opcion=="1"):
        Generadora()
    if (opcion=="2"):
        Verificadora()
    if (opcion=="3"):    
        GrafoTanner()
    if (opcion=="4"):
        Representa_BER_frente_Ruidos()
    if (opcion=="5"):
        Representa_Tiempo_frente_Iteraciones()

os.system("cls")
os._exit(0)          # Se sale de la Aplicación
