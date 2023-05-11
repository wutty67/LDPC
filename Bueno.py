import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import keyboard
import time
from networkx.algorithms import bipartite
from pyldpc import make_ldpc, decode, get_message, encode
from belief_propagation import TannerGraph, bsc_llr

######################################################################################################
#                                     FUNCIONES                                                      #
######################################################################################################

def Generadora():
    print ("La matriz Generadora es ",n," x ",k)
    print ("Longitud del Código :",LongitudCodigo)
    print("Bits de Datos : ", k)
    print ("Matriz Generadora : ")
    print (G)
    EsperaTecla()

def Verificadora():        
    print ("La matriz Verificadora es ",m," x ",r)
    print ("Matriz Verificadora : ")
    print (H)
    EsperaTecla()

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
    plt.show()

def EsperaTecla():
    print("Presione 'Esc' para continuar...")
    key=" "
    while key != 'esc':
        key = keyboard.read_key()
    os.system("cls")

def RepresentaRuidos():
   iteraciones=input ("Iteraciones : ")
   iteraciones=int(iteraciones)
   inicio=time.time()
   errores = []
   snrs = np.linspace(-2, 10, 20)
   v = np.arange(k) % 2  # fixed k bits message
   intentos = 50  # number of transmissions with different noise
   V = np.tile(v, (intentos, 1)).T  # stack v in columns

   anterior=time.time();
   for snr in snrs:
      y = encode(G, V, snr, seed=seed)
      D = decode(H, y, snr, iteraciones)
      actual=time.time()
      print("Ruido ",snr, " en ",actual-anterior, "Segundos")
      anterior=actual
      error = 0.
      for i in range(intentos):
         x = get_message(G, D[:, i])
         error += abs(v - x).sum() / (k * intentos)
      errores.append(error)


   plt.figure()
   plt.plot(snrs, errores, color="indianred")
   plt.ylabel("Bit error rate")
   plt.xlabel("SNR")
   final=time.time()
   
   plt.title("Número de Iteraciones "+str(iteraciones)+ " en "+str(final-inicio))
   plt.show()

""""
LongitudCodigo=9
Wr=3
Wc=2
"""
# Se piden datos al usuario
os.system("cls")
LongitudCodigo=0
while LongitudCodigo<=4:
    LongitudCodigo = input ("Número de bits de Código (mayor o igual que 4): ")
    LongitudCodigo=int(LongitudCodigo)

Wc=0
while (Wc<2) or (Wc>LongitudCodigo):
    Wc = input ("Unos por columna, debe ser menor o igual al Número de bits de código y mayor o igual 2) :")
    Wc=int(Wc)

permitidos=[];
for cont in range (Wc+1,LongitudCodigo):
    if LongitudCodigo//cont == LongitudCodigo/cont:
        permitidos.append(cont)

Wr=0
while (Wr<Wc) or (LongitudCodigo//Wr != LongitudCodigo/Wr):
    print ("Valores permitidos : ",permitidos)
    Wr = input ("Unos por fila (Mayor que unos por columna y divisor de Número de bits de código) :")
    Wr=int(Wr)

 # Se genera Semilla aleatoria
seed = np.random.RandomState(42)

# Se construyen la matriz Generadora G y la matri verificadora H y se obtienen sus dimensiones
H, G = make_ldpc(LongitudCodigo, Wc, Wr, seed=seed, systematic=True, sparse=True)
n, k = G.shape
m, r=H.shape

# Generamos el gáfico de Tanner



######################################################################################################
#                                     MENU                                                      #
######################################################################################################


os.system("cls")
key=" "
while (key!="0"):
    os.system("cls")
    print ("1 - Datos de la Matriz Generadora")
    print ("2 - Datos de la Matriz Verificadora")
    print ("3 - Gráfico de Tanner")
    print ("4 - Simulación de ruido")
    print ("0 - Salir")
    key=keyboard.read_key()
    print(key)
    if (key=="1"):
        Generadora()
    if (key=="2"):
        Verificadora()
    if (key=="3"):    
        GrafoTanner()
    if (key=="4"):
        RepresentaRuidos()

print("Saliendo")



"""""
          ASI CREABA YO EL GRÄFICO PERO DESCUBRI LA LIBRERIA BELIEF_PROPAGATION 
          QUE GENERA EL GRÁFICO DIRECTAMENTE. YO NO ERA CAPAZ DE ORDENAR EL GRAFICO
          Y BUSCANDO COMO ORDENARLO, LA ENCONTRÉ.
          DE TODAS FORMAS; DEJO EL CÖDIGO HAGO LOS RECORRIDOS DE LA MATRIZ PARA CREAR NODOS
          Y ARCOS Y ES POSIBLE QUE VUELVA A NECESITARLO


#          CREO GRAFICO DE TANNER
T = nx.Graph()

#   Inserto Nodos de Fila
NodosF=[]
for i in range (m):
   nodo='F'+str(i+1)
   NodosF.append(nodo)

print (NodosF)
T.add_nodes_from(NodosF, bipartite=0)

#         Inserto nodos de Columna

NodosC=[]
for j in range (r):
   nodo='C'+str(j+1)
   NodosC.append(nodo)
print (NodosC)
T.add_nodes_from(NodosC,bipartite=1)

print (T)

#       Inserto Relaciones

Miedges=[];
for i in range (m):
   for j in range (r):
        if H[i,j]==1:
            edge='F'+str(i+1), 'C'+str(j+1)
            print (edge)
            Miedges.append(edge)

          
print (Miedges)
T.add_edges_from(Miedges)

#   COLOREO LOS NODOS

NodosC, NodosF = bipartite.sets(T)
color = bipartite.color(T)
color_dict={0:'b', 1:'r'}
color_list = [color_dict[i[1]] for i in T.nodes.data('bipartite')] 


nx.draw_networkx (T , pos = nx.drawing.layout.bipartite_layout(T, NodosC), node_color = color_list)
plt.show()"""


