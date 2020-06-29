#Dylan Valerio Cortés
#B47180
#Tarea 3 Modelos Probabilisticos de Señales y Sistemas
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats 
from mpl_toolkits import mplot3d


################### Punto 1 #########################



A = []
Matriz_xy = []
Matriz_xyp = []

# Leemos el archivo xy
with open('xy.csv', newline='') as archivo:
  linea = archivo.read().splitlines()
  linea.pop(0)

  for l in linea:
    linea = l.split(',')
    A.append(linea[1:22])


for sublist in A:
 float_sublist = []  
 for x in sublist:
   float_sublist.append(float(x))    
 Matriz_xy.append(float_sublist)

#Hacemos un vector con la suma de las filas o las columnas según corresponda
fx = np.sum(Matriz_xy, axis = 1)
fy = np.sum(Matriz_xy, axis = 0)

#Creamos vectores de 5-15 para x y de 5-25 para y
Vx = np.arange(5, 16, 1)
Vy = np.arange(5, 26, 1)

#plt.plot(Vx, fx,'red')
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.plot(Vy, fy,'green')
#plt.xlabel('y')
#plt.ylabel('f(y)')
#plt.show()


#Definimos la función Gaussiana
def gaussiana(a, mu, sigma):
  return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(a - mu)**2/(2*sigma**2))

#Buscamos los parametros mu y sigma para fx y fy, para despues encontar la curva de ajuste
parametrosX, _ = curve_fit(gaussiana, Vx, fx)
parametrosY, _ = curve_fit(gaussiana, Vy, fy)

mu_x = parametrosX[0]
sigma_x = parametrosX[1]

mu_y = parametrosY[0]
sigma_y = parametrosY[1]

#plt.plot(Vx, stats.norm.pdf(Vx, mu_x, sigma_x),'red')
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.plot(Vy, stats.norm.pdf(Vy,mu_y, sigma_y),'green')
#plt.xlabel('y')
#plt.ylabel('f(y)')
#plt.show()


################### Punto 3 #########################

# Leemos el archivo xyp que es mas facil de trabajar
with open('xyp.csv') as f:

  lineas = f.read().splitlines() 
  lineas.pop(0)
  
  
  for row in lineas:
    linea = row.split(',')
    Matriz_xyp.append([float(linea[0]), float(linea[1]), float(linea[2])])

Fxy = []

for i in range(len(Matriz_xyp)):
  Fxy.append(Matriz_xyp[i][0] * Matriz_xyp[i][1] * Matriz_xyp[i][2])

Cxy = []

for i in range(len(Matriz_xyp)):
  Cxy.append((Matriz_xyp[i][0] - mu_x) * (Matriz_xyp[i][1] - mu_y) * Matriz_xyp[i][2])
  
correlacion = np.sum(Fxy, axis = 0)

covarianza = np.sum(Cxy, axis = 0)

coeficiente_correlacion = covarianza / (sigma_x * sigma_y)

print( correlacion, '\n', covarianza, '\n',  coeficiente_correlacion, '\n')

################# Punto 4 #########################

# Primero calculamos la ecuacion de la densidad conjunta de x y de y
def gaussiana_xy(a, mux, sigmax, b, muy, sigmay):
   return (1/(np.sqrt(2*np.pi*sigmax**2)) * np.exp(-(a - mux)**2/(2*sigmax**2)))*(1/(np.sqrt(2*np.pi*sigmay**2)) * np.exp(-(b - muy)**2/(2*sigmay**2)))

# Se grafica la pdf conjunta de x y y.
X, Y = np.meshgrid(Vx, Vy)

f = gaussiana_xy(X, mu_x, sigma_x, Y, mu_y, sigma_y)

#fig = plt.figure()
#eje = plt.axes(projection='3d')
#eje.plot_surface(X, Y, f, rstride=1, cstride=1,
#cmap='inferno', edgecolor='black')
#eje.set_xlabel('Valores de x')
#eje.set_ylabel('Valores de y')
#eje.set_zlabel('Función de densidad conjunta f(x,y) ')
#plt.show()