import random
import numpy as np
import matplotlib.pyplot as plt
import time
import math

## CONSTANT :
g=9.809
bpoint = 7              # nombre de points de contrôle: nb point = bpoint+2
c=0.12                  # coefficient défini empiriquement
n = 10                  # nombre de toboggans initiaux
m = 1.00               #masse (en kg)
gene_type = '2'         # '1' pour n**2 et '2' pour prendre 25% de la population
mutrate = .4            # Décalage maximal de la mutation selon l'axe des x et des y
mutchance = .05         # Pourcentage de chance de mutation
x, y = (4, 1)           # longueur et hauteur du toboggan
iter_max = 40           # nombre d'itérations maximales
down = -1.5             # de combien on accepte d'aller en-dessous de la courbe des abscisses
precision=400           # nombre de points par courbe
type_de_courbe = 'euler'# 'bezier' ou 'euler' ou 'lagrange'


l1 = [(0, y)] + [(x*i/bpoint, y) for i in range(1,bpoint+1)] + [(x,0)]      #toboggan supérieur
l2 = [(0, y)] + [(x*i/bpoint, y*down-x/2) for i in range(bpoint)] + [(x,0)] #toboggan inférieur

npoint = len(l2) #nombre de points de contrôle + 2 (les extrémités)

## FROTTEMENTS
frottement_solide = False
frottement_fluide = True
viscosite_dynamique = 10**-3 # pour l'eau: 10*-3 (Pa.s), pour l'air: 1.8*10**-6 (Pa.s)
masse_volumique = 997 #kg/m**3
Cx = 0.47 #sphere
rayon_objet = 0.01 #m
surface = np.pi*rayon_objet**2
V_max = np.sqrt(2*g*y)
Re= masse_volumique*V_max*2*rayon_objet/viscosite_dynamique #>>>100 donc frottements turbulants

## FUNCTION :
def somme (T: list, c: float,n:int) -> list:
    # fonction qui calcule la somme garantissant la somme des influences des points de contrôle qui vaut 1 (ctte fonction n'est calculée qu'une seule fois pour nos courbes paramétriques)
    F=[]
    for t in T:
        f = 0
        for i in range(n):
            f += np.exp(-np.pi*c*((n*t)**2-2*t*n*i+i**2))
        F.append(f)
    return np.array(F)

T = np.linspace (-1, 2, precision) #création du tableau du paramètre (les bornes du paramètre sont -1 et 2)

F = somme(T,c,npoint) #tableau contenant l'influence totale en fonction de T

def fact (n : int) -> int:
    """
    fait la factorielle de n
    """
    temp = 1
    while n > 0 :
        temp *= n
        n += -1
    return temp

def bezier (l :list, p: int) -> tuple:
    """
        prend une liste de coordonnées de points et un nombre de précision 'p'
        renvoie un tuple de deux listes de points modélisant une courbe de p-points
    """
    n = len(l) - 1
    Xp = []
    Yp = []
    for i in range(n+1):
        x,y= l[i]
        Xp = Xp + [x]
        Yp = Yp + [y]
    T = np.linspace (0,1,p)
    factn = fact(n)
    X = []
    Y = []
    for t in range(len(T)) :
        tempx = 0
        tempy = 0
        for i in range (n+1):
            tempx += factn/(fact(n-i)*fact(i))*Xp[i]*(T[t]**i)*(1-T[t])**(n-i)
            tempy += factn/(fact(n-i)*fact(i))*Yp[i]*(T[t]**i)*(1-T[t])**(n-i)
        X = X + [tempx]
        Y = Y + [tempy]
    return np.array(X),np.array(Y)

def lagrange(l:list,p: int)-> tuple:
    """
    prend une liste de coordonnés de point de controle et un nombre de point sur la courbe
    renvoie les polynome d'interpolation de lagrange
    """
    n= len(l)
    Xp, Yp = l[:,0], l[:,1]
    X = np.linspace(Xp[0],Xp[-1],p)
    Y=[]
    for x in X:
        y=0
        for j in range(n):
            temp =1
            for i in range (n):
                if i!=j:
                    temp*=(x-Xp[i])/(Xp[j]-Xp[i])
            y+= temp*Yp[j]
        Y.append(y)
    return X, np.array(Y)

def V_fluide_ss_solide(X,Y)-> list:
    V= [0]
    for i in range(1,len(X)):
        v_i_1= np.sqrt(abs(2*(V[i-1]**2/2+g*(Y[i-1]-Y[i]))))
        v_moy = (v_i_1+V[i-1])/2
        l = np.sqrt((X[i-1]-X[i])**2+(Y[i-1]-Y[i])**2)
        F = -0.5*masse_volumique*v_moy**2*Cx*surface
        Ef = F*l
        Etot=2*(V[i-1]**2/2+g*(Y[i-1]-Y[i])+Ef/m)
        if Etot<0:
            Etot=0
        V.append( np.sqrt(Etot))
    return np.array(V)

## CYCLOID :
def dicotocyclo(d):
    p,m = 0,2*np.pi
    t=(p+m)/2
    while abs((np.cos(t)+y/x*(t-np.sin(t))-1))>d:
        if (np.cos(t)+y/x*(t-np.sin(t))-1) <0:
            p = t
            t = (p+m)/2
        else :
            m = t
            t = (p+m)/2
    return y/(1-np.cos(t)), t

def cycloide():
    c,t = dicotocyclo(0.001)
    Tc =np.linspace (0, t, precision)
    Cx = (Tc- np.sin(Tc))*c
    Cy = (np.cos(Tc)-1)*c + y
    return Cx,Cy

def cyclo_show():
    Cx,Cy =cycloide()
    plt.plot(Cx,Cy,color ='k')
    plt.show()

def cyclo_time():
    X,Y = Cx,Cy
    pts = np.c_[X, Y]
    V = np.sqrt(2*g*(Y[0]-Y))
    temps = 0.0
    for i in range (len(X)-1):
        l = np.sqrt((X[i]-X[i+1])**2 + (Y[i]-Y[i+1])**2)
        if V[i] != 0:
            temps += l/(V[i])
    return temps

## CLASS TOBOGGAN :
class Toboggan:
    def __init__(self, points, ecart, precision=200, mutrate=0.2, mutchance=.01):
        self.points = np.array(points)
        self.ecart = ecart
        self.precision = precision
        self.curve_X, self.curve_Y=self.generate_curve()
        self.time = self.chrono()
        self.mutrate = mutrate
        self.mutchance = mutchance

    def generate_curve(self):
        """
            prend une liste de coordonnées de points et un nombre de point 'p' et un parametre de selcetivité 'c'
            renvoie un tuple de deux listes de points modélisant une courbe de p-points
        """
        l = self.points
        c = self.ecart
        p = self.precision
        n = len(l)
        if type_de_courbe == 'euler':
            Xp, Yp = l[:,0], l[:,1]
            X = []
            Y = []
            for t in range(len(T)) :
                tempx = 0
                tempy = 0
                for i in range (n):
                    tempx += Xp[i]*np.exp(-np.pi*c*((n*T[t])**2-2*T[t]*n*i+i**2))/F[t]
                    tempy += Yp[i]*np.exp(-np.pi*c*((n*T[t])**2-2*T[t]*n*i+i**2))/F[t]
                X = X + [tempx]
                Y = Y + [tempy]
            return np.array(X),np.array(Y)
        if type_de_courbe == 'bezier':
            return(bezier(l,p))
        if type_de_courbe == 'lagrange':
            return(lagrange(l,p))

    def VssF (self) -> list :
        if not(frottement_solide) and  not(frottement_fluide):
            Y = self.curve_Y
            Y0 = Y[0]
            return np.sqrt((Y0-Y)*2*g)
        elif not(frottement_solide):
            return V_fluide_ss_solide(self.curve_X,self.curve_Y)

    def check (self) -> bool:
        Y = self.curve_Y
        Y0 = Y[0]
        return np.all(Y<=Y0)

    def __add__(self, other): #fonction permettant d'"additionner" 2 toboggans
        global x,y
        ecart = (self.ecart-other.ecart)*random.random()+other.ecart
        points = []
        for i,((spx, spy), (opx, opy)) in enumerate(zip(self.points, other.points)):
            if points:
                oldx, oldy = points[i-1]
                sx, sy = max(spx, oldx), spy
                ox, oy = max(opx, oldx), opy
                points.append((sx+(ox-sx)*random.random(),sy+(oy-sy)*random.random()))
                continue
            points.append((spx,spy))
        points = list(map(lambda p: (p[0] + random.uniform(-1, 1)*self.mutrate, p[1] + random.uniform(-1, 1)*self.mutrate) if (random.random() < self.mutchance) else p, points))
        points[0]=(0,y)
        points[-1]=(x,0)
        x0 = points[0][0]
        for i, (xx, yy) in enumerate(points[1:]):
            if xx < x0:
                points[i+1]=(x0, yy)
            else:
                x0 = xx
        return Toboggan(points, ecart, self.precision)

    def chrono (self) -> float:
        if not self.check():
            return float('inf')

        X,Y = self.curve_X, self.curve_Y
        pts = np.c_[X, Y]
        V = self.VssF ()

        temps = 0.0

        for i in range (len(X)-1):
            l = np.sqrt((X[i]-X[i+1])**2 + (Y[i]-Y[i+1])**2)
            if V[i] != 0:
                temps += l/(V[i])
        return temps

    def __lt__(self,other):
        return self.time<other.time
## GENETIC BEHAVIOUR
def genetique_1 (t):
    n = math.floor(math.sqrt(len(t)))
    reduce_t = t[:n]
    toboggans = []
    for i in reduce_t:
        for j in reduce_t:
            toboggans.append(i+j)
    return toboggans

def genetique_2(t, per):
    reduce_t = t[:int(len(t)*per)+1]
    toboggans = []
    n= len(reduce_t)
    for i in range(len(t)):
        toboggans.append(reduce_t[random.randint(0,n-1)]+reduce_t[random.randint(0,n-1)])
    return toboggans
## BENCHMARK
def func_time(func, itt, *args, **kargs):
    t = time.time()
    for i in range(itt):
        func(*args, **kargs)
    return time.time()-t

## DISPLAY
def random_show(toboggans): # Dans la liste des n toboggans, affiche un toboggan quelconque
    t = random.choice(toboggans)
    print(t.time)
    plt.plot(*(t.points.T),'+')
    plt.plot(t.curve_X,t.curve_Y, label=f'time = {t.time}')
    plt.show()

def better_show(toboggans, tt=None): # Dans la liste des n toboggans, affiche le meilleur
    t = min(toboggans)
    print(t.time)
    if tt == None:
        plt.plot(*(t.points.T),'+')
        plt.plot(t.curve_X,t.curve_Y, label=f'time = {t.time}')
        plt.show()

def show_all(bests): # montre le meilleur toboggan de chaque génération
    l = len(bests)
    for i,t in bests:
        #plt.plot(*(t.points.T),'+')
        plt.plot(t.curve_X,t.curve_Y,color=(1-i/(l-1),1-i/(l-1),i/(l-1)), label=f'time = {t.time},G={i}')
    plt.show()

## PARENTS
t1 = Toboggan(l1,c,precision, mutrate, mutchance)
t2 = Toboggan(l2,c,precision, mutrate, mutchance)

"""
plt.plot(t1.curve_X,t1.curve_Y)
plt.plot(*(t1.points.T),'+')

plt.plot(t2.curve_X,t2.curve_Y)
plt.plot(*(t2.points.T),'+')

plt.show()
"""

toboggans = [t1, t2] + [t1+t2 for _ in range(n*n-2)]

## GENETIQUE

print(f'iter=0/{iter_max}')
temps_min = min(toboggans).time +1
bests = []
i=0
bouclage = 0
while i < iter_max and bouclage < 3:
    i+=1
    if min(toboggans).time >= temps_min-0.00001:
        bouclage+=1
    else:
        bouclage=0
        temps_min =min(toboggans).time
    print(f'iter={i}/{iter_max}')
    toboggans.sort()
    if gene_type == '2':
        toboggans = genetique_2(toboggans,0.25)
    elif gene_type == '1':
        toboggans = genetique_1(toboggans)
    bests.append((i-1,min(toboggans)))
print("fin")


""""[[ 0.0573955 ,  1.01383231],
       [ 0.10303722,  0.41847875],
       [ 0.41737845,  0.49103514],
       [ 0.70817726, -0.01044475],
       [ 1.00355546,  0.10833995],
       [ 1.20378191, -0.08724398],
       [ 1.55992421, -0.02873832],
       [ 1.84078545, -0.03567922],
       [ 1.95378423, -0.00598812]]
       c=0.12"""



