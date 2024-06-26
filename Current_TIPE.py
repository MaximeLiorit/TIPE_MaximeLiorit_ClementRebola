import random
import numpy as np
import matplotlib.pyplot as plt

## CONSTANT :
g=9.809
bpoint = 7 # nombre de points de contrôle: nb point = bpoint+2
c=0.12  #coefficient défini empiriquement

n = 30 #nombre de toboggans initiaux

mutrate = .01 #coefficient de mutation (permet de limiter le risque d'erreurs)
x, y = (3, 1) #longueur et hauteur du toboggan
down = -1.5 #de combien on accepte d'aller en-dessous de la courbe des abscisses
iter = 15 #nombre d'itérations
precision=150 #nombre de points par courbe

l1 = [(0, y)] + [(x*i/bpoint, y) for i in range(1,bpoint+1)] + [(x,0)] #toboggan supérieur

l2 = [(0, y)] + [(x*i/bpoint, y*down-x/2) for i in range(bpoint)] + [(x,0)] #toboggan inférieur

npoint = len(l2) #nombre de points de contrôle + 2 (les extrémités)

## Function
def somme (T: list, c: float,n:int) -> list:
    # fonction qui calcule la somme garantissant la somme des influences des points de contrôle qui vaut 1 (ctte fonction n'est calculée qu'une seule fois pour nos courbes paramétriques)
    F=[]
    for t in T:
        f = 0
        for i in range(n):
            f += np.exp(-np.pi*c*((n*t)**2-2*t*n*i+i**2))
        F.append(f)
    return np.array(F)

T = np.linspace (-1, 2, precision) #création du tableau du paramètre (bornes du paramètres sont -1 et 2)

F = somme(T,c,npoint) #tableau contenant l'influence totale en fonction de T

## cycloid
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
    c,t = dicotocyclo (0.001)
    Tc =np.linspace (0, t, precision)
    Cx = (Tc- np.sin(Tc))*c
    Cy = (np.cos(Tc)-1)*c + y
    return Cx,Cy

Cx,Cy =cycloide()
plt.plot(Cx,Cy)
plt.show()

## Class
class Toboggan:
    def __init__(self, points, ecart, precision=200, mutrate=0.01):
        self.points = np.array(points)
        self.ecart = ecart
        self.precision = precision
        self.curve_X, self.curve_Y=self.generate_curve()
        self.time = self.chrono()
        self.mutrate = mutrate
    def generate_curve(self):
        """
            prend une liste de coordonnées de points et un nombre de point 'p' et un parametre de selcetivité 'c'
            renvoie un tuple de deux listes de points modélisant une courbe de p-points
        """
        l = self.points
        c = self.ecart
        p = self.precision
        n = len(l)
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

    def VssF (self) -> list :
        Y = self.curve_Y
        Y0 = Y[0]

        # Real formula : np.sqrt(2*g*(Y[0]-Y]))]
        return np.sqrt((Y0-Y)*2*g)

    def check (self) -> bool:
        Y = self.curve_Y
        Y0 = Y[0]
        return np.all(Y<=Y0)

    def __add__(self, other): #fonction permettant d'"additionner" 2 toboggans
        global x,y
        ecart = (self.ecart-other.ecart)*random.random()+other.ecart
        while True:
            points = []
            for i,((spx, spy), (opx, opy)) in enumerate(zip(self.points, other.points)):
                if points:
                    oldx, oldy = points[i-1]
                    sx, sy = max(spx, oldx), spy
                    ox, oy = max(opx, oldx), opy
                    points.append((sx+(ox-sx)*random.random(),sy+(oy-sy)*random.random()))
                    continue
                points.append((spx,spy))
            points = list(map(lambda p: (p[0] + random.uniform(-1, 1)*self.mutrate, p[1] + random.uniform(-1, 1)*self.mutrate) if (random.random() < self.mutrate) else p, points))
            points[0]=(0,y)
            points[-1]=(x,0)
            x0 = points[0][0]
            for x, _ in points[1:]:
                if x < x0:
                    break
                x0 = x
            else:
                break
        return Toboggan(points, ecart, self.precision)

    def chrono (self) -> float: #!! Renvoie un faux temps car très coûteux et seul la comparaison entre toboggans nous intéresse
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

## Genetic
t1 = Toboggan(l1,c,precision, mutrate)
t2 = Toboggan(l2,c,precision, mutrate)


plt.plot(t1.curve_X,t1.curve_Y)
plt.plot(*(t1.points.T),'+')

plt.plot(t2.curve_X,t2.curve_Y)
plt.plot(*(t2.points.T),'+')

plt.show()
##

print(f'iter=0/{iter}')

toboggans = [t1, t2] + list(map(lambda _: t1+t2, range(n*n-2))) #[t1+t2 for _ in range(n*n-2)]

for i in range(iter):
    print(f'iter={i+1}/{iter}')
    toboggans.sort()
    reduce_t = toboggans[:n]
    toboggans = []
    for i in reduce_t:
        for j in reduce_t:
            toboggans.append(i+j)
    better_show (toboggans)


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
## r show
def random_show(toboggans): # Dans la liste des n toboggans, affiche un toboggan quelconque
    t = random.choice(toboggans)
    print(t.time)
    plt.plot(*(t.points.T),'+')
    plt.plot(t.curve_X,t.curve_Y, label=f'time = {t.time}')
    plt.show()

def better_show(toboggans): # Dans la liste des n toboggans, affiche le meilleur
    t = min(toboggans)
    print(t.time)
    plt.plot(*(t.points.T),'+')
    plt.plot(t.curve_X,t.curve_Y, label=f'time = {t.time}')
    plt.show()