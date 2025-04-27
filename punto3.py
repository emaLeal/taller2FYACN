import numpy as np
import matplotlib.pyplot as plt

# Función del Polinomio
def p(x):
    return x**7 - 5*x**6 + 9*x**5 - 7*x**4 + 3*x**3 - 2*x**2 + x - 1

#Muller
x0 = 0
x1 = 1
x2 = 4

# Bairstow
r0 = -2
s0 = 3

#Valor de Tolerancia: 0.000001
tol = 1e-6

def muller(x0, x1, x2, max_iter=100):
    history = [x0, x1, x2]
    for i in range(max_iter):
        # Definimos las distancias entre cada punto
        h1 = x1 - x0
        h2 = x2 - x1
        # Definimos las pendientes
        p1 = (p(x1) - p(x0)) / h1
        p2 = (p(x2) - p(x1)) / h2
        # Definimos la curvatura
        d = (p2 - p1) / (h2 - h1)

        #Punto b
        b = p2 + h2 * d
        D = np.sqrt(b**2 - 4 * p(x2) * d)

        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b - D

        h = -2 * p(x2) / E
        x3 = x2 + h
        history.append(x3)

        #Verificar Convergencia
        if abs(h) < tol:
            return x3, history, i
        
        x0 = x1
        x1 = x2
        x2 = x3

    raise Exception("No se encontró la raíz en el número máximo de iteraciones")

def bairstow(coeficientes, r, s, max_iter=100):
    n = len(coeficientes) - 1
    b = np.zeros(n+1)
    c = np.zeros(n+1)

    history_r = [r]
    history_s = [s]

    for i in range(max_iter):
        b[n] = coeficientes[n]
        b[n - 1] = coeficientes[n - 1] + r * b[n]

        for j in range(n - 2, -1, -1):
            b[j] = coeficientes[j] + r * b[j + 1] + s * b[j + 2]

        c[n] = b[n]
        c[n - 1] = b[n - 1] + r * c[n]

        for j in range(n - 2, -1, -1):
            c[j] = b[j] + r * c[j + 1] + s * c[j + 2]

        determinante = c[2] * c[2] - c[3] * c[1]
        if abs(determinante) < 1e-12:
            raise Exception("Determinante muy pequeño")
        
        dr = (-b[1]*c[2] + b[0]*c[3]) / determinante
        ds = (-b[0]*c[2] + b[1]*c[1]) / determinante
        
        r += dr
        s += ds
        
        history_r.append(r)
        history_s.append(s)

        if abs(dr) < tol and abs(ds) < tol:
            break
        # Una vez que encontramos r y s, resolvemos las raíces del cuadrático
    discriminante = r**2 - 4*s
    if discriminante >= 0:
        root1 = (-r + np.sqrt(discriminante)) / 2
        root2 = (-r - np.sqrt(discriminante)) / 2
    else:
        root1 = complex(-r/2, np.sqrt(-discriminante)/2)
        root2 = complex(-r/2, -np.sqrt(-discriminante)/2)
    
    return (root1, root2), history_r, history_s, i

coeffs = [1, -5, 9, -7, 3, -2, 1, -1]




raiz, history_muller, iteraciones = muller(x0, x1, x2)
print(f"La raíz encontrada con el método de Müller es: {raiz:.6f} con {iteraciones} iteraciones")

plt.figure(figsize=(8,5))
plt.plot(range(len(history_muller)), history_muller, marker='o', label='Müller')
plt.title('Convergencia del Método de Müller')
plt.xlabel('Iteración')
plt.ylabel('Aproximación de la raíz')
plt.grid()
plt.legend()
plt.savefig("graficas/grafica_muller.png")  # Guarda la gráfica como imagen

(raiz1_bairstow, raiz2_bairstow), history_r, history_s, iteraciones = bairstow(coeffs, r0, s0)
print(f"Las raíces encontradas con Bairstow son: {raiz1_bairstow:.6f} y {raiz2_bairstow:.6f} con {iteraciones} iteraciones")

# Graficar evolución de r y s
plt.figure(figsize=(10,5))
plt.plot(range(len(history_r)), history_r, marker='o', label='r')
plt.plot(range(len(history_s)), history_s, marker='x', label='s')
plt.title('Convergencia del Método de Bairstow')
plt.xlabel('Iteración')
plt.ylabel('Valor de r y s')
plt.grid()
plt.legend()
plt.savefig("graficas/grafica_bairstow.png")  # Guarda la gráfica como imagen
