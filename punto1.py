import numpy as np
import matplotlib.pyplot as plt

# Parámetros
X = 3_000_000
Q = 50000
r = 0.08
A_target = 15_000_000
tol = 0.0001  # Tolerancia
e = np

# Función f(t)
def f(t):
    return X * e.exp(r * t) + (Q / r) * (e.exp(r * t) - 1) - A_target

# Derivada de f(t) para Newton-Raphson
def f_prime(t):
    return X * r * e.exp(r * t) + Q * e.exp(r * t)

# Gráfica de la función
t_vals = np.linspace(0, 30, 300)
f_vals = f(t_vals)

plt.figure(figsize=(10, 5))
plt.plot(t_vals, f_vals, label="f(t)", color='blue')
plt.axhline(0, color='black', linestyle='--', label='f(t) = 0')
plt.xlabel("t (años)")
plt.ylabel("f(t)")
plt.title("Gráfica de f(t) = A(t) - 15,000,000")
plt.grid(True)
plt.legend()
plt.savefig("grafica_ft.png")  # Guarda la gráfica como imagen

# Buscar intervalo con cambio de signo para Bisección y Secante
a, b = None, None
for i in range(len(t_vals) - 1):
    if f_vals[i] * f_vals[i + 1] < 0:
        a = t_vals[i]
        b = t_vals[i + 1]
        break

print(f"\nIntervalo con cambio de signo: t = {a:.4f} a {b:.4f}")

# Bisección
def biseccion(a, b, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        c = (a + b) / 2
        biseccion_iter.append(c)
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, abs(f(c))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, abs(f(c))

# Secante
def secante(x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        secante_iter.append(x2)
        if abs(f(x2)) < tol:
            return x2, abs(f(x2))
        x0, x1 = x1, x2
    return x2, abs(f(x2))

# Newton-Raphson
def newton(df, x0, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        fx, dfx = f(x0), df(x0)
        if dfx == 0:
            break
        x1 = x0 - fx / dfx
        newton_iter.append(x1)
        if abs(f(x1)) < tol:
            return x1, abs(f(x1))
        x0 = x1
    return x1, abs(f(x1))

# Guardar iteraciones para graficar
biseccion_iter = []
secante_iter = []
newton_iter = []

# Ejecutar métodos
t_bis, err_bis = biseccion( a, b, tol)
t_sec, err_sec = secante( a, b, tol)
t_newt, err_newt = newton( f_prime, (a + b) / 2, tol)

# Resultados
print(f"\nMétodo de Bisección:      t = {t_bis:.6f} años, Error = {err_bis:.2e}")
print(f"Método de la Secante:     t = {t_sec:.6f} años, Error = {err_sec:.2e}")
print(f"Método Newton-Raphson:    t = {t_newt:.6f} años, Error = {err_newt:.2e}")

# Gráficas de convergencia
plt.figure(figsize=(10, 6))
plt.plot(biseccion_iter, 'o-', label="Bisección")
plt.plot(secante_iter, 's--', label="Secante")
plt.plot(newton_iter, 'x-.', label="Newton-Raphson")
plt.axhline(t_bis, color='gray', linestyle='--', linewidth=0.5, label="Raíz estimada")
plt.title("Convergencia de los métodos numéricos")
plt.xlabel("Iteraciones")
plt.ylabel("Valor de t (años)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("grafica_biseccion.png")  # Guarda la gráfica como imagen
