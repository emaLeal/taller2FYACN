import numpy as np

def f(vec):
    x, y, t0 = vec
    v = 5  # velocidad en km/s
    
    f1 = np.sqrt(x**2 + y**2) - v * (12 - t0)
    f2 = np.sqrt((x - 100)**2 + y**2) - v * (28.5 - t0)
    f3 = np.sqrt(x**2 + (y - 100)**2) - v * (28.5 - t0)
    
    return np.array([f1, f2, f3])

def jacobian(vec):
    x, y, t0 = vec
    v = 5
    
    d1 = np.sqrt(x**2 + y**2)
    d2 = np.sqrt((x - 100)**2 + y**2)
    d3 = np.sqrt(x**2 + (y - 100)**2)
    
    J = np.zeros((3, 3))
    
    # Derivadas parciales de f1
    J[0, 0] = x / d1
    J[0, 1] = y / d1
    J[0, 2] = v
    
    # Derivadas parciales de f2
    J[1, 0] = (x - 100) / d2
    J[1, 1] = y / d2
    J[1, 2] = v

    # Derivadas parciales de f3
    J[2, 0] = x / d3
    J[2, 1] = (y - 100) / d3
    J[2, 2] = v

    return J

def newton_raphson(F, J, x0, tol=0.001, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = F(x)
        Jx = J(x)
        delta = np.linalg.solve(Jx, -fx)
        x = x + delta
        if np.linalg.norm(delta, ord=2) < tol:
            print(f"Convergió en {i+1} iteraciones")
            return x
    raise RuntimeError("No convergió")




# Primer vector inicial
x0_1 = np.array([50.0, 50.0, 10.0])

# Segundo vector inicial
x0_2 = np.array([90.0, 80.0, 5.0])

print("Solución con x0_1 = [50, 50, 0]:")
sol1 = newton_raphson(f, jacobian, x0_1)
print(f"x = {sol1[0]:.4f} km, y = {sol1[1]:.4f} km, t0 = {sol1[2]:.4f} s\n")

print("Solución con x0_2 = [20, 80, 5]:")
sol2 = newton_raphson(f, jacobian, x0_2)
print(f"x = {sol2[0]:.4f} km, y = {sol2[1]:.4f} km, t0 = {sol2[2]:.4f} s")
