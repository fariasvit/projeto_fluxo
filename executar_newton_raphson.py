import numpy as np
from dados_sistema import N_BARRAS
from modelagem import montar_ybus, montar_tensao_inicial, calcular_potencias
from algoritmo_newton_raphson import newton_raphson

if __name__ == "__main__":
    Y = montar_ybus()
    V0 = montar_tensao_inicial([1.0]*5, [0.0]*5, [0.0]*2)
    V, n, erro, historico, convergiu = newton_raphson(Y, V0)
    print("MÉTODO NEWTON-RAPHSON")
    print(f"Convergiu: {convergiu}")
    print(f"Iterações: {n}")
    print(f"Mismatch máximo: {erro:.10e} p.u.")
    P, Q = calcular_potencias(V, Y)
    for i in range(N_BARRAS):
        print(f"Barra {i+1}: |V|={abs(V[i]):.6f} p.u., ângulo={np.degrees(np.angle(V[i])):.6f}°, P={P[i]:.6f} p.u., Q={Q[i]:.6f} p.u.")
