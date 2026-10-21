"""Executa Gauss-Jacobi com chute inicial plano e imprime os resultados."""
import numpy as np

from algoritmo_jacobi import gauss_jacobi
from dados_sistema import TIPOS_BARRA
from modelagem import montar_ybus, montar_tensao_inicial, calcular_potencias


def main():
    Ybus = montar_ybus()
    V0 = montar_tensao_inicial(
        pq_modulos=[1.0] * 5,
        pq_angulos_graus=[0.0] * 5,
        pv_angulos_graus=[0.0] * 2,
    )
    V, iteracoes, erro, historico, convergiu = gauss_jacobi(Ybus, V0)
    P, Q = calcular_potencias(V, Ybus)

    print("MÉTODO DE GAUSS-JACOBI")
    print(f"Convergiu: {'Sim' if convergiu else 'Não'}")
    print(f"Iterações: {iteracoes}")
    print(f"Mismatch máximo final: {erro:.10e} p.u.")
    print(f"Erro inicial: {historico[0]:.10e} p.u.")
    print("\nBarra | Tipo  | Módulo (p.u.) | Ângulo (graus) | P calculada (p.u.) | Q calculada (p.u.)")
    print("-" * 91)
    for i in range(len(V)):
        print(f"{i+1:5d} | {TIPOS_BARRA[i]:5s} | {abs(V[i]):13.6f} | "
              f"{np.degrees(np.angle(V[i])):14.6f} | {P[i]:17.6f} | {Q[i]:17.6f}")

    if not convergiu:
        raise SystemExit("O método não convergiu com os parâmetros definidos.")


if __name__ == "__main__":
    main()
