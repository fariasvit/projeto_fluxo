"""Executa Gauss-Seidel com o chute plano de referência."""
import numpy as np
from algoritmo_seidel import gauss_seidel
from dados_sistema import INDICE_SLACK, INDICES_PV, V_MOD_ESPEC
from modelagem import montar_ybus, montar_tensao_inicial, calcular_potencias


def main():
    Ybus = montar_ybus()
    V0 = montar_tensao_inicial(
        pq_modulos=[1.0] * 5,
        pq_angulos_graus=[0.0] * 5,
        pv_angulos_graus=[0.0] * 2,
    )
    V, n, erro, historico, convergiu = gauss_seidel(Ybus, V0)
    print("MÉTODO DE GAUSS-SEIDEL")
    print(f"Convergiu: {'Sim' if convergiu else 'Não'}")
    print(f"Iterações: {n}")
    print(f"Mismatch máximo: {erro:.10e} p.u.")
    print("\nTensões finais:")
    print("Barra | Tipo  | Módulo (p.u.) | Ângulo (graus)")
    tipos = ["Slack", "PV", "PQ", "PQ", "PQ", "PQ", "PV", "PQ"]
    for i, v in enumerate(V):
        print(f"{i+1:5d} | {tipos[i]:5s} | {abs(v):13.8f} | {np.degrees(np.angle(v)):14.8f}")
    P, Q = calcular_potencias(V, Ybus)
    print("\nPotências calculadas:")
    print("Barra | P calculada (p.u.) | Q calculada (p.u.)")
    for i in range(len(V)):
        print(f"{i+1:5d} | {P[i]:17.8f} | {Q[i]:17.8f}")
    if not convergiu:
        print("\nAviso: o método atingiu o limite de iterações sem satisfazer a tolerância.")

if __name__ == "__main__":
    main()
