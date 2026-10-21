"""Compara a solução dos três métodos e calcula fluxos nas linhas."""
import numpy as np
import pandas as pd

from modelagem import montar_ybus, montar_tensao_inicial, erro_maximo_mismatch
from algoritmo_jacobi import gauss_jacobi
from algoritmo_seidel import gauss_seidel
from algoritmo_newton_raphson import newton_raphson
from fluxos_linhas import calcular_fluxos_linhas
from dados_sistema import N_BARRAS


def main():
    Y = montar_ybus()
    V0 = montar_tensao_inicial([1.0]*5, [0.0]*5, [0.0]*2)
    metodos = {
        "Gauss-Jacobi": gauss_jacobi,
        "Gauss-Seidel": gauss_seidel,
        "Newton-Raphson": newton_raphson,
    }
    solucoes = {}
    resumo = []
    for nome, funcao in metodos.items():
        V, n, erro, historico, convergiu = funcao(Y, V0, tolerancia=1e-6)
        solucoes[nome] = V
        resumo.append({
            "Método": nome, "Convergiu": convergiu,
            "Iterações": n, "Mismatch máximo (p.u.)": erro
        })
    print("\nCOMPARAÇÃO DOS MÉTODOS")
    print(pd.DataFrame(resumo).to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    if not all(item["Convergiu"] for item in resumo):
        print("\nATENÇÃO: pelo menos um método não convergiu; não comparar como solução final.")
        return

    referencia = solucoes["Newton-Raphson"]
    print("\nDIFERENÇA MÁXIMA DE TENSÃO EM RELAÇÃO AO NEWTON-RAPHSON")
    for nome, V in solucoes.items():
        dif_mod = float(np.max(np.abs(np.abs(V) - np.abs(referencia))))
        dif_ang = float(np.max(np.abs(np.angle(V) - np.angle(referencia))))
        print(f"{nome}: diferença de módulo={dif_mod:.3e} p.u.; diferença angular={np.degrees(dif_ang):.3e} graus")

    print("\nTENSÕES FINAIS — NEWTON-RAPHSON")
    print("Barra |V| (p.u.) Ângulo (graus) Mismatch máximo (p.u.)")
    for i, v in enumerate(referencia):
        print(f"{i+1:>5} {abs(v):>10.6f} {np.degrees(np.angle(v)):>14.6f}")
    print(f"Erro final: {erro_maximo_mismatch(referencia, Y):.8e} p.u.")

    print("\nFLUXOS DE POTÊNCIA NAS LINHAS — NEWTON-RAPHSON")
    df = pd.DataFrame(calcular_fluxos_linhas(referencia))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nNota: todos os resultados estão em p.u.; para converter potência em MW/MVAr, multiplique por 100 MVA.")


if __name__ == "__main__":
    main()
