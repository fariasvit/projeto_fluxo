"""Implementação do método de Gauss-Seidel para fluxo de potência."""
import numpy as np

from dados_sistema import (
    N_BARRAS, P_ESP, Q_ESP, INDICE_SLACK, INDICES_PV,
    V_MOD_ESPEC, ANG_SLACK_GRAUS,
)
from modelagem import calcular_potencias, erro_maximo_mismatch


def gauss_seidel(Ybus, V_inicial, tolerancia=1e-6, max_iter=10000):
    """Retorna (V, iteracoes, erro, historico, convergiu).

    Ao contrário de Jacobi, atualiza cada barra imediatamente; as barras
    seguintes usam os valores mais recentes já calculados na iteração.
    Para barras PV, estima Q a partir do estado atual e repõe o módulo
    especificado após a atualização do fasor.
    """
    if tolerancia <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    if max_iter < 1:
        raise ValueError("max_iter deve ser pelo menos 1.")
    if len(V_inicial) != N_BARRAS:
        raise ValueError(f"V_inicial deve conter {N_BARRAS} tensões.")

    V = np.asarray(V_inicial, dtype=complex).copy()
    V[INDICE_SLACK] = V_MOD_ESPEC[INDICE_SLACK] * np.exp(
        1j * np.radians(ANG_SLACK_GRAUS)
    )
    historico = []

    for iteracao in range(1, max_iter + 1):
        for i in range(N_BARRAS):
            if i == INDICE_SLACK:
                continue

            # Usa o estado mais recente de V para estimar Q na barra PV.
            if i in INDICES_PV:
                _, Q_atual = calcular_potencias(V, Ybus)
                q_usado = Q_atual[i]
            else:
                q_usado = Q_ESP[i]

            soma_outros = Ybus[i, :] @ V - Ybus[i, i] * V[i]
            if abs(V[i]) < 1e-12:
                raise FloatingPointError(
                    f"Módulo de tensão muito próximo de zero na barra {i + 1}."
                )
            V_calculado = (
                (P_ESP[i] - 1j * q_usado) / np.conj(V[i]) - soma_outros
            ) / Ybus[i, i]

            if i in INDICES_PV:
                V_calculado = V_MOD_ESPEC[i] * np.exp(1j * np.angle(V_calculado))
            V[i] = V_calculado

        V[INDICE_SLACK] = V_MOD_ESPEC[INDICE_SLACK] * np.exp(
            1j * np.radians(ANG_SLACK_GRAUS)
        )
        erro = erro_maximo_mismatch(V, Ybus)
        historico.append(erro)
        if erro < tolerancia:
            return V, iteracao, erro, historico, True

    return V, max_iter, erro_maximo_mismatch(V, Ybus), historico, False
