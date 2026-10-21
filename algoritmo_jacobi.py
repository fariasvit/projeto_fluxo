"""Implementação isolada do método de Gauss-Jacobi para fluxo de potência."""
import numpy as np

from dados_sistema import (
    N_BARRAS, P_ESP, Q_ESP, INDICE_SLACK, INDICES_PV,
    INDICES_PQ, INDICES_NAO_SLACK, V_MOD_ESPEC, ANG_SLACK_GRAUS,
)
from modelagem import calcular_potencias, erro_maximo_mismatch


def gauss_jacobi(Ybus, V_inicial, tolerancia=1e-6, max_iter=10000):
    """Retorna (V, iteracoes, erro, historico, convergiu).

    Todos os cálculos de atualização usam exclusivamente tensões da iteração
    anterior, característica central do método de Gauss-Jacobi.
    """
    if tolerancia <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    if max_iter < 1:
        raise ValueError("max_iter deve ser pelo menos 1.")
    if len(V_inicial) != N_BARRAS:
        raise ValueError(f"V_inicial deve conter {N_BARRAS} tensões.")

    V = np.asarray(V_inicial, dtype=complex).copy()
    V[INDICE_SLACK] = V_MOD_ESPEC[INDICE_SLACK] * np.exp(1j * np.radians(ANG_SLACK_GRAUS))
    historico = []

    for iteracao in range(1, max_iter + 1):
        V_ant = V.copy()
        P_ant, Q_ant = calcular_potencias(V_ant, Ybus)
        V_novo = V_ant.copy()

        for i in range(N_BARRAS):
            if i == INDICE_SLACK:
                continue

            q_usado = Q_ant[i] if i in INDICES_PV else Q_ESP[i]
            soma_outros = Ybus[i, :] @ V_ant - Ybus[i, i] * V_ant[i]
            V_calculado = (
                (P_ESP[i] - 1j * q_usado) / np.conj(V_ant[i]) - soma_outros
            ) / Ybus[i, i]

            if i in INDICES_PV:
                V_calculado = V_MOD_ESPEC[i] * np.exp(1j * np.angle(V_calculado))
            V_novo[i] = V_calculado

        V_novo[INDICE_SLACK] = V_MOD_ESPEC[INDICE_SLACK] * np.exp(1j * np.radians(ANG_SLACK_GRAUS))
        V = V_novo
        erro = erro_maximo_mismatch(V, Ybus)
        historico.append(erro)

        if erro < tolerancia:
            return V, iteracao, erro, historico, True

    return V, max_iter, erro_maximo_mismatch(V, Ybus), historico, False
