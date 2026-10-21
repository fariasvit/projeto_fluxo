"""Implementação do método Newton-Raphson para fluxo de potência em coordenadas polares."""
import numpy as np
from dados_sistema import (
    N_BARRAS, P_ESP, Q_ESP, INDICE_SLACK, INDICES_PV, INDICES_PQ,
    INDICES_NAO_SLACK, V_MOD_ESPEC, ANG_SLACK_GRAUS,
)
from modelagem import calcular_potencias, calcular_mismatch, erro_maximo_mismatch


def _montar_jacobiana(V, Ybus):
    P, Q = calcular_potencias(V, Ybus)
    G, B = Ybus.real, Ybus.imag
    Vm, ang = np.abs(V), np.angle(V)
    nP, nQ = len(INDICES_NAO_SLACK), len(INDICES_PQ)
    J = np.zeros((nP + nQ, nP + nQ), dtype=float)

    # H = dP/d(delta)
    for r, i in enumerate(INDICES_NAO_SLACK):
        for c, j in enumerate(INDICES_NAO_SLACK):
            if i == j:
                J[r, c] = -Q[i] - B[i, i] * Vm[i] ** 2
            else:
                d = ang[i] - ang[j]
                J[r, c] = Vm[i] * Vm[j] * (G[i, j] * np.sin(d) - B[i, j] * np.cos(d))

    # N = dP/d|V| (PQ)
    for r, i in enumerate(INDICES_NAO_SLACK):
        for k, j in enumerate(INDICES_PQ):
            c = nP + k
            if i == j:
                J[r, c] = P[i] / Vm[i] + G[i, i] * Vm[i]
            else:
                d = ang[i] - ang[j]
                J[r, c] = Vm[i] * (G[i, j] * np.cos(d) + B[i, j] * np.sin(d))

    # M = dQ/d(delta)
    for k, i in enumerate(INDICES_PQ):
        r = nP + k
        for c, j in enumerate(INDICES_NAO_SLACK):
            if i == j:
                J[r, c] = P[i] - G[i, i] * Vm[i] ** 2
            else:
                d = ang[i] - ang[j]
                J[r, c] = -Vm[i] * Vm[j] * (G[i, j] * np.cos(d) + B[i, j] * np.sin(d))

    # L = dQ/d|V| (PQ)
    for k, i in enumerate(INDICES_PQ):
        r = nP + k
        for c, j in enumerate(INDICES_PQ):
            col = nP + c
            if i == j:
                J[r, col] = Q[i] / Vm[i] - B[i, i] * Vm[i]
            else:
                d = ang[i] - ang[j]
                J[r, col] = Vm[i] * (G[i, j] * np.sin(d) - B[i, j] * np.cos(d))
    return J


def _atualizar_tensoes(V, correcao):
    V_novo = V.copy()
    nP = len(INDICES_NAO_SLACK)
    ang, mod = np.angle(V_novo), np.abs(V_novo)
    for k, i in enumerate(INDICES_NAO_SLACK):
        ang[i] += correcao[k]
    for k, i in enumerate(INDICES_PQ):
        mod[i] += correcao[nP + k]
        if mod[i] <= 0:
            raise FloatingPointError(f"Módulo não positivo na barra {i + 1}.")
    for i, valor in V_MOD_ESPEC.items():
        mod[i] = valor
    ang[INDICE_SLACK] = np.radians(ANG_SLACK_GRAUS)
    return mod * np.exp(1j * ang)


def newton_raphson(Ybus, V_inicial, tolerancia=1e-6, max_iter=100):
    if tolerancia <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    if max_iter < 1:
        raise ValueError("max_iter deve ser pelo menos 1.")
    if len(V_inicial) != N_BARRAS:
        raise ValueError(f"V_inicial deve conter {N_BARRAS} tensões.")

    V = np.asarray(V_inicial, dtype=complex).copy()
    mod, ang = np.abs(V), np.angle(V)
    mod[INDICE_SLACK] = V_MOD_ESPEC[INDICE_SLACK]
    ang[INDICE_SLACK] = np.radians(ANG_SLACK_GRAUS)
    for i in INDICES_PV:
        mod[i] = V_MOD_ESPEC[i]
    V = mod * np.exp(1j * ang)

    historico = []
    erro = erro_maximo_mismatch(V, Ybus)
    if erro < tolerancia:
        return V, 0, erro, historico, True

    for iteracao in range(1, max_iter + 1):
        mismatch = calcular_mismatch(V, Ybus)
        J = _montar_jacobiana(V, Ybus)
        try:
            correcao = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("A Jacobiana do Newton-Raphson é singular.") from exc
        V = _atualizar_tensoes(V, correcao)
        erro = erro_maximo_mismatch(V, Ybus)
        historico.append(erro)
        if erro < tolerancia:
            return V, iteracao, erro, historico, True
    return V, max_iter, erro, historico, False
