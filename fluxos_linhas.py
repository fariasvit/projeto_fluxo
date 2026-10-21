"""Cálculo dos fluxos de potência nas extremidades de cada linha."""
import numpy as np
from dados_sistema import LINHAS


def calcular_fluxos_linhas(V):
    """
    Calcula potência complexa enviada de i para j e de j para i.

    O modelo pi considera a admitância série y=1/(R+jX) e a susceptância
    total Bsh dividida em j*Bsh/2 em cada extremidade.
    Retorna uma lista de dicionários, valores em p.u.
    """
    resultados = []
    for de, para, r, x, bsh in LINHAS:
        i, j = de - 1, para - 1
        y = 1 / complex(r, x)
        corrente_ij = (V[i] - V[j]) * y + V[i] * (1j * bsh / 2)
        corrente_ji = (V[j] - V[i]) * y + V[j] * (1j * bsh / 2)
        S_ij = V[i] * np.conj(corrente_ij)
        S_ji = V[j] * np.conj(corrente_ji)
        soma = S_ij + S_ji
        resultados.append({
            "de": de,
            "para": para,
            "P_de_para_pu": float(S_ij.real),
            "Q_de_para_pu": float(S_ij.imag),
            "P_para_de_pu": float(S_ji.real),
            "Q_para_de_pu": float(S_ji.imag),
            "P_perdas_pu": float(soma.real),
            "Q_soma_pu": float(soma.imag),
        })
    return resultados
