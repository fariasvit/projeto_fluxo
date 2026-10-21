"""
Funções de modelagem elétrica:
- impedâncias e admitâncias das linhas;
- montagem da matriz Ybus;
- cálculo de potências injetadas;
- cálculo do mismatch.
"""

import numpy as np

from dados_sistema import (
    N_BARRAS, LINHAS, P_ESP, Q_ESP,
    INDICE_SLACK, INDICES_PV, INDICES_PQ, INDICES_NAO_SLACK,
    V_MOD_ESPEC, ANG_SLACK_GRAUS,
)


def calcular_parametros_linhas():
    """Retorna tabela numérica com Z série e y série de cada linha."""
    parametros = []
    for de, para, r, x, bsh in LINHAS:
        z = complex(r, x)
        y = 1 / z
        parametros.append({
            "de": de,
            "para": para,
            "R_pu": r,
            "X_pu": x,
            "Bsh_pu": bsh,
            "Z_real": z.real,
            "Z_imag": z.imag,
            "y_real": y.real,
            "y_imag": y.imag,
            "y_complexa": y,
        })
    return parametros


def montar_ybus():
    """
    Monta Ybus.
    Convenção adotada: Bsh total da linha dividido igualmente entre as
    duas extremidades, adicionando j*Bsh/2 em cada elemento diagonal.
    """
    Y = np.zeros((N_BARRAS, N_BARRAS), dtype=complex)

    for de, para, r, x, bsh in LINHAS:
        i, j = de - 1, para - 1
        z = complex(r, x)
        y = 1 / z

        Y[i, i] += y + 1j * bsh / 2
        Y[j, j] += y + 1j * bsh / 2
        Y[i, j] -= y
        Y[j, i] -= y

    return Y


def calcular_potencias(V, Ybus):
    """
    Calcula injeções líquidas por barra usando S_i = V_i * conj(I_i),
    com I = Ybus @ V.
    Retorna P_calc e Q_calc em p.u.
    """
    correntes = Ybus @ V
    potencias = V * np.conj(correntes)
    return potencias.real, potencias.imag



def calcular_potencias_polar(V, Ybus):
    """
    Calcula P e Q pelas equações de fluxo de potência em coordenadas polares.

    P_i = |V_i| sum_j |V_j| [G_ij cos(delta_i-delta_j)
                            + B_ij sin(delta_i-delta_j)]
    Q_i = |V_i| sum_j |V_j| [G_ij sin(delta_i-delta_j)
                            - B_ij cos(delta_i-delta_j)]

    Esta implementação é uma verificação independente da forma complexa
    S = V * conj(Ybus @ V).
    """
    modulos = np.abs(V)
    angulos = np.angle(V)
    G = Ybus.real
    B = Ybus.imag
    n = len(V)

    P = np.zeros(n, dtype=float)
    Q = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            diferenca = angulos[i] - angulos[j]
            P[i] += modulos[i] * modulos[j] * (
                G[i, j] * np.cos(diferenca)
                + B[i, j] * np.sin(diferenca)
            )
            Q[i] += modulos[i] * modulos[j] * (
                G[i, j] * np.sin(diferenca)
                - B[i, j] * np.cos(diferenca)
            )

    return P, Q

def calcular_mismatch(V, Ybus):
    """
    Vetor de mismatch para formulação polar:
    [Pesp - Pcalc nas barras não Slack,
     Qesp - Qcalc nas barras PQ].
    """
    P_calc, Q_calc = calcular_potencias(V, Ybus)
    delta_P = P_ESP[INDICES_NAO_SLACK] - P_calc[INDICES_NAO_SLACK]
    delta_Q = Q_ESP[INDICES_PQ] - Q_calc[INDICES_PQ]
    return np.concatenate((delta_P, delta_Q))


def erro_maximo_mismatch(V, Ybus):
    """Maior módulo entre todos os mismatches incluídos no vetor."""
    return float(np.max(np.abs(calcular_mismatch(V, Ybus))))


def montar_tensao_inicial(pq_modulos, pq_angulos_graus, pv_angulos_graus):
    """
    Monta o vetor de tensão inicial complexo.
    Os módulos Slack/PV especificados permanecem fixos; só os valores
    iniciais das variáveis desconhecidas são fornecidos pelo usuário.
    """
    from dados_sistema import INDICES_PQ, INDICES_PV

    if len(pq_modulos) != len(INDICES_PQ):
        raise ValueError("Informe exatamente cinco módulos iniciais para as barras PQ.")
    if len(pq_angulos_graus) != len(INDICES_PQ):
        raise ValueError("Informe exatamente cinco ângulos iniciais para as barras PQ.")
    if len(pv_angulos_graus) != len(INDICES_PV):
        raise ValueError("Informe exatamente dois ângulos iniciais para as barras PV.")

    modulos = np.ones(N_BARRAS, dtype=float)
    angulos = np.zeros(N_BARRAS, dtype=float)

    for indice, modulo in V_MOD_ESPEC.items():
        modulos[indice] = modulo

    modulos[INDICE_SLACK] = V_MOD_ESPEC[INDICE_SLACK]
    angulos[INDICE_SLACK] = ANG_SLACK_GRAUS

    for k, i in enumerate(INDICES_PQ):
        if pq_modulos[k] <= 0:
            raise ValueError("Módulos iniciais de tensão devem ser positivos.")
        modulos[i] = float(pq_modulos[k])
        angulos[i] = float(pq_angulos_graus[k])

    for k, i in enumerate(INDICES_PV):
        angulos[i] = float(pv_angulos_graus[k])

    return modulos * np.exp(1j * np.radians(angulos))


def validar_ybus(Ybus, tolerancia=1e-10):
    """Retorna verificações estruturais básicas da Ybus."""
    simetrica = np.allclose(Ybus, Ybus.T, atol=tolerancia)
    parte_real_nula = np.allclose(Ybus.real, 0.0, atol=tolerancia)
    dimensao_correta = Ybus.shape == (N_BARRAS, N_BARRAS)
    finita = np.all(np.isfinite(Ybus))
    return {
        "dimensao_8x8": dimensao_correta,
        "simetrica": simetrica,
        "parte_real_nula": parte_real_nula,
        "valores_finitos": finita,
    }
