"""Testes automáticos da modelagem elétrica (pytest)."""

import numpy as np

from dados_sistema import N_BARRAS, LINHAS, INDICES_PQ, INDICES_PV, V_MOD_ESPEC
from modelagem import montar_ybus, montar_tensao_inicial, calcular_potencias, calcular_mismatch


def test_dimensao_ybus():
    Y = montar_ybus()
    assert Y.shape == (N_BARRAS, N_BARRAS)


def test_ybus_simetrica():
    Y = montar_ybus()
    assert np.allclose(Y, Y.T, atol=1e-12)


def test_diagonais_correspondem_a_soma_das_admitancias():
    Y = montar_ybus()
    esperado = np.zeros(N_BARRAS, dtype=complex)
    for de, para, r, x, bsh in LINHAS:
        i, j = de - 1, para - 1
        y = 1 / complex(r, x)
        esperado[i] += y + 1j * bsh / 2
        esperado[j] += y + 1j * bsh / 2
    assert np.allclose(np.diag(Y), esperado, atol=1e-12)


def test_elementos_fora_diagonal():
    Y = montar_ybus()
    esperado = np.zeros_like(Y)
    for de, para, r, x, bsh in LINHAS:
        i, j = de - 1, para - 1
        y = 1 / complex(r, x)
        esperado[i, j] -= y
        esperado[j, i] -= y
    assert np.allclose(Y - np.diag(np.diag(Y)), esperado, atol=1e-12)


def test_tensao_inicial_respeita_modulos_especificados():
    V = montar_tensao_inicial(
        pq_modulos=[1.0] * len(INDICES_PQ),
        pq_angulos_graus=[0.0] * len(INDICES_PQ),
        pv_angulos_graus=[-2.0, -6.0],
    )
    for i, vm in V_MOD_ESPEC.items():
        assert np.isclose(abs(V[i]), vm)
    assert np.isclose(np.angle(V[0]), 0.0)


def test_potencia_ativa_zero_com_tensoes_em_fase():
    Y = montar_ybus()
    V = montar_tensao_inicial(
        pq_modulos=[1.0] * len(INDICES_PQ),
        pq_angulos_graus=[0.0] * len(INDICES_PQ),
        pv_angulos_graus=[0.0] * len(INDICES_PV),
    )
    P, Q = calcular_potencias(V, Y)
    # Como R=0 e todas as tensões estão em fase, P deve ser zero
    # numericamente, mesmo com módulos diferentes nas barras.
    assert np.allclose(P, 0.0, atol=1e-10)


def test_mismatch_tem_12_equacoes():
    Y = montar_ybus()
    V = montar_tensao_inicial(
        pq_modulos=[1.0] * len(INDICES_PQ),
        pq_angulos_graus=[0.0] * len(INDICES_PQ),
        pv_angulos_graus=[0.0] * len(INDICES_PV),
    )
    mismatch = calcular_mismatch(V, Y)
    assert mismatch.shape == (12,)
    assert np.all(np.isfinite(mismatch))
