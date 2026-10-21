import numpy as np

from dados_sistema import (
    P_ESP, Q_ESP, INDICES_NAO_SLACK, INDICES_PQ, INDICE_SLACK,
    INDICES_PV, V_MOD_ESPEC, ANG_SLACK_GRAUS
)
from modelagem import (
    montar_ybus, montar_tensao_inicial, calcular_potencias,
    calcular_potencias_polar, calcular_mismatch, erro_maximo_mismatch
)


def tensao_inicial_plana():
    return montar_tensao_inicial(
        pq_modulos=[1.0] * 5,
        pq_angulos_graus=[0.0] * 5,
        pv_angulos_graus=[0.0] * 2,
    )


def test_potencias_complexas_concordam_com_equacoes_polares():
    Y = montar_ybus()
    V = montar_tensao_inicial(
        pq_modulos=[0.98, 0.97, 0.96, 0.95, 0.99],
        pq_angulos_graus=[-3, -5, -7, -9, -4],
        pv_angulos_graus=[-2, -6],
    )
    P1, Q1 = calcular_potencias(V, Y)
    P2, Q2 = calcular_potencias_polar(V, Y)
    assert np.allclose(P1, P2, atol=1e-10)
    assert np.allclose(Q1, Q2, atol=1e-10)


def test_mismatch_tem_12_equacoes_na_ordem_documentada():
    Y = montar_ybus()
    V = tensao_inicial_plana()
    P, Q = calcular_potencias(V, Y)
    esperado = np.concatenate((
        P_ESP[INDICES_NAO_SLACK] - P[INDICES_NAO_SLACK],
        Q_ESP[INDICES_PQ] - Q[INDICES_PQ],
    ))
    mismatch = calcular_mismatch(V, Y)
    assert mismatch.shape == (12,)
    assert np.allclose(mismatch, esperado)


def test_mismatch_nao_inclui_potencia_da_slack_nem_q_das_pv():
    Y = montar_ybus()
    V = tensao_inicial_plana()
    mismatch = calcular_mismatch(V, Y)
    # 7 mismatches de P (barras 2,3,4,5,6,7,8), seguidos por
    # 5 mismatches de Q (barras 3,4,5,6,8).
    assert mismatch[:7].shape == (7,)
    assert mismatch[7:].shape == (5,)
    assert len(mismatch) == 12


def test_erro_maximo_e_maximo_absoluto_do_mismatch():
    Y = montar_ybus()
    V = tensao_inicial_plana()
    mismatch = calcular_mismatch(V, Y)
    assert np.isclose(erro_maximo_mismatch(V, Y), np.max(np.abs(mismatch)))


def test_potencia_ativa_calculada_e_zero_com_angulos_iguais_e_r_zero():
    Y = montar_ybus()
    V = tensao_inicial_plana()
    P, _ = calcular_potencias(V, Y)
    assert np.allclose(P, 0.0, atol=1e-10)
