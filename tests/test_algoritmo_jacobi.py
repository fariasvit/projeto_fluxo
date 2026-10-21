import numpy as np
import pytest

from dados_sistema import INDICE_SLACK, INDICES_PV, V_MOD_ESPEC
from modelagem import montar_ybus, montar_tensao_inicial, erro_maximo_mismatch
from algoritmo_jacobi import gauss_jacobi


def chute_plano():
    return montar_tensao_inicial(
        pq_modulos=[1.0] * 5,
        pq_angulos_graus=[0.0] * 5,
        pv_angulos_graus=[0.0] * 2,
    )


def test_jacobi_retorna_resultado_e_historico_consistentes():
    Y = montar_ybus()
    V, n, erro, historico, convergiu = gauss_jacobi(Y, chute_plano())
    assert len(V) == 8
    assert n == len(historico)
    assert np.isclose(erro, erro_maximo_mismatch(V, Y))
    assert len(historico) > 0
    assert convergiu
    assert erro < 1e-6


def test_jacobi_preserva_tensao_slack_e_modulos_pv():
    Y = montar_ybus()
    V, _, _, _, convergiu = gauss_jacobi(Y, chute_plano())
    assert convergiu
    assert np.isclose(abs(V[INDICE_SLACK]), V_MOD_ESPEC[INDICE_SLACK])
    for i in INDICES_PV:
        assert np.isclose(abs(V[i]), V_MOD_ESPEC[i])


def test_jacobi_respeita_limite_de_iteracoes():
    Y = montar_ybus()
    V, n, erro, historico, convergiu = gauss_jacobi(
        Y, chute_plano(), tolerancia=1e-15, max_iter=1
    )
    assert n == 1
    assert len(historico) == 1
    assert not convergiu


def test_jacobi_rejeita_tolerancia_invalida():
    with pytest.raises(ValueError):
        gauss_jacobi(montar_ybus(), chute_plano(), tolerancia=0)
