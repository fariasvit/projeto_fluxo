import numpy as np
import pytest

from dados_sistema import INDICE_SLACK, INDICES_PV, V_MOD_ESPEC
from modelagem import montar_ybus, montar_tensao_inicial, erro_maximo_mismatch
from algoritmo_seidel import gauss_seidel


def chute_plano():
    return montar_tensao_inicial(
        pq_modulos=[1.0] * 5,
        pq_angulos_graus=[0.0] * 5,
        pv_angulos_graus=[0.0] * 2,
    )


def test_seidel_converge_com_chute_plano():
    Y = montar_ybus()
    V, n, erro, historico, convergiu = gauss_seidel(Y, chute_plano())
    assert convergiu
    assert n == len(historico)
    assert n > 0
    assert len(V) == 8
    assert np.isclose(erro, erro_maximo_mismatch(V, Y))
    assert erro < 1e-6


def test_seidel_preserva_tensoes_slack_e_modulos_pv():
    Y = montar_ybus()
    V, _, _, _, convergiu = gauss_seidel(Y, chute_plano())
    assert convergiu
    assert np.isclose(abs(V[INDICE_SLACK]), V_MOD_ESPEC[INDICE_SLACK])
    for i in INDICES_PV:
        assert np.isclose(abs(V[i]), V_MOD_ESPEC[i])


def test_seidel_respeita_limite_de_iteracoes():
    Y = montar_ybus()
    _, n, _, hist, convergiu = gauss_seidel(Y, chute_plano(), tolerancia=1e-15, max_iter=1)
    assert n == 1
    assert len(hist) == 1
    assert not convergiu


def test_seidel_rejeita_tolerancia_invalida():
    with pytest.raises(ValueError):
        gauss_seidel(montar_ybus(), chute_plano(), tolerancia=0)
