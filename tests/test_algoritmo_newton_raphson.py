import numpy as np
import pytest
from modelagem import montar_ybus, montar_tensao_inicial, erro_maximo_mismatch
from algoritmo_newton_raphson import newton_raphson

def chute_plano():
    return montar_tensao_inicial([1.0]*5, [0.0]*5, [0.0]*2)

def test_newton_raphson_converge_com_chute_plano():
    Y = montar_ybus()
    V, n, erro, hist, ok = newton_raphson(Y, chute_plano())
    assert ok and erro < 1e-6 and n <= 10
    assert len(hist) == n
    assert np.isclose(erro, erro_maximo_mismatch(V, Y))

def test_newton_preserva_modulos_slack_e_pv():
    from dados_sistema import V_MOD_ESPEC, INDICES_PV, INDICE_SLACK
    V, *_ = newton_raphson(montar_ybus(), chute_plano())
    assert np.isclose(abs(V[INDICE_SLACK]), V_MOD_ESPEC[INDICE_SLACK])
    for i in INDICES_PV:
        assert np.isclose(abs(V[i]), V_MOD_ESPEC[i])

def test_newton_rejeita_tolerancia_invalida():
    with pytest.raises(ValueError):
        newton_raphson(montar_ybus(), chute_plano(), tolerancia=0)

def test_newton_rejeita_max_iter_invalido():
    with pytest.raises(ValueError):
        newton_raphson(montar_ybus(), chute_plano(), max_iter=0)

def test_newton_rejeita_tensao_inicial_tamanho_errado():
    with pytest.raises(ValueError):
        newton_raphson(montar_ybus(), [1+0j]*7)
