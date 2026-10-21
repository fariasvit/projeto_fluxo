import numpy as np
from modelagem import montar_ybus, montar_tensao_inicial
from algoritmo_newton_raphson import newton_raphson
from fluxos_linhas import calcular_fluxos_linhas
from dados_sistema import LINHAS


def solucao():
    Y = montar_ybus()
    V0 = montar_tensao_inicial([1.0]*5, [0.0]*5, [0.0]*2)
    V, _, _, _, ok = newton_raphson(Y, V0)
    assert ok
    return V


def test_fluxos_cobrem_todas_as_linhas():
    assert len(calcular_fluxos_linhas(solucao())) == len(LINHAS) == 10


def test_fluxos_sao_finitos():
    for linha in calcular_fluxos_linhas(solucao()):
        for chave, valor in linha.items():
            if chave not in ("de", "para"):
                assert np.isfinite(valor)


def test_perdas_ativas_nulas_com_resistencias_nulas():
    for linha in calcular_fluxos_linhas(solucao()):
        assert abs(linha["P_perdas_pu"]) < 1e-8


def test_fluxo_em_ambos_os_sentidos_tem_sentidos_consistentes():
    fluxos = calcular_fluxos_linhas(solucao())
    assert all(l["de"] != l["para"] for l in fluxos)
