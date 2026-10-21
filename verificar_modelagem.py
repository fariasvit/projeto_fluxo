"""Rotina de conferência da modelagem elétrica."""

import numpy as np

from dados_sistema import N_BARRAS, LINHAS, BASE_MVA
from modelagem import (
    calcular_parametros_linhas,
    montar_ybus,
    validar_ybus,
    calcular_potencias,
    montar_tensao_inicial,
)

np.set_printoptions(precision=6, suppress=True)

print("=" * 72)
print("1. PARÂMETROS DAS LINHAS")
print("=" * 72)
for item in calcular_parametros_linhas():
    print(
        f"{item['de']}-{item['para']}: "
        f"Z = {item['Z_real']:.6f} {item['Z_imag']:+.6f}j p.u.; "
        f"y = {item['y_real']:.6f} {item['y_imag']:+.6f}j p.u."
    )

Ybus = montar_ybus()

print("\n" + "=" * 72)
print("2. MATRIZ YBUS (p.u.)")
print("=" * 72)
print(Ybus)

print("\n" + "=" * 72)
print("3. VERIFICAÇÕES DA YBUS")
print("=" * 72)
for nome, ok in validar_ybus(Ybus).items():
    print(f"{nome}: {'OK' if ok else 'FALHOU'}")

print("\n" + "=" * 72)
print("4. TESTE DE POTÊNCIAS COM UMA TENSÃO DE TESTE")
print("=" * 72)
# Esse vetor é apenas um teste da função de cálculo, não a solução do fluxo.
V_teste = montar_tensao_inicial(
    pq_modulos=[1.0] * 5,
    pq_angulos_graus=[0.0] * 5,
    pv_angulos_graus=[0.0] * 2,
)
P, Q = calcular_potencias(V_teste, Ybus)
for i in range(N_BARRAS):
    print(f"Barra {i+1}: Pcalc={P[i]: .6f} p.u.; Qcalc={Q[i]: .6f} p.u.")

print(f"\nBase de potência: {BASE_MVA:.1f} MVA")
print("Observação: o teste de potência usa uma tensão de teste e não representa a solução final.")
