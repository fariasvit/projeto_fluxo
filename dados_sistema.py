"""
Dados fixos do sistema elétrico conforme o enunciado do projeto.

IMPORTANTE:
Os dados elétricos não devem ser alterados para os experimentos de chute inicial.
"""

import numpy as np

N_BARRAS = 8
BASE_MVA = 100.0

# Cada elemento: (barra_de, barra_para, R_pu, X_pu, Bsh_pu)
# A numeração das barras segue o enunciado: 1 a 8.
LINHAS = [
    (1, 2, 0.0, 0.1730, 0.00865),
    (1, 8, 0.0, 0.1584, 0.00792),
    (2, 3, 0.0, 0.1248, 0.00624),
    (2, 4, 0.0, 0.0720, 0.00360),
    (3, 4, 0.0, 0.1200, 0.00600),
    (3, 5, 0.0, 0.0816, 0.00408),
    (4, 5, 0.0, 0.1152, 0.00576),
    (5, 6, 0.0, 0.1200, 0.00600),
    (6, 7, 0.0, 0.1056, 0.00528),
    (7, 8, 0.0, 0.0864, 0.00432),
]

# Potências líquidas especificadas: Pesp = Pg - Pd; Qesp = Qg - Qd (p.u.)
P_ESP = np.array([0.0, 0.20, -0.50, -0.60, -0.70, -0.80, 0.30, -0.40])
Q_ESP = np.array([0.0, 0.0, -0.20, -0.30, -0.40, -0.50, 0.0, -0.20])

TIPOS_BARRA = ["Slack", "PV", "PQ", "PQ", "PQ", "PQ", "PV", "PQ"]

# Índices Python: barra 1 -> índice 0, barra 2 -> índice 1, etc.
INDICE_SLACK = 0
INDICES_PV = [1, 6]                # Barras 2 e 7
INDICES_PQ = [2, 3, 4, 5, 7]      # Barras 3, 4, 5, 6 e 8
INDICES_NAO_SLACK = [1, 2, 3, 4, 5, 6, 7]

# Módulos especificados para Slack e PV (p.u.)
V_MOD_ESPEC = {0: 1.05, 1: 1.05, 6: 1.04}
ANG_SLACK_GRAUS = 0.0


# Dados da Tabela 2 do enunciado, separados em geração e carga.
# np.nan representa grandeza não especificada, e não zero.
P_GERACAO = np.array([np.nan, 0.20, 0.0, 0.0, 0.0, 0.0, 0.30, 0.0])
Q_GERACAO = np.array([np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0])
P_CARGA = np.array([0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.0, 0.4])
Q_CARGA = np.array([0.0, 0.0, 0.2, 0.3, 0.4, 0.5, 0.0, 0.2])
V_ESP_TABELA = {0: 1.05, 1: 1.05, 6: 1.04}
ANG_ESP_TABELA = {0: 0.0}
