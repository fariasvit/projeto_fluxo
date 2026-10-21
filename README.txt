PROJETO DE FLUXO DE POTÊNCIA — SEP I
ETAPA 1: DADOS FIXOS E MODELAGEM ELÉTRICA

Arquivos:
- dados_sistema.py: dados fixos do enunciado.
- modelagem.py: impedâncias, admitâncias, Ybus, potências, mismatch e tensão inicial.
- verificar_modelagem.py: executa verificações e imprime os resultados.
- requirements.txt: bibliotecas necessárias.

Como executar:
1. Instale Python 3.10 ou superior.
2. Instale NumPy: pip install -r requirements.txt
3. Execute: python verificar_modelagem.py

Importante:
- Os dados de barras e linhas são tratados como fixos.
- A susceptância Bsh foi dividida igualmente entre as duas extremidades de cada linha, conforme a convenção adotada na implementação. Confirmar essa interpretação com o enunciado do professor.
- Esta etapa não executa ainda Gauss-Jacobi, Gauss-Seidel ou Newton-Raphson.


ETAPA 3 — Potências e mismatch:
Execute todos os testes com: python -m pytest -q
O relatório está em RELATORIO_VERIFICACAO_POTENCIAS_MISMATCH.txt.

ETAPA 4 — Método de Gauss-Jacobi:
- Implementação em algoritmo_jacobi.py.
- Testes em tests/test_algoritmo_jacobi.py.
- Relatório em RELATORIO_GAUSS_JACOBI.txt.
- Para executar todos os testes: python -m pytest -q
- Para visualizar um exemplo numérico, execute: python executar_jacobi.py


ETAPA 5 — Gauss-Seidel:
Execute o método com: python executar_seidel.py
Execute todos os testes com: python -m pytest -q
Consulte RELATORIO_GAUSS_SEIDEL.txt para a descrição do algoritmo.

ETAPA 7 — INTERFACE STREAMLIT:
Instale as dependências com: python -m pip install -r requirements.txt
Execute a interface com: python -m streamlit run app.py

ETAPA 8 — VALIDAÇÃO DA SOLUÇÃO E FLUXOS:
Execute: python validar_solucao.py
Relatório: RELATORIO_ETAPA8_VALIDACAO_SOLUCAO_E_FLUXOS.txt
\nETAPA 9 — Layout sem barra lateral: configuração dos chutes iniciais no corpo principal. Consulte RELATORIO_ETAPA9_LAYOUT_INTERFACE.txt.\n
ETAPA 10 — INTERFACE DIDÁTICA:
Execute: python -m streamlit run app.py
Relatório: RELATORIO_ETAPA10_INTERFACE_DIDATICA.txt

ETAPA 11 — Tabela de barras conforme o enunciado: RELATORIO_ETAPA11_TABELA_DADOS_ORIGINAIS.txt.

ETAPA 12 — Formatação de campos ausentes e fórmulas de potência líquida: RELATORIO_ETAPA12_FORMATACAO_E_FORMULAS.txt.


ETAPA 15 — INTERFACE SIMPLIFICADA
A interface foi reorganizada em três abas:
1. Sistema elétrico: diagrama, dados das barras, linhas e matriz Ybus.
2. Configuração e métodos: mismatch inicial, explicações dos algoritmos e históricos de convergência.
3. Resultados e análise: resumo, gráficos, tensões, comparação com Newton-Raphson e fluxos nas linhas.

Os controles de chute inicial, tolerância, limites de iteração e seleção de métodos ficam no painel superior para acesso direto antes de executar o cálculo.

Execução:
python -m pip install -r requirements.txt
python -m streamlit run app.py


ETAPA 16 — INTERFACE REFINADA
- Configuração e chutes iniciais movidos para a barra lateral.
- Tolerâncias disponíveis: 1e-04, 1e-06 e 1e-08 p.u.
- Campos de máximo de iterações removidos da interface; limites internos fixos:
  5000 para Gauss-Jacobi/Gauss-Seidel e 100 para Newton-Raphson.
- Três cenários predefinidos: Flat Start, Chute alternativo A e Chute alternativo B.
  A opção Personalizado também está disponível para experimentos manuais.
- Mismatch inicial apresentado com explicação, número decimal legível e traços
  para grandezas não aplicáveis.
- Resultados reorganizados em resumo de convergência e abas internas para tensões,
  convergência, fluxos e Ybus.
