import numpy as np
import pandas as pd
import streamlit as st

from dados_sistema import (
    N_BARRAS, BASE_MVA, LINHAS, P_ESP, Q_ESP, TIPOS_BARRA,
    INDICES_PQ, INDICES_PV, INDICES_NAO_SLACK, V_MOD_ESPEC,
    P_GERACAO, Q_GERACAO, P_CARGA, Q_CARGA, V_ESP_TABELA, ANG_ESP_TABELA,
)
from modelagem import (
    montar_ybus, montar_tensao_inicial, calcular_potencias,
    calcular_mismatch, erro_maximo_mismatch,
)
from algoritmo_jacobi import gauss_jacobi
from algoritmo_seidel import gauss_seidel
from algoritmo_newton_raphson import newton_raphson
from fluxos_linhas import calcular_fluxos_linhas

st.set_page_config(page_title="Fluxo de Potência", page_icon="⚡", layout="wide")
st.title("Projeto de Fluxo de Potência - 2026.2")
st.caption("Discentes: Alvaro Oliveira, João Pedro Almeida, Larissa Vianna, Maria Eduarda Neumayer e Vitor Farias")
##st.info("1. Conheça o sistema. 2. Configure os chutes na barra lateral. 3. Compare os resultados e a convergência.")

pq_barras = [3, 4, 5, 6, 8]
CENARIOS_CHUTES = {
    "A": {"mod": [1.00, 1.00, 1.00, 1.00, 1.00], "pq": [0, 0, 0, 0, 0], "pv": [0, 0]},
    "B": {"mod": [0.98, 0.97, 0.96, 0.95, 0.98], "pq": [-3, -5, -7, -9, -4], "pv": [-2, -6]},
    "C": {"mod": [1.02, 1.02, 1.02, 1.02, 1.02], "pq": [5, 5, 5, 5, 5], "pv": [5, 5]},
    "Personalizado": {"mod": [1.00, 1.00, 1.00, 1.00, 1.00], "pq": [0, 0, 0, 0, 0], "pv": [0, 0]},
}

def aplicar_cenario():
    nome = st.session_state.get("cenario_chute", "A")
    if nome != "Personalizado":
        preset = CENARIOS_CHUTES[nome]
        for k, b in enumerate(pq_barras):
            st.session_state[f"mod_{b}"] = float(preset["mod"][k])
            st.session_state[f"ang_pq_{b}"] = float(preset["pq"][k])
        for k, b in enumerate([2, 7]):
            st.session_state[f"ang_pv_{b}"] = float(preset["pv"][k])

if "cenario_chute" not in st.session_state:
    st.session_state["cenario_chute"] = "A"
for k, b in enumerate(pq_barras):
    st.session_state.setdefault(f"mod_{b}", 1.00)
    st.session_state.setdefault(f"ang_pq_{b}", 0.0)
for b in [2, 7]:
    st.session_state.setdefault(f"ang_pv_{b}", 0.0)

with st.sidebar:
    st.header("Configurar simulação")
    ##st.caption("Os dados elétricos do enunciado são fixos. Altere apenas os chutes iniciais e os parâmetros numéricos.")
    st.selectbox(
        "Cenário de chute inicial",
        ["A", "B", "C", "Personalizado"],
        key="cenario_chute",
        on_change=aplicar_cenario,
    )
    with st.expander("Editar chutes iniciais", expanded=(st.session_state.get("cenario_chute") == "Personalizado")):
        st.markdown("**Módulos iniciais das barras PQ (p.u.)**")
        pq_modulos = [
            st.number_input(f"Barra {b} · |V| inicial", min_value=0.10, max_value=1.50,
                            step=0.01, format="%.3f", key=f"mod_{b}")
            for b in pq_barras
        ]
        st.markdown("**Ângulos iniciais das barras PQ (°)**")
        pq_angulos = [
            st.number_input(f"Barra {b} · ângulo inicial", min_value=-180.0, max_value=180.0,
                            step=1.0, key=f"ang_pq_{b}")
            for b in pq_barras
        ]
        st.markdown("**Ângulos iniciais das barras PV (°)**")
        pv_angulos = [
            st.number_input(f"Barra {b} · ângulo inicial", min_value=-180.0, max_value=180.0,
                            step=1.0, key=f"ang_pv_{b}")
            for b in [2, 7]
        ]
        st.caption("Os módulos das barras Slack e PV permanecem fixos.")
    tolerancia = st.selectbox(
        "Tolerância do mismatch (p.u.)",
        options=[1e-4, 1e-6, 1e-8],
        index=1,
        format_func=lambda x: f"{x:.0e}",
    )
    st.markdown("**Métodos a executar**")
    usar_jacobi = st.checkbox("Gauss-Jacobi", value=True)
    usar_seidel = st.checkbox("Gauss-Seidel", value=True)
    usar_newton = st.checkbox("Newton-Raphson", value=True)
    ##st.caption("Limites internos: 5000 iterações para Jacobi/Seidel e 100 para Newton-Raphson.")
    executar = st.button("Calcular fluxo de potência", type="primary", use_container_width=True)

Ybus = montar_ybus()
V0 = montar_tensao_inicial(pq_modulos, pq_angulos, pv_angulos)
P0, Q0 = calcular_potencias(V0, Ybus)
mismatch0 = calcular_mismatch(V0, Ybus)
erro0 = erro_maximo_mismatch(V0, Ybus)

if "resultados_fluxo" not in st.session_state:
    st.session_state["resultados_fluxo"] = None

if executar:
    if not (usar_jacobi or usar_seidel or usar_newton):
        st.error("Selecione pelo menos um método para executar.")
    else:
        metodos = []
        if usar_jacobi:
            metodos.append(("Gauss-Jacobi", gauss_jacobi))
        if usar_seidel:
            metodos.append(("Gauss-Seidel", gauss_seidel))
        if usar_newton:
            metodos.append(("Newton-Raphson", newton_raphson))
        resultados, estados = [], {}
        barra = st.progress(0)
        status = st.empty()
        for idx, (nome, funcao) in enumerate(metodos, start=1):
            status.write(f"Executando {nome}...")
            try:
                limite_metodo = 100 if nome == "Newton-Raphson" else 5000
                V, n, erro, historico, convergiu = funcao(
                    Ybus, V0, tolerancia=float(tolerancia), max_iter=limite_metodo
                )
                P, Q = calcular_potencias(V, Ybus)
                resultados.append({
                    "Método": nome, "Convergiu": "Sim" if convergiu else "Não",
                    "Iterações": int(n), "Mismatch máximo (p.u.)": float(erro),
                })
                estados[nome] = {
                    "V": V, "P": P, "Q": Q, "historico": historico,
                    "convergiu": convergiu, "iteracoes": n, "erro": erro,
                }
            except Exception as exc:
                resultados.append({
                    "Método": nome, "Convergiu": "Erro", "Iterações": None,
                    "Mismatch máximo (p.u.)": np.nan,
                })
                estados[nome] = {"erro_execucao": str(exc)}
            barra.progress(idx / len(metodos))
        status.empty()
        barra.empty()
        st.session_state["resultados_fluxo"] = {
            "resumo": resultados, "estados": estados, "tolerancia": tolerancia,
            "cenario": st.session_state.get("cenario_chute", "Personalizado"),
            "max_iter_jacobi_seidel": 5000, "max_iter_newton": 100,
            "chutes": {"PQ módulos": pq_modulos, "PQ ângulos": pq_angulos, "PV ângulos": pv_angulos},
        }

dados = st.session_state["resultados_fluxo"]
tab_sistema, tab_config, tab_resultados = st.tabs([
    "1 · Sistema elétrico", "2 · Configuração e métodos", "3 · Resultados e análise"
])

with tab_sistema:
    st.markdown(
        """
        <h2 style="text-align: center;">Sistema elétrico</h2>
        
        """,
        unsafe_allow_html=True
    )
    st.subheader("Sistema elétrico")
    ##st.caption("Topologia, parâmetros das barras e linhas e matriz de admitâncias da rede.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Barras", N_BARRAS)
    k2.metric("Linhas", len(LINHAS))
    ##k3.metric("Base de potência", f"{BASE_MVA:.0f} MVA")
    ##k4.metric("Equações de mismatch", 12)
    st.markdown("#### Diagrama de conectividade")
    ##st.caption("A posição dos nós é esquemática; as conexões correspondem aos dados do sistema.")
    dot = """
    graph sistema {
      graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.55", ranksep="0.75"];
      node [shape=circle, style=filled, fontname="Arial", fontsize=12, width=0.7, height=0.7, color="#B7C9D6", penwidth=1.5];
      edge [color="#8B9AA7", penwidth=1.6];
      1 [label="1\\nSlack", fillcolor="#F2C14E", fontcolor="#202020"];
      2 [label="2\\nPV", fillcolor="#78C6A3", fontcolor="#202020"];
      3 [label="3\\nPQ", fillcolor="#8ECAE6", fontcolor="#202020"];
      4 [label="4\\nPQ", fillcolor="#8ECAE6", fontcolor="#202020"];
      5 [label="5\\nPQ", fillcolor="#8ECAE6", fontcolor="#202020"];
      6 [label="6\\nPQ", fillcolor="#8ECAE6", fontcolor="#202020"];
      7 [label="7\\nPV", fillcolor="#78C6A3", fontcolor="#202020"];
      8 [label="8\\nPQ", fillcolor="#8ECAE6", fontcolor="#202020"];
      1 -- 2; 1 -- 8; 2 -- 3; 2 -- 4; 3 -- 4;
      3 -- 5; 4 -- 5; 5 -- 6; 6 -- 7; 7 -- 8;
    }
    """
    st.graphviz_chart(dot, use_container_width=True)
    ##st.markdown("**Legenda:** amarelo = Slack · verde = PV · azul = PQ")
    st.markdown("#### Dados das barras — Tabela 2")
    
    tabela_barras = [{
        "Barra": i + 1,
        "Tipo": TIPOS_BARRA[i],
        "Geração ativa Pg (p.u.)": P_GERACAO[i],
        "Geração reativa Qg (p.u.)": Q_GERACAO[i],
        "Carga ativa Pd (p.u.)": P_CARGA[i],
        "Carga reativa Qd (p.u.)": Q_CARGA[i],
        "Tensão especificada V (p.u.)": V_ESP_TABELA.get(i, np.nan),
        "Ângulo especificado δ (°)": ANG_ESP_TABELA.get(i, np.nan),
        "P líquida usada (p.u.)": P_ESP[i] if i != 0 else np.nan,
        "Q líquida usada (p.u.)": Q_ESP[i] if TIPOS_BARRA[i] == "PQ" else np.nan,
    } for i in range(N_BARRAS)]
    df_barras = pd.DataFrame(tabela_barras)

    
    colunas_com_hifen = [
        "Geração ativa Pg (p.u.)",
        "Geração reativa Qg (p.u.)",
        "Carga ativa Pd (p.u.)",
        "Carga reativa Qd (p.u.)",
        "Tensão especificada V (p.u.)",
        "Ângulo especificado δ (°)",
        "P líquida usada (p.u.)",
        "Q líquida usada (p.u.)",
    ]
    for coluna in colunas_com_hifen:
        df_barras[coluna] = df_barras[coluna].map(
            lambda valor: "-" if pd.isna(valor) else (
                f"{valor:.1f}" if coluna == "Ângulo especificado δ (°)"
                else f"{valor:.3f}"
            )
        )

    st.dataframe(
        df_barras,
        use_container_width=True,
        hide_index=True
    )
    st.caption("As potências líquidas usadas pelo algoritmo são calculadas por:")
    st.latex(r"P_i = P_{g,i} - P_{d,i}")
    st.latex(r"Q_i = Q_{g,i} - Q_{d,i}")
    st.caption(
        "Na barra Slack, P e Q são calculadas pela solução. "
        "Nas barras PV, Q é calculada pela solução."
    )
    st.markdown("#### Parâmetros das linhas")
    df_linhas = pd.DataFrame(LINHAS, columns=["Barra de", "Barra para", "R (p.u.)", "X (p.u.)", "Bsh (p.u.)"])
    st.dataframe(df_linhas, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Matriz de admitância nodal")
    st.subheader("Etapa 2 — Matriz de admitância nodal")
    st.write("A Ybus representa as admitâncias vistas entre as barras. Os elementos fora da diagonal representam conexões diretas; a diagonal reúne as contribuições das linhas conectadas.")
    st.latex(r"I=Y_{bus}V")
    st.latex(r"S_i=P_i+jQ_i=V_i I_i^*")
    modo_ybus = st.radio("Exibir matriz como", ["Forma complexa", "Parte real G", "Parte imaginária B"], horizontal=True)
    if modo_ybus == "Forma complexa":
        df_y = pd.DataFrame(
            [[f"{Ybus[i,j].real:.4f} {'+' if Ybus[i,j].imag >= 0 else '−'} j{abs(Ybus[i,j].imag):.4f}" for j in range(N_BARRAS)] for i in range(N_BARRAS)],
            index=[f"Barra {i}" for i in range(1, N_BARRAS+1)],
            columns=[str(i) for i in range(1, N_BARRAS+1)]
        )
    elif modo_ybus == "Parte real G":
        df_y = pd.DataFrame(Ybus.real, index=[f"Barra {i}" for i in range(1, N_BARRAS+1)], columns=[str(i) for i in range(1, N_BARRAS+1)])
    else:
        df_y = pd.DataFrame(Ybus.imag, index=[f"Barra {i}" for i in range(1, N_BARRAS+1)], columns=[str(i) for i in range(1, N_BARRAS+1)])
    st.dataframe(df_y, use_container_width=True)




with tab_config:
    st.markdown(
        '<h2 style="text-align: center;">Configuração e métodos</h2>',
        unsafe_allow_html=True
    )

    st.latex(r"\Delta P_i = P_i^{\mathrm{esp}} - P_i^{\mathrm{calc}}")
    st.latex(r"\Delta Q_i = Q_i^{\mathrm{esp}} - Q_i^{\mathrm{calc}}")

    with st.container(border=True):
        st.metric("Diferença máxima de potência", f"{erro0:.6f} p.u.")
        
    linhas_mis = []

    for i in range(N_BARRAS):
        tipo = TIPOS_BARRA[i]
        linhas_mis.append({
            "Barra": i+1, "Tipo": tipo,
            "P especificada": P_ESP[i], "P calculada": P0[i],
            "ΔP (p.u.)": P_ESP[i]-P0[i] if i != 0 else np.nan,
            "Q especificada": Q_ESP[i], "Q calculada": Q0[i],
            "ΔQ (p.u.)": Q_ESP[i]-Q0[i] if tipo == "PQ" else np.nan,
        })
    df_mismatch = pd.DataFrame(linhas_mis)
    for coluna in ["P especificada", "P calculada", "ΔP (p.u.)", "Q especificada", "Q calculada", "ΔQ (p.u.)"]:
        df_mismatch[coluna] = df_mismatch[coluna].map(
            lambda valor: "—" if pd.isna(valor) else f"{valor:.5f}"
        )
    st.dataframe(df_mismatch, use_container_width=True, hide_index=True)
    

    st.divider()
    st.markdown("### Como os métodos resolvem o sistema")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Gauss-Jacobi**")
            st.write("Calcula as novas tensões usando somente os valores da iteração anterior.")
            st.latex(r"V_i^{(k+1)}=\frac{1}{Y_{ii}}\left[\frac{P_i-jQ_i}{(V_i^{(k)})^*}-\sum_{j\ne i}Y_{ij}V_j^{(k)}\right]")
    with c2:
        with st.container(border=True):
            st.markdown("**Gauss-Seidel**")
            st.write("Atualiza as barras sequencialmente e usa imediatamente os valores novos já calculados.")
            st.latex(r"V_i^{(k+1)}=\frac{1}{Y_{ii}}\left[\frac{P_i-jQ_i}{(V_i^{(k)})^*}-\sum_{j<i}Y_{ij}V_j^{(k+1)}-\sum_{j>i}Y_{ij}V_j^{(k)}\right]")
    with c3:
        with st.container(border=True):
            st.markdown("**Newton-Raphson**")
            st.write("Calcula correções para os ângulos e módulos usando a Jacobiana.")
            st.latex(r"J\,\Delta x=\begin{bmatrix}\Delta P\\\Delta Q\end{bmatrix}")



with tab_resultados:
    st.markdown(
        """
        <h2 style="text-align: center;">Resultados e análise</h2>
        """,
        unsafe_allow_html=True
    )

    if not dados:
        st.info(
            "Configure a simulação na barra lateral e clique em "
            "Calcular fluxo de potência."
        )

    else:
        estados_validos = {
            nome: estado
            for nome, estado in dados["estados"].items()
            if "historico" in estado
        }

        # ==================================================
        # 1. RESUMO DA CONVERGÊNCIA
        # ==================================================
        resumo_df = pd.DataFrame(dados["resumo"])

        if not resumo_df.empty:
            st.markdown("#### Resumo da convergência")

            st.dataframe(
                resumo_df.style.format({
                    "Mismatch máximo (p.u.)": "{:.3e}"
                }),
                use_container_width=True,
                hide_index=True
            )

            if any(
                row.get("Convergiu") != "Sim"
                for row in dados["resumo"]
            ):
                st.warning(
                    "Pelo menos um método não atingiu o critério "
                    "de convergência."
                )
            else:
                st.success(
                    "Todos os métodos selecionados atingiram "
                    "o critério de convergência."
                )

        # ==================================================
        # 2. GRÁFICO DE CONVERGÊNCIA
        # ==================================================
        if estados_validos:
            st.markdown("#### Evolução do mismatch")

            st.caption(
                "Comparação da redução do erro ao longo das iterações. "
                "A linha tracejada representa a tolerância configurada."
            )

            import plotly.graph_objects as go

            fig = go.Figure()

            for nome, estado in estados_validos.items():
                hist = np.maximum(
                    np.asarray(estado["historico"], dtype=float),
                    1e-16
                )

                if len(hist):
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(1, len(hist) + 1),
                            y=hist,
                            mode="lines",
                            name=nome
                        )
                    )

            fig.add_hline(
                y=float(dados["tolerancia"]),
                line_dash="dash",
                annotation_text=(
                    f"Tolerância = {dados['tolerancia']:.0e}"
                )
            )

            fig.update_layout(
                xaxis_title="Iteração",
                yaxis_title="Mismatch máximo de potência (p.u.)",
                yaxis_type="log",
                height=450,
                legend_title="Método",
                margin=dict(l=20, r=20, t=35, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

        # ==================================================
        # 3. TENSÕES FINAIS POR BARRA
        # ==================================================
        if estados_validos:
            st.markdown("#### Tensões finais por barra")

            linhas_tensoes = []

            for nome, estado in estados_validos.items():
                V = estado["V"]
                P = estado["P"]
                Q = estado["Q"]

                for i in range(N_BARRAS):
                    linhas_tensoes.append({
                        "Método": nome,
                        "Barra": i + 1,
                        "Tipo": TIPOS_BARRA[i],
                        "|V| (p.u.)": abs(V[i]),
                        "Ângulo (°)": np.degrees(
                            np.angle(V[i])
                        ),
                        "P calculada (p.u.)": P[i],
                        "Q calculada (p.u.)": Q[i],
                    })

            df_tensoes = pd.DataFrame(linhas_tensoes)

            st.dataframe(
                df_tensoes.style.format({
                    "|V| (p.u.)": "{:.5f}",
                    "Ângulo (°)": "{:.4f}",
                    "P calculada (p.u.)": "{:.5f}",
                    "Q calculada (p.u.)": "{:.5f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=450
            )

            st.download_button(
                "Baixar tensões finais por barra (CSV)",
                df_tensoes.to_csv(index=False).encode("utf-8-sig"),
                "tensoes_finais_por_barra.csv",
                "text/csv",
                key="download_tensoes_comparacao"
            )

        # ==================================================
        # 4. ERROS DE EXECUÇÃO
        # ==================================================
        for nome, estado in dados["estados"].items():
            if "erro_execucao" in estado:
                st.error(
                    f"{nome}: {estado['erro_execucao']}"
                )

                # ===== VERIFICAÇÃO ENTRE MÉTODOS =====
        if len(estados_validos) > 1:
            nomes = list(estados_validos)
            ref = estados_validos[nomes[0]]["V"]
            for nome in nomes[1:]:
                dif = float(np.max(np.abs(estados_validos[nome]["V"] - ref)))
                if dif > 1e-3:
                    st.warning(
                        f"{nome} convergiu para uma solução diferente de {nomes[0]} "
                        f"(diferença máxima de tensão = {dif:.3f} p.u.). Isso indica "
                        "outra raiz das equações, causada pelo chute inicial. "
                        "Teste o cenário A."
                    )
                else:
                    st.success(f"{nome} e {nomes[0]} chegaram à mesma solução "
                               f"(diferença máxima = {dif:.1e} p.u.).")

            st.markdown("#### Comparação lado a lado")
            comp = {"Barra": list(range(1, N_BARRAS + 1))}
            for nome in nomes:
                comp[f"|V| {nome}"] = np.abs(estados_validos[nome]["V"])
                comp[f"δ(°) {nome}"] = np.degrees(np.angle(estados_validos[nome]["V"]))
            st.dataframe(pd.DataFrame(comp).style.format(
                {c: "{:.5f}" for c in comp if c != "Barra"}),
                use_container_width=True, hide_index=True)

        # ===== FLUXOS NAS LINHAS =====
        if estados_validos:
            st.markdown("#### Fluxos de potência nas linhas")
            metodo_fluxo = st.selectbox("Método usado para os fluxos", list(estados_validos))
            fl = pd.DataFrame(calcular_fluxos_linhas(estados_validos[metodo_fluxo]["V"]))
            fl = fl.rename(columns={
                "de": "De", "para": "Para",
                "P_de_para_pu": "P De→Para", "Q_de_para_pu": "Q De→Para",
                "P_para_de_pu": "P Para→De", "Q_para_de_pu": "Q Para→De",
                "P_perdas_pu": "Perda P", "Q_soma_pu": "Soma Q (perda − carga)",
            })
            st.dataframe(fl.style.format({c: "{:.5f}" for c in fl.columns if c not in ("De", "Para")}),
                         use_container_width=True, hide_index=True)
            P_ger = estados_validos[metodo_fluxo]["P"]
            c1, c2, c3 = st.columns(3)
            c1.metric("ΣP gerada − ΣP carga (perdas)", f"{P_ger.sum():.2e} p.u.")
            c2.metric("Soma das perdas P nas linhas", f"{fl['Perda P'].sum():.2e} p.u.")
            c3.metric("P da slack", f"{P_ger[0]:.4f} p.u.")        

st.divider()