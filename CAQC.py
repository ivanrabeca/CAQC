import os
import base64 
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp

# Configuração da página (tem de ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Otimizador Tático — CAQC | SwimRankings Style",
    layout="wide",
)


# ==============================================================================
# ESTILO SWIMRANKINGS (FUNDO #006691 | TEXTO #2499C7 | TABELAS #94E7FF)
# ==============================================================================
def aplicar_estilo_swimrankings():
    st.markdown(
        """
        <style>
        /* Fonte e fundo geral */
        .stApp {
            background-color: #006691;
            color: #FFFFFF;
            font-family: Verdana, Arial, Helvetica, sans-serif;
        }

        /* Texto geral e parágrafos em branco */
        .stApp p, .stApp label, .stApp span, .stMarkdown {
            color: #FFFFFF;
        }

        /* Barra lateral */
        [data-testid="stSidebar"] {
            background-color: #005073;
            border-right: 2px solid #2499C7;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #FFFFFF !important;
        }

        /* Cabeçalho superior estilo SwimRankings */
        .sr-header {
            background-color: #004A6B;
            border: 1px solid #2499C7;
            border-bottom: 3px solid #94E7FF;
            padding: 14px 20px;
            margin-bottom: 18px;
        }
        .sr-header h1 {
            color: #FFFFFF !important;
            font-family: Verdana, Arial, sans-serif !important;
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.4px;
        }
        .sr-header p {
            color: #FFFFFF !important;
            font-size: 0.85rem;
            margin: 4px 0 0 0;
            font-weight: 600;
        }

        /* Faixa de título de secção estilo SwimRankings */
        .sr-section-title {
            background-color: #004A6B;
            color: #FFFFFF;
            border-left: 5px solid #94E7FF;
            border-bottom: 1px solid #2499C7;
            padding: 7px 12px;
            font-size: 0.95rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 18px 0 10px 0;
        }

        /* Cartões de métricas */
        [data-testid="stMetric"] {
            background-color: #004A6B;
            border: 1px solid #2499C7;
            border-top: 3px solid #94E7FF;
            padding: 10px 14px;
        }
        [data-testid="stMetricLabel"] {
            color: #FFFFFF !important;
            font-size: 0.78rem !important;
            font-weight: 700;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-family: Verdana, Arial, sans-serif !important;
            font-size: 1.6rem !important;
            font-weight: 700;
        }
        [data-testid="stMetricDelta"] {
            color: #FFFFFF !important;
        }

        /* Tabelas HTML estilo SwimRankings (Fundo #94E7FF) */
        .sr-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid #2499C7;
            margin-bottom: 12px;
        }
        table.sr-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #94E7FF;
            color: #003B54;
            font-size: 0.82rem;
            font-family: Verdana, Arial, sans-serif;
        }
        table.sr-table thead th {
            background-color: #004A6B;
            color: #FFFFFF;
            text-align: left;
            padding: 7px 10px;
            border-bottom: 2px solid #006691;
            border-right: 1px solid #006691;
            font-weight: 700;
            white-space: nowrap;
        }
        table.sr-table tbody td {
            background-color: #94E7FF;
            color: #003B54;
            padding: 5px 10px;
            border-bottom: 1px solid #2499C7;
            border-right: 1px solid #6FD0EE;
        }
        table.sr-table tbody tr:hover td {
            background-color: #B4EFFF;
        }
        table.sr-table tbody tr.sr-highlight td {
            background-color: #004A6B !important;
            color: #FFFFFF !important;
            font-weight: 700;
        }

        /* Caixa de informação (suplentes) */
        .sr-info {
            background-color: #004A6B;
            border: 1px solid #2499C7;
            border-left: 4px solid #94E7FF;
            color: #FFFFFF;
            padding: 9px 14px;
            font-size: 0.84rem;
            margin-top: 6px;
        }

        hr {
            border-color: #2499C7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mostrar_tabela_sr(df, destacar_coluna=None, destacar_valor=None):
    """Renderiza um DataFrame como tabela HTML compacta no estilo SwimRankings (#94E7FF)."""
    colunas = df.columns.tolist()
    html = ['<div class="sr-table-wrap"><table class="sr-table"><thead><tr>']
    for col in colunas:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        destacar = (
            destacar_coluna is not None
            and str(row[destacar_coluna]) == str(destacar_valor)
        )
        classe = ' class="sr-highlight"' if destacar else ""
        html.append(f"<tr{classe}>")
        for col in colunas:
            html.append(f"<td>{row[col]}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ==============================================================================
# CONFIGURAÇÃO DOS FICHEIROS E REGRAS
# ==============================================================================
FICHEIRO_1DIV = "Campeonatos_Nacionais_Clubes_1Div_2025-v2.xlsx"
FICHEIRO_CAQC = "Recordes_Pessoais_25m_Condeixa_Aqua_Clube.xlsx"
CLUBE_ALVO_2025 = "Condeixa Clube"
NOME_CLUBE_SIM = "Condeixa Aqua Clube (CAQC)"


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def converter_para_segundos(valor):
    """Converte tempos (datetime.time, timedelta ou string MM:SS.cc) para segundos."""
    if pd.isna(valor) or str(valor).strip() in ["—", "-", "", "DSQ", "DNS"]:
        return None
    if isinstance(valor, datetime.time):
        return valor.hour * 3600 + valor.minute * 60 + valor.second + valor.microsecond / 1e6
    if isinstance(valor, datetime.timedelta):
        return valor.total_seconds()
    texto = str(valor).strip()
    partes = texto.split(":")
    try:
        if len(partes) == 1:
            return float(partes[0])
        if len(partes) == 2:
            return int(partes[0]) * 60 + float(partes[1])
        if len(partes) == 3:
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + float(partes[2])
    except ValueError:
        return None
    return None


def formatar_tempo(segundos):
    """Formata segundos para string legível (MM:SS.cc ou SS.cc)."""
    if segundos is None or pd.isna(segundos):
        return "—"
    minutos = int(segundos // 60)
    resto = segundos - minutos * 60
    return f"{minutos}:{resto:05.2f}" if minutos > 0 else f"{resto:.2f}"


def pontos_fpn(classificacao):
    """Tabela de pontuação FPN 1.ª Divisão (1.º = 17 pts; 2.º = 15 pts ... 16.º = 1 pt)."""
    if classificacao is None or str(classificacao) == "DSQ":
        return 0
    pos = int(classificacao)
    if pos == 1:
        return 17
    if 2 <= pos <= 16:
        return 17 - pos
    return 0


def estimar_pontos_fina(tempo_base, segundos):
    """Calcula a pontuação FINA aproximada: P = floor(1000 * (B / T)^3)."""
    if segundos is None or pd.isna(segundos) or segundos <= 0:
        return 0
    return int(np.floor(1000.0 * ((tempo_base / segundos) ** 3)))


# ==============================================================================
# 1. LEITURA E PREPARAÇÃO DOS DADOS (COM CACHE PARA SER MAIS RÁPIDO)
# ==============================================================================
@st.cache_data
def carregar_dados():
    df_1div = pd.read_excel(FICHEIRO_1DIV, sheet_name="Base_Dados", skiprows=3)
    df_1div["Segundos"] = df_1div["Tempo"].apply(converter_para_segundos)
    df_1div["Dia"] = df_1div["Data / Hora"].str.extract(r"(\d+ Dez \d+)")
    df_1div["Hora"] = df_1div["Data / Hora"].str.extract(r"- (\d+):").astype(int)
    df_1div["Sessao"] = df_1div["Dia"] + " " + df_1div["Hora"].apply(
        lambda h: "Manhã" if h < 13 else "Tarde"
    )

    provas = df_1div["Prova"].unique().tolist()

    base_fina = {}
    for prova in provas:
        sub = df_1div[
            (df_1div["Prova"] == prova)
            & df_1div["Segundos"].notna()
            & df_1div["Pts. FINA"].notna()
        ]
        tempos_b = sub["Segundos"] * ((sub["Pts. FINA"] / 1000.0) ** (1 / 3))
        base_fina[prova] = float(tempos_b.median())

    df_matriz = pd.read_excel(FICHEIRO_CAQC, sheet_name="Matriz_Recordes_25m", skiprows=4)
    colunas_ignorar = {"Estilo", "Prova", "Melhor Marca (25m)", "Atleta Mais Rápida"}
    atletas = [col for col in df_matriz.columns if col not in colunas_ignorar]

    recordes_caqc = {}
    for _, linha in df_matriz.iterrows():
        prova = str(linha["Prova"]).strip()
        if prova in provas:
            for atleta in atletas:
                recordes_caqc[(atleta, prova)] = converter_para_segundos(linha[atleta])

    return df_1div, provas, base_fina, atletas, recordes_caqc


# ==============================================================================
# 2. OTIMIZAÇÃO TÁTICA (PROGRAMAÇÃO LINEAR INTEIRA MISTA - MILP)
# ==============================================================================
def otimizar_alinhamento(
    df_1div, provas, base_fina, atletas, recordes_caqc, max_provas_atleta, max_provas_sessao
):
    n_atletas = len(atletas)
    n_provas = len(provas)

    sessoes_prova = df_1div.drop_duplicates("Prova").set_index("Prova")["Sessao"].to_dict()
    sessoes_unicas = sorted(df_1div["Sessao"].unique().tolist())

    mat_tempo = np.full((n_atletas, n_provas), np.nan)
    mat_class = np.full((n_atletas, n_provas), 17, dtype=int)
    mat_pts = np.full((n_atletas, n_provas), -1000.0)
    mat_fina = np.zeros((n_atletas, n_provas), dtype=int)

    rivais_por_prova = {}
    for j, prova in enumerate(provas):
        rivais = df_1div[
            (df_1div["Prova"] == prova) & (df_1div["Clube"] != CLUBE_ALVO_2025)
        ].copy()
        rivais_por_prova[prova] = rivais
        tempos_rivais = rivais["Segundos"].dropna()

        for i, atleta in enumerate(atletas):
            t = recordes_caqc.get((atleta, prova))
            if t is None:
                continue
            mat_tempo[i, j] = t
            mais_rapidas = int((tempos_rivais < t).sum())
            pos = mais_rapidas + 1
            mat_class[i, j] = pos
            mat_pts[i, j] = pontos_fpn(pos)
            mat_fina[i, j] = estimar_pontos_fina(base_fina[prova], t)

    c = np.zeros(n_atletas * n_provas)
    for i in range(n_atletas):
        for j in range(n_provas):
            idx = i * n_provas + j
            if mat_pts[i, j] < 0:
                c[idx] = 1e6
            else:
                c[idx] = -(mat_pts[i, j] + 1e-5 * mat_fina[i, j])

    restricoes_A = []
    lb_list = []
    ub_list = []

    for j in range(n_provas):
        linha = np.zeros(n_atletas * n_provas)
        for i in range(n_atletas):
            linha[i * n_provas + j] = 1
        restricoes_A.append(linha)
        lb_list.append(1)
        ub_list.append(1)

    for i in range(n_atletas):
        linha = np.zeros(n_atletas * n_provas)
        for j in range(n_provas):
            linha[i * n_provas + j] = 1
        restricoes_A.append(linha)
        lb_list.append(0)
        ub_list.append(max_provas_atleta)

    for i in range(n_atletas):
        for sessao in sessoes_unicas:
            linha = np.zeros(n_atletas * n_provas)
            for j, prova in enumerate(provas):
                if sessoes_prova[prova] == sessao:
                    linha[i * n_provas + j] = 1
            restricoes_A.append(linha)
            lb_list.append(0)
            ub_list.append(max_provas_sessao)

    resultado = milp(
        c=c,
        integrality=np.ones(n_atletas * n_provas),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.array(restricoes_A), lb_list, ub_list),
    )

    if not resultado.success:
        raise RuntimeError(f"Erro na otimização: {resultado.message}")

    x = np.round(resultado.x).reshape(n_atletas, n_provas)

    alinhamento = []
    for j, prova in enumerate(provas):
        i = int(np.argmax(x[:, j]))
        alinhamento.append({
            "Prova": prova,
            "Sessão": sessoes_prova[prova],
            "Atleta": atletas[i],
            "Tempo": formatar_tempo(mat_tempo[i, j]),
            "Segundos": mat_tempo[i, j],
            "Class. Est.": f"{int(mat_class[i, j])}º",
            "Pts. FPN": int(mat_pts[i, j]),
            "Pts. FINA": int(mat_fina[i, j]),
        })

    return pd.DataFrame(alinhamento), rivais_por_prova


# ==============================================================================
# 3. SIMULAÇÃO DA CLASSIFICAÇÃO GERAL DOS 16 CLUBES
# ==============================================================================
def simular_classificacao_clubes(df_alinhamento, rivais_por_prova):
    pontos_clube = {}
    fina_clube = {}

    for _, linha_caqc in df_alinhamento.iterrows():
        prova = linha_caqc["Prova"]
        rivais = rivais_por_prova[prova][["Clube", "Segundos", "Pts. FINA", "Class."]].copy()

        entrada_caqc = pd.DataFrame([{
            "Clube": NOME_CLUBE_SIM,
            "Segundos": linha_caqc["Segundos"],
            "Pts. FINA": linha_caqc["Pts. FINA"],
            "Class.": "OK",
        }])

        combinado = pd.concat([rivais, entrada_caqc], ignore_index=True)
        combinado["DSQ"] = combinado["Class."].astype(str) == "DSQ"
        combinado = combinado.sort_values(
            ["DSQ", "Segundos"], ascending=[True, True]
        ).reset_index(drop=True)

        for idx, r in combinado.iterrows():
            clube = r["Clube"]
            pts = 0 if r["DSQ"] else pontos_fpn(idx + 1)
            fn = 0 if pd.isna(r["Pts. FINA"]) else int(r["Pts. FINA"])
            pontos_clube[clube] = pontos_clube.get(clube, 0) + pts
            fina_clube[clube] = fina_clube.get(clube, 0) + fn

    tabela = pd.DataFrame([
        {"Clube": cl, "Pts. FPN": pontos_clube[cl], "Pts. FINA": fina_clube[cl]}
        for cl in pontos_clube
    ]).sort_values(["Pts. FPN", "Pts. FINA"], ascending=False).reset_index(drop=True)
    tabela.insert(0, "Pos.", [f"{i}º" for i in range(1, len(tabela) + 1)])
    return tabela


# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
def main():
    aplicar_estilo_swimrankings()

    caminho_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cabecalho_caqc.png")
    if os.path.exists(caminho_img):
        with open(caminho_img, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="
                background-color: #FFFFFF;
                border: 1px solid #2499C7;
                border-bottom: 3px solid #94E7FF;
                padding: 8px 16px;
                margin-bottom: 18px;
                text-align: center;
            ">
                <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; max-height: 140px; height: auto;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Imagem não encontrada em: {caminho_img}")

    with st.sidebar:
        st.header("Regras da Otimização")
        max_provas_atleta = st.number_input(
            "Máx. provas por atleta", min_value=1, max_value=10, value=3
        )
        max_provas_sessao = st.number_input(
            "Máx. provas por sessão", min_value=1, max_value=5, value=2
        )

    try:
        df_1div, provas, base_fina, atletas, recordes_caqc = carregar_dados()
        df_alinhamento, rivais_por_prova = otimizar_alinhamento(
            df_1div, provas, base_fina, atletas, recordes_caqc,
            max_provas_atleta, max_provas_sessao
        )
        tabela_clubes = simular_classificacao_clubes(df_alinhamento, rivais_por_prova)
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar os dados: {e}")
        return

    # Métricas principais
    pos_caqc = tabela_clubes.loc[tabela_clubes["Clube"] == NOME_CLUBE_SIM, "Pos."].values[0]
    pts_caqc = int(df_alinhamento["Pts. FPN"].sum())
    fina_caqc = int(df_alinhamento["Pts. FINA"].sum())
    media_fina = fina_caqc / len(provas)

    col1, col2, col3 = st.columns(3)
    col1.metric("Classificação Estimada", f"{pos_caqc} Lugar", "em 16 clubes")
    col2.metric("Total de Pontos FPN", f"{pts_caqc} pts")
    col3.metric("Total de Pontos FINA", f"{fina_caqc} pts", f"Média: {media_fina:.1f} pts/prova")

    # Secção 1: Alinhamento Ótimo
    st.markdown(
        '<div class="sr-section-title">1. Alinhamento Ótimo por Prova</div>',
        unsafe_allow_html=True,
    )
    cols_mostrar = ["Prova", "Sessão", "Atleta", "Tempo", "Class. Est.", "Pts. FPN", "Pts. FINA"]
    mostrar_tabela_sr(df_alinhamento[cols_mostrar])

    # Secção 2: Pontos por Atleta
    st.markdown(
        '<div class="sr-section-title">2. Pontos Estimados por Atleta</div>',
        unsafe_allow_html=True,
    )
    resumo_atletas = (
        df_alinhamento.groupby("Atleta")
        .agg(
            Provas=("Prova", lambda s: ", ".join(s)),
            **{"N.º Provas": ("Prova", "count")},
            **{"Pts. FPN": ("Pts. FPN", "sum")},
            **{"Pts. FINA": ("Pts. FINA", "sum")},
        )
        .sort_values(["Pts. FPN", "Pts. FINA"], ascending=False)
        .reset_index()
    )
    mostrar_tabela_sr(resumo_atletas)

    suplentes = [a for a in atletas if a not in resumo_atletas["Atleta"].values]
    texto_suplentes = ", ".join(suplentes) if suplentes else "Nenhum (todos os atletas têm pelo menos 1 prova)."
    st.markdown(
        f'<div class="sr-info"><strong>Atletas suplentes (0 provas individuais):</strong> {texto_suplentes}</div>',
        unsafe_allow_html=True,
    )

    # Secção 3: Classificação Geral dos Clubes
    st.markdown(
        '<div class="sr-section-title">3. Classificação Final Estimada — 1.ª Divisão (16 Provas Individuais)</div>',
        unsafe_allow_html=True,
    )
    mostrar_tabela_sr(tabela_clubes, destacar_coluna="Clube", destacar_valor=NOME_CLUBE_SIM)


if __name__ == "__main__":
    main()