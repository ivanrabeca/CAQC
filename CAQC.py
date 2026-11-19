import datetime
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp

# Configuração da página (tem de ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Otimizador Tático — CAQC",
    page_icon="🏊",
    layout="wide",
)

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
    # Ler resultados da 1.ª Divisão 2025
    df_1div = pd.read_excel(FICHEIRO_1DIV, sheet_name="Base_Dados", skiprows=3)
    df_1div["Segundos"] = df_1div["Tempo"].apply(converter_para_segundos)
    df_1div["Dia"] = df_1div["Data / Hora"].str.extract(r"(\d+ Dez \d+)")
    df_1div["Hora"] = df_1div["Data / Hora"].str.extract(r"- (\d+):").astype(int)
    df_1div["Sessao"] = df_1div["Dia"] + " " + df_1div["Hora"].apply(
        lambda h: "Manhã" if h < 13 else "Tarde"
    )

    provas = df_1div["Prova"].unique().tolist()

    # Calcular tempo base FINA por prova a partir dos dados oficiais de 2025
    base_fina = {}
    for prova in provas:
        sub = df_1div[
            (df_1div["Prova"] == prova)
            & df_1div["Segundos"].notna()
            & df_1div["Pts. FINA"].notna()
        ]
        tempos_b = sub["Segundos"] * ((sub["Pts. FINA"] / 1000.0) ** (1 / 3))
        base_fina[prova] = float(tempos_b.median())

    # Ler matriz de recordes pessoais do CAQC (25m)
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

    # Função objetivo: maximizar Pontos FPN (desempate por Pontos FINA)
    c = np.zeros(n_atletas * n_provas)
    for i in range(n_atletas):
        for j in range(n_provas):
            idx = i * n_provas + j
            if mat_pts[i, j] < 0:
                c[idx] = 1e6  # Impedir escolha sem marca
            else:
                c[idx] = -(mat_pts[i, j] + 1e-5 * mat_fina[i, j])

    restricoes_A = []
    lb_list = []
    ub_list = []

    # Restrição 1: Exatamente 1 atleta por prova
    for j in range(n_provas):
        linha = np.zeros(n_atletas * n_provas)
        for i in range(n_atletas):
            linha[i * n_provas + j] = 1
        restricoes_A.append(linha)
        lb_list.append(1)
        ub_list.append(1)

    # Restrição 2: Máximo de provas individuais por atleta
    for i in range(n_atletas):
        linha = np.zeros(n_atletas * n_provas)
        for j in range(n_provas):
            linha[i * n_provas + j] = 1
        restricoes_A.append(linha)
        lb_list.append(0)
        ub_list.append(max_provas_atleta)

    # Restrição 3: Limite de provas por atleta em cada sessão
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
            "Class_Est": int(mat_class[i, j]),
            "Pts_FPN": int(mat_pts[i, j]),
            "Pts_FINA": int(mat_fina[i, j]),
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
            "Pts. FINA": linha_caqc["Pts_FINA"],
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
        {"Clube": cl, "Pts_FPN": pontos_clube[cl], "Pts_FINA": fina_clube[cl]}
        for cl in pontos_clube
    ]).sort_values(["Pts_FPN", "Pts_FINA"], ascending=False).reset_index(drop=True)
    tabela.index = tabela.index + 1
    tabela.index.name = "Pos."
    return tabela


# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
def main():
    st.title("🏊 Condeixa Aqua Clube (CAQC)")
    st.caption("Otimizador Tático — 1.ª Divisão Nacional (Piscina 25m)")

    # Barra lateral com parâmetros configuráveis
    with st.sidebar:
        st.header("⚙️ Regras da Otimização")
        max_provas_atleta = st.number_input(
            "Máx. provas por atleta", min_value=1, max_value=10, value=3
        )
        max_provas_sessao = st.number_input(
            "Máx. provas por sessão", min_value=1, max_value=5, value=2
        )

    # Carregar dados e executar otimização
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

    # Cartões de Resumo (Métricas)
    pos_caqc = tabela_clubes.index[tabela_clubes["Clube"] == NOME_CLUBE_SIM][0]
    pts_caqc = int(df_alinhamento["Pts_FPN"].sum())
    fina_caqc = int(df_alinhamento["Pts_FINA"].sum())
    media_fina = fina_caqc / len(provas)

    col1, col2, col3 = st.columns(3)
    col1.metric("Classificação Estimada", f"{pos_caqc}.º Lugar", "em 16 clubes")
    col2.metric("Total de Pontos FPN", f"{pts_caqc} pts")
    col3.metric("Total de Pontos FINA", f"{fina_caqc} pts", f"Média: {media_fina:.1f} pts/prova")

    st.divider()

    # Secção 1: Alinhamento Ótimo
    st.subheader("1. Quem deve nadar cada prova (Alinhamento Ótimo)")
    cols_mostrar = ["Prova", "Sessão", "Atleta", "Tempo", "Class_Est", "Pts_FPN", "Pts_FINA"]
    st.dataframe(
        df_alinhamento[cols_mostrar],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Secção 2: Pontos por Atleta
    st.subheader("2. Pontos Estimados por Atleta")
    resumo_atletas = (
        df_alinhamento.groupby("Atleta")
        .agg(
            Provas=("Prova", lambda s: ", ".join(s)),
            N_Provas=("Prova", "count"),
            Pts_FPN=("Pts_FPN", "sum"),
            Pts_FINA=("Pts_FINA", "sum"),
        )
        .sort_values(["Pts_FPN", "Pts_FINA"], ascending=False)
    )
    st.dataframe(resumo_atletas, use_container_width=True)

    suplentes = [a for a in atletas if a not in resumo_atletas.index]
    if suplentes:
        st.info(f"**Atletas suplentes (0 provas individuais):** {', '.join(suplentes)}")
    else:
        st.info("**Atletas suplentes:** Nenhum (todos os atletas têm pelo menos 1 prova).")

    st.divider()

    # Secção 3: Classificação Geral dos Clubes
    st.subheader("3. Classificação Final Estimada — 1.ª Divisão (16 Provas Individuais)")

    def realcar_condeixa(linha):
        if linha["Clube"] == NOME_CLUBE_SIM:
            return ["background-color: #005f73; color: white; font-weight: bold"] * len(linha)
        return [""] * len(linha)

    st.dataframe(
        tabela_clubes.style.apply(realcar_condeixa, axis=1),
        use_container_width=True,)


if __name__ == "__main__":
    main()
