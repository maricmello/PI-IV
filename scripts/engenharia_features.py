import numpy as np
import pandas as pd

FAIXAS_IDADE = [0, 21, 25, 29, 33, 100]
ROTULOS_FAIXAS_IDADE = ["ate_21", "22_25", "26_29", "30_33", "34_mais"]

POSICOES = ["Attack", "Defender", "Goalkeeper", "Midfield"]
PES = {"both": "pe_ambos", "left": "pe_esquerdo", "right": "pe_direito"}


def calcular_faixa_idade(idade: pd.Series) -> pd.Series:
    """Faixas fixas (definidas pelo dominio, nao pelos dados) para nao vazar informacao."""
    return pd.cut(idade, bins=FAIXAS_IDADE, labels=ROTULOS_FAIXAS_IDADE).astype(str)


def calcular_anos_para_fim_contrato(
    jogador_temporada: pd.DataFrame,
    jogos_brasileirao: pd.DataFrame,
) -> pd.Series:
    """Anos entre o ultimo jogo da temporada no dataset e o fim do contrato."""
    data_ref = pd.to_datetime(jogos_brasileirao.groupby("season")["date"].max())
    fim_contrato = pd.to_datetime(jogador_temporada["contract_expiration_date"], errors="coerce")
    ref = jogador_temporada["season"].map(data_ref)
    return (fim_contrato - ref).dt.days / 365.25


def montar_features(
    jogador_temporada: pd.DataFrame,
    jogos_brasileirao: pd.DataFrame,
) -> pd.DataFrame:
    df = jogador_temporada.copy()

    # Alvo: so entram jogadores com valor de mercado conhecido
    df = df.dropna(subset=["market_value_in_eur"]).copy()
    df = df.rename(columns={"market_value_in_eur": "valor_mercado_eur"})
    df["log_valor_mercado"] = np.log(df["valor_mercado_eur"])

    # Perfil
    df["posicao"] = df["position"]
    df["faixa_idade"] = calcular_faixa_idade(df["idade"])
    df["idade2"] = df["idade"] ** 2
    df["altura_cm"] = df["height_in_cm"]
    df["eh_estrangeiro"] = (df["country_of_citizenship"] != "Brazil").astype(int)

    for pos in POSICOES:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    for pe, coluna in PES.items():
        df[coluna] = (df["foot"] == pe).astype(int)

    # Contrato (NaN mantido: a imputacao e feita dentro do pipeline, so com dados de treino)
    df["anos_para_fim_contrato"] = calcular_anos_para_fim_contrato(df, jogos_brasileirao)
    df["sem_info_contrato"] = df["anos_para_fim_contrato"].isna().astype(int)

    colunas = (
        ["season", "player_id", "name", "clube_temporada", "temporada_completa",
         "posicao", "faixa_idade", "valor_mercado_eur", "log_valor_mercado"]
        + ["idade", "idade2", "altura_cm", "eh_estrangeiro"]
        + [f"pos_{p}" for p in POSICOES]
        + list(PES.values())
        + ["gols_por90", "assistencias_por90", "cartoes_amarelos_por90", "cartoes_vermelhos_por90",
           "minutos_jogados", "jogos_disputados"]
        + ["anos_para_fim_contrato", "sem_info_contrato"]
    )
    return df[colunas].sort_values(["season", "player_id"]).reset_index(drop=True)
