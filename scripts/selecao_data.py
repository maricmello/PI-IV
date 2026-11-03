from pathlib import Path

import numpy as np
import pandas as pd


def classificar_cartoes(descricao: pd.Series) -> pd.Series:
    texto = descricao.fillna("").str.lower()
    tipo = pd.Series(np.nan, index=descricao.index, dtype="object")
    tipo[texto.str.contains("yellow")] = "yellow_card"
    tipo[texto.str.contains("red|second yellow|2nd yellow")] = "red_card"
    return tipo


def ler_csv_filtrado(caminho: Path, ids_jogos: set) -> pd.DataFrame:
    tabela = pd.read_csv(caminho, low_memory=False)
    return tabela[tabela["game_id"].isin(ids_jogos)].copy()


def calcular_minutos(
    escalacoes: pd.DataFrame,
    substituicoes: pd.DataFrame,
    cartoes: pd.DataFrame,
    minuto_final_padrao: int,
) -> pd.DataFrame:
    entrada_sub = substituicoes.groupby(["game_id", "player_in_id"])["minute"].min().rename("inicio_sub")
    saida_sub = substituicoes.groupby(["game_id", "player_id"])["minute"].min().rename("fim_sub")
    saida_vermelho = (
        cartoes[cartoes["tipo_cartao"] == "red_card"]
        .groupby(["game_id", "player_id"])["minute"].min().rename("fim_vermelho")
    )

    pg = escalacoes
    for serie in (entrada_sub, saida_sub, saida_vermelho):
        pg = pg.merge(serie, left_on=["game_id", "player_id"], right_index=True, how="left")

    pg["minuto_inicio"] = np.where(pg["type"] == "starting_lineup", 0, pg["inicio_sub"])
    pg = pg[~((pg["type"] != "starting_lineup") & pg["inicio_sub"].isna())].copy()
    pg["minuto_fim"] = pg[["fim_sub", "fim_vermelho"]].fillna(minuto_final_padrao).min(axis=1)
    pg["minutos_jogados"] = (pg["minuto_fim"] - pg["minuto_inicio"]).clip(lower=0)
    pg = pg[pg["minutos_jogados"] > 0]

    aparicoes = (
        pg[["game_id", "player_id", "club_id", "position", "minutos_jogados"]]
        .rename(columns={"position": "posicao_no_jogo"})
        .reset_index(drop=True)
    )
    aparicoes["minutos_jogados"] = aparicoes["minutos_jogados"].astype(int)
    return aparicoes


def contar_eventos(gols: pd.DataFrame, cartoes: pd.DataFrame) -> tuple:
    n_gols = gols.groupby(["game_id", "player_id"]).size().rename("gols")
    n_assistencias = (
        gols.dropna(subset=["player_assist_id"])
        .groupby(["game_id", "player_assist_id"]).size().rename("assistencias")
    )
    n_assistencias.index = n_assistencias.index.set_names(["game_id", "player_id"])
    n_amarelos = (
        cartoes[cartoes["tipo_cartao"] == "yellow_card"]
        .groupby(["game_id", "player_id"]).size().rename("cartoes_amarelos")
    )
    n_vermelhos = (
        cartoes[cartoes["tipo_cartao"] == "red_card"]
        .groupby(["game_id", "player_id"]).size().rename("cartoes_vermelhos")
    )
    return n_gols, n_assistencias, n_amarelos, n_vermelhos


def montar_aparicoes_br(
    jogos_br: pd.DataFrame,
    pasta_bruto: Path,
    minuto_final_padrao: int = 90,
) -> pd.DataFrame:
    pasta_bruto = Path(pasta_bruto)
    ids_jogos = set(jogos_br["game_id"])

    escalacoes = ler_csv_filtrado(pasta_bruto / "game_lineups.csv", ids_jogos)
    eventos = ler_csv_filtrado(pasta_bruto / "game_events.csv", ids_jogos)

    substituicoes = eventos[eventos["type"] == "Substitutions"]
    gols = eventos[eventos["type"] == "Goals"]
    cartoes = eventos[eventos["type"] == "Cards"].copy()
    cartoes["tipo_cartao"] = classificar_cartoes(cartoes["description"])

    aparicoes = calcular_minutos(escalacoes, substituicoes, cartoes, minuto_final_padrao)

    aparicoes = aparicoes.set_index(["game_id", "player_id"])
    for serie in contar_eventos(gols, cartoes):
        aparicoes = aparicoes.join(serie, how="left")
    aparicoes = aparicoes.reset_index()

    for col in ("gols", "assistencias", "cartoes_amarelos", "cartoes_vermelhos"):
        aparicoes[col] = aparicoes[col].fillna(0).astype(int)

    aparicoes.insert(0, "season", aparicoes["game_id"].map(jogos_br.set_index("game_id")["season"]))
    return aparicoes


def agregar_temporada(
    aparicoes_br: pd.DataFrame,
    jogos_br: pd.DataFrame,
    minutos_minimos: int,
) -> pd.DataFrame:
    aparicoes = aparicoes_br.copy()
    aparicoes["data_jogo"] = pd.to_datetime(aparicoes["game_id"].map(jogos_br.set_index("game_id")["date"]))

    jogador_temporada = aparicoes.groupby(["season", "player_id"], as_index=False).agg(
        gols=("gols", "sum"),
        assistencias=("assistencias", "sum"),
        minutos_jogados=("minutos_jogados", "sum"),
        cartoes_amarelos=("cartoes_amarelos", "sum"),
        cartoes_vermelhos=("cartoes_vermelhos", "sum"),
        jogos_disputados=("game_id", "count"),
        data_media=("data_jogo", "mean"),
        data_ultimo_jogo=("data_jogo", "max"),
    )

    jogos_por_temporada = jogos_br.groupby("season")["game_id"].count()
    temporada_completa = jogos_por_temporada == jogos_por_temporada.max()
    jogador_temporada["temporada_completa"] = jogador_temporada["season"].map(temporada_completa)

    return jogador_temporada[jogador_temporada["minutos_jogados"] >= minutos_minimos].copy()


def adicionar_por90(jogador_temporada: pd.DataFrame) -> pd.DataFrame:
    for col in ("gols", "assistencias", "cartoes_amarelos", "cartoes_vermelhos"):
        jogador_temporada[f"{col}_por90"] = jogador_temporada[col] / jogador_temporada["minutos_jogados"] * 90
    return jogador_temporada


def adicionar_clube_temporada(
    jogador_temporada: pd.DataFrame,
    aparicoes_br: pd.DataFrame,
    clubes: pd.DataFrame,
) -> pd.DataFrame:
    nomes_clubes = clubes[["club_id", "name"]].rename(columns={"name": "clube_temporada"})
    clube_temporada = (
        aparicoes_br.groupby(["season", "player_id", "club_id"], as_index=False)["minutos_jogados"].sum()
        .sort_values(["season", "player_id", "minutos_jogados"], ascending=[True, True, False])
        .drop_duplicates(["season", "player_id"])[["season", "player_id", "club_id"]]
        .merge(nomes_clubes, on="club_id", how="left")
        .drop(columns="club_id")
    )
    return jogador_temporada.merge(clube_temporada, on=["season", "player_id"], how="left")


def adicionar_dados_jogador(jogador_temporada: pd.DataFrame, jogadores: pd.DataFrame) -> pd.DataFrame:
    colunas_jogador = [
        c for c in [
            "player_id", "name", "position", "sub_position", "foot", "height_in_cm",
            "date_of_birth", "country_of_citizenship", "current_club_name", "contract_expiration_date",
        ] if c in jogadores.columns
    ]
    jogador_temporada = jogador_temporada.merge(jogadores[colunas_jogador], on="player_id", how="left")

    nascimento = pd.to_datetime(jogador_temporada["date_of_birth"], errors="coerce")
    jogador_temporada["idade"] = ((jogador_temporada["data_media"] - nascimento).dt.days / 365.25).round(1)
    return jogador_temporada.drop(columns="data_media")


def adicionar_valor_mercado(jogador_temporada: pd.DataFrame, valores_jogadores: pd.DataFrame) -> pd.DataFrame:
    valores = valores_jogadores[["player_id", "date", "market_value_in_eur"]].copy()
    valores["date"] = pd.to_datetime(valores["date"]).astype("datetime64[ns]")
    valores = valores.dropna(subset=["date"]).sort_values("date")

    jogador_temporada["data_ultimo_jogo"] = jogador_temporada["data_ultimo_jogo"].astype("datetime64[ns]")
    jogador_temporada = pd.merge_asof(
        jogador_temporada.sort_values("data_ultimo_jogo"),
        valores,
        left_on="data_ultimo_jogo",
        right_on="date",
        by="player_id",
        direction="backward",
    )
    return jogador_temporada.drop(columns=["data_ultimo_jogo", "date"])


def montar_jogador_temporada(
    aparicoes_br: pd.DataFrame,
    jogos_br: pd.DataFrame,
    jogadores: pd.DataFrame,
    clubes: pd.DataFrame,
    valores_jogadores: pd.DataFrame,
    minutos_minimos: int = 900,
) -> pd.DataFrame:
    jogador_temporada = agregar_temporada(aparicoes_br, jogos_br, minutos_minimos)
    jogador_temporada = adicionar_por90(jogador_temporada)
    jogador_temporada = adicionar_clube_temporada(jogador_temporada, aparicoes_br, clubes)
    jogador_temporada = adicionar_dados_jogador(jogador_temporada, jogadores)
    jogador_temporada = adicionar_valor_mercado(jogador_temporada, valores_jogadores)
    return jogador_temporada.sort_values(["season", "player_id"]).reset_index(drop=True)


def montar_base_brasileirao(
    jogos_br: pd.DataFrame,
    jogadores: pd.DataFrame,
    clubes: pd.DataFrame,
    valores_jogadores: pd.DataFrame,
    pasta_bruto: Path,
    minutos_minimos: int = 900,
) -> pd.DataFrame:
    aparicoes_br = montar_aparicoes_br(jogos_br, pasta_bruto)
    return montar_jogador_temporada(
        aparicoes_br, jogos_br, jogadores, clubes, valores_jogadores, minutos_minimos
    )