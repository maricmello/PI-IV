"""Variaveis de contexto usadas na modelagem final (notebooks 006 a 008).

Motivacao (diagnostico dos residuos do baseline `ridge_completo`):
- O residuo medio variava muito por clube: falta contexto do clube.
- `minutos_jogados` e `jogos_disputados` sao totais acumulados e a temporada 2025 e
  incompleta: usamos a participacao relativa aos jogos do clube.
- `position` agrupa demais (lateral e zagueiro, ponta e centroavante): usamos `sub_position`.
"""
import numpy as np
import pandas as pd

SUBPOSICOES = [
    "Goalkeeper", "Centre-Back", "Left-Back", "Right-Back",
    "Defensive Midfield", "Central Midfield", "Attacking Midfield",
    "Left Winger", "Right Winger", "Centre-Forward",
]
COLUNAS_SUBPOSICAO = ["sub_" + s.lower().replace(" ", "_").replace("-", "_") for s in SUBPOSICOES]


def estatisticas_clube(jogos_br: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por clube-temporada, calculada so com os resultados das partidas."""
    casa = pd.DataFrame({
        "season": jogos_br["season"], "clube_temporada": jogos_br["home_club_name"],
        "gols_pro": jogos_br["home_club_goals"], "gols_contra": jogos_br["away_club_goals"],
    })
    fora = pd.DataFrame({
        "season": jogos_br["season"], "clube_temporada": jogos_br["away_club_name"],
        "gols_pro": jogos_br["away_club_goals"], "gols_contra": jogos_br["home_club_goals"],
    })
    p = pd.concat([casa, fora], ignore_index=True)
    p["pontos"] = np.select([p["gols_pro"] > p["gols_contra"], p["gols_pro"] == p["gols_contra"]], [3, 1], 0)

    clube = p.groupby(["season", "clube_temporada"], as_index=False).agg(
        jogos_clube=("pontos", "size"),
        pontos=("pontos", "sum"),
        gols_clube=("gols_pro", "sum"),
        gols_sofridos_clube=("gols_contra", "sum"),
    )
    clube["pontos_por_jogo_clube"] = clube["pontos"] / clube["jogos_clube"]
    clube["saldo_gols_por_jogo_clube"] = (clube["gols_clube"] - clube["gols_sofridos_clube"]) / clube["jogos_clube"]
    return clube


def montar_features_contexto(
    features: pd.DataFrame,
    jogador_temporada: pd.DataFrame,
    jogos_br: pd.DataFrame,
) -> pd.DataFrame:
    """Parte de features_modelo.csv (notebook 004) e acrescenta as variaveis de contexto."""
    df = features.copy()

    extra = jogador_temporada[["season", "player_id", "sub_position", "gols", "assistencias"]]
    df = df.merge(extra, on=["season", "player_id"], how="left", validate="one_to_one")

    clube = estatisticas_clube(jogos_br)
    df = df.merge(
        clube[["season", "clube_temporada", "jogos_clube", "gols_clube",
               "pontos_por_jogo_clube", "saldo_gols_por_jogo_clube"]],
        on=["season", "clube_temporada"], how="left", validate="many_to_one",
    )

    # Participacao: comparavel entre temporada completa e incompleta.
    # clip(upper=1) cobre quem trocou de clube no meio da temporada.
    df["participacao_minutos"] = (df["minutos_jogados"] / (df["jogos_clube"] * 90)).clip(upper=1)
    df["participacao_jogos"] = (df["jogos_disputados"] / df["jogos_clube"]).clip(upper=1)
    df["participacao_gols_clube"] = ((df["gols"] + df["assistencias"]) / df["gols_clube"]).clip(upper=1)

    for nome, coluna in zip(SUBPOSICOES, COLUNAS_SUBPOSICAO):
        df[coluna] = (df["sub_position"] == nome).astype(int)

    return df.drop(columns=["gols", "assistencias", "gols_clube"])
