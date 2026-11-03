import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALVO = "log_valor_mercado"
GRUPO = "player_id"
SEED = 42

COLUNAS_POSICAO = ["pos_Attack", "pos_Defender", "pos_Goalkeeper", "pos_Midfield"]
COLUNAS_PERFIL = ["idade", "idade2", "eh_estrangeiro"] + COLUNAS_POSICAO
COLUNAS_COMPLETO = COLUNAS_PERFIL + [
    "altura_cm", "pe_ambos", "pe_esquerdo", "pe_direito",
    "gols_por90", "assistencias_por90", "cartoes_amarelos_por90", "cartoes_vermelhos_por90",
    "minutos_jogados", "jogos_disputados",
    "anos_para_fim_contrato", "sem_info_contrato",
]


class MedianaPorGrupo(BaseEstimator, RegressorMixin):
    """Preve a mediana do alvo no grupo do jogador.

    `niveis` e uma lista de combinacoes de colunas, da mais especifica para a mais geral.
    Grupos com menos de `min_obs` jogadores no treino sao ignorados; se nenhum nivel
    serve, usa a mediana global do treino. Tudo e calculado so com os dados do fit.
    """

    def __init__(self, niveis=(("posicao", "faixa_idade"), ("posicao",)), min_obs=5):
        self.niveis = niveis
        self.min_obs = min_obs

    @staticmethod
    def _chaves(X, colunas):
        return X[list(colunas)].astype(str).agg("|".join, axis=1)

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.mediana_global_ = float(y.median())
        self.tabelas_ = []
        for colunas in self.niveis:
            estat = y.groupby(self._chaves(X, colunas)).agg(["median", "size"])
            estat = estat[estat["size"] >= self.min_obs]
            self.tabelas_.append((tuple(colunas), estat["median"].to_dict()))
        return self

    def predict(self, X):
        pred = pd.Series(np.nan, index=X.index, dtype=float)
        for colunas, tabela in self.tabelas_:
            candidato = self._chaves(X, colunas).map(tabela)
            pred = pred.fillna(candidato)
        return pred.fillna(self.mediana_global_).to_numpy()


def _ridge():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def construir_modelos():
    """nome -> (fabrica do modelo, colunas usadas, nivel)."""
    return {
        "mediana_global": (lambda: DummyRegressor(strategy="median"), ["idade"], "1 - ingenuo"),
        "media_global": (lambda: DummyRegressor(strategy="mean"), ["idade"], "1 - ingenuo"),
        "mediana_posicao": (
            lambda: MedianaPorGrupo(niveis=(("posicao",),)), ["posicao"], "2 - regra de dominio"
        ),
        "mediana_posicao_idade": (
            lambda: MedianaPorGrupo(), ["posicao", "faixa_idade"], "2 - regra de dominio"
        ),
        "ridge_perfil": (_ridge, COLUNAS_PERFIL, "3 - modelo simples"),
        "ridge_completo": (_ridge, COLUNAS_COMPLETO, "referencia"),
    }


def metricas_em_euros(y_log_real, y_log_pred):
    y_real = np.exp(y_log_real)
    y_pred = np.exp(y_log_pred)
    return {
        "MAE_eur": mean_absolute_error(y_real, y_pred),
        "RMSE_eur": mean_squared_error(y_real, y_pred) ** 0.5,
        "R2_log": r2_score(y_log_real, y_log_pred),
        "MAE_log": mean_absolute_error(y_log_real, y_log_pred),
    }


def avaliar_modelos(df: pd.DataFrame, modelos: dict) -> pd.DataFrame:
    """Mesmas validacoes para todos os modelos: K-Fold aleatorio, GroupKFold por jogador
    e holdout temporal (treino = temporada completa, teste = temporada incompleta)."""
    y = df[ALVO]
    esquemas = {
        "kfold_5_aleatorio": (KFold(n_splits=5, shuffle=True, random_state=SEED), None),
        "groupkfold_5": (GroupKFold(n_splits=5), df[GRUPO]),
    }
    treino = df["temporada_completa"].to_numpy()
    teste = ~treino

    linhas = []
    for nome, (fabrica, colunas, nivel) in modelos.items():
        X = df[colunas]
        for validacao, (cv, grupos) in esquemas.items():
            pred = cross_val_predict(fabrica(), X, y, cv=cv, groups=grupos)
            linhas.append({"nivel": nivel, "modelo": nome, "validacao": validacao,
                           "n_avaliado": len(y), **metricas_em_euros(y.to_numpy(), pred)})

        modelo = fabrica().fit(X[treino], y[treino])
        pred = modelo.predict(X[teste])
        linhas.append({"nivel": nivel, "modelo": nome, "validacao": "holdout_temporal",
                       "n_avaliado": int(teste.sum()), **metricas_em_euros(y[teste].to_numpy(), pred)})

    return pd.DataFrame(linhas)[
        ["validacao", "nivel", "modelo", "n_avaliado", "MAE_eur", "RMSE_eur", "R2_log", "MAE_log"]
    ]
