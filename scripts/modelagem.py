"""Funcoes de modelagem usadas nos notebooks 005 (baseline), 006 (modelagem) e 007 (residuos).

Tudo usa o mesmo alvo, o mesmo grupo (jogador) e a mesma semente, entao as metricas
de todas as etapas sao diretamente comparaveis.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from variaveis_contexto import COLUNAS_SUBPOSICAO

# ---------------------------------------------------------------------------
# Alvo, semente e conjuntos de variaveis
# ---------------------------------------------------------------------------
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

# Modelagem final: cada conjunto acrescenta um bloco ao anterior
PARTICIPACAO = ["participacao_minutos", "participacao_jogos", "participacao_gols_clube"]
FORCA_CLUBE = ["pontos_por_jogo_clube", "saldo_gols_por_jogo_clube"]
CLUBE = ["clube_temporada"]

COLUNAS_BASE = list(COLUNAS_COMPLETO)
COLUNAS_PARTICIPACAO = [c for c in COLUNAS_BASE if c not in ("minutos_jogados", "jogos_disputados")] + PARTICIPACAO
COLUNAS_SUBPOS = [c for c in COLUNAS_PARTICIPACAO if c not in COLUNAS_POSICAO] + COLUNAS_SUBPOSICAO
COLUNAS_FORCA = COLUNAS_SUBPOS + FORCA_CLUBE
COLUNAS_FINAIS = COLUNAS_FORCA + CLUBE

BLOCOS = {
    "idade (idade + idade2)": ["idade", "idade2"],
    "subposicao (sub_*)": COLUNAS_SUBPOSICAO,
    "pe preferencial (pe_*)": ["pe_ambos", "pe_esquerdo", "pe_direito"],
    "contrato (anos + sem_info)": ["anos_para_fim_contrato", "sem_info_contrato"],
    "participacao (minutos + jogos)": ["participacao_minutos", "participacao_jogos"],
    "forca do clube (pontos + saldo)": FORCA_CLUBE,
    "clube (target encoding)": CLUBE,
}


# ---------------------------------------------------------------------------
# Baseline 
# ---------------------------------------------------------------------------
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


def _ridge_baseline():
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
        "ridge_perfil": (_ridge_baseline, COLUNAS_PERFIL, "3 - modelo simples"),
        "ridge_completo": (_ridge_baseline, COLUNAS_COMPLETO, "referencia"),
    }


# ---------------------------------------------------------------------------
# Modelagem final 
# ---------------------------------------------------------------------------
def _target_encoder():
    # A semente vai no `cv` (validacao cruzada interna do TargetEncoder)
    return TargetEncoder(target_type="continuous", cv=KFold(n_splits=5, shuffle=True, random_state=SEED))


def _etapas_preparo(colunas, escalar):
    """Clube vira numero com TargetEncoder (media do alvo no clube, so com o treino,
    com validacao cruzada interna e suavizacao para clubes com poucos jogadores).
    Clubes que nao aparecem no treino recebem a media geral."""
    etapas = []
    if "clube_temporada" in colunas:
        etapas.append(("clube", ColumnTransformer(
            [("te", _target_encoder(), ["clube_temporada"])],
            remainder="passthrough", verbose_feature_names_out=False,
        ).set_output(transform="pandas")))
    etapas.append(("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")))
    if escalar:
        etapas.append(("scaler", StandardScaler().set_output(transform="pandas")))
    return etapas


def _fabrica(colunas, modelo, escalar):
    def _f():
        return Pipeline(_etapas_preparo(colunas, escalar) + [("modelo", modelo())])
    return _f


def _ridge():
    return Ridge(alpha=1.0)


def _elasticnet():
    return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000, random_state=SEED)


def _random_forest():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED)


def _gradient_boosting():
    return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                                     min_samples_leaf=5, random_state=SEED)


def _hist_gradient_boosting():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, l2_regularization=1.0, random_state=SEED)


def construir_modelos_ablacao():
    """Ridge com um bloco novo de cada vez: mostra quanto cada melhoria acrescenta."""
    return {
        "ridge_completo": (_fabrica(COLUNAS_BASE, _ridge, True), COLUNAS_BASE, "baseline (005)"),
        "ridge_+participacao": (_fabrica(COLUNAS_PARTICIPACAO, _ridge, True), COLUNAS_PARTICIPACAO, "ablacao"),
        "ridge_+subposicao": (_fabrica(COLUNAS_SUBPOS, _ridge, True), COLUNAS_SUBPOS, "ablacao"),
        "ridge_+forca_clube": (_fabrica(COLUNAS_FORCA, _ridge, True), COLUNAS_FORCA, "ablacao"),
        "ridge_+clube": (_fabrica(COLUNAS_FINAIS, _ridge, True), COLUNAS_FINAIS, "ablacao"),
    }


def construir_modelos_finais():
    """Todos os algoritmos com o conjunto completo de variaveis."""
    c = COLUNAS_FINAIS
    return {
        "ridge": (_fabrica(c, _ridge, True), c, "modelo"),
        "elasticnet": (_fabrica(c, _elasticnet, True), c, "modelo"),
        "random_forest": (_fabrica(c, _random_forest, False), c, "modelo"),
        "gradient_boosting": (_fabrica(c, _gradient_boosting, False), c, "modelo"),
        "hist_gradient_boosting": (_fabrica(c, _hist_gradient_boosting, False), c, "modelo"),
    }


GRADES = {
    "ridge": {"modelo__alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "elasticnet": {"modelo__alpha": [0.001, 0.01, 0.05, 0.1], "modelo__l1_ratio": [0.2, 0.5, 0.8]},
    "random_forest": {
        "modelo__min_samples_leaf": [3, 5, 10],
        "modelo__max_features": [0.33, 0.5, 1.0],
        "modelo__max_depth": [None, 8],
    },
    "gradient_boosting": {
        "modelo__learning_rate": [0.02, 0.05],
        "modelo__max_depth": [2, 3],
        "modelo__n_estimators": [200, 400],
        "modelo__min_samples_leaf": [5, 10],
    },
    "hist_gradient_boosting": {
        "modelo__learning_rate": [0.03, 0.08],
        "modelo__max_leaf_nodes": [7, 15],
        "modelo__min_samples_leaf": [10, 20],
        "modelo__l2_regularization": [0.0, 1.0],
    },
}


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------
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


def metricas_por_fold(df: pd.DataFrame, modelos: dict, n_splits: int = 5) -> pd.DataFrame:
    """R2 e MAE (log) em cada fold do GroupKFold, para ver a variacao entre folds."""
    y = df[ALVO]
    cv = GroupKFold(n_splits=n_splits)
    linhas = []
    for nome, (fabrica, colunas, nivel) in modelos.items():
        X = df[colunas]
        for fold, (tr, te) in enumerate(cv.split(X, y, groups=df[GRUPO])):
            pred = fabrica().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            linhas.append({"modelo": nome, "nivel": nivel, "fold": fold,
                           "R2_log": r2_score(y.iloc[te], pred),
                           "MAE_log": mean_absolute_error(y.iloc[te], pred)})
    return pd.DataFrame(linhas)


def previsoes_oof(df: pd.DataFrame, fabrica, colunas, n_splits: int = 5) -> np.ndarray:
    """Previsoes fora da amostra (out-of-fold) com GroupKFold por jogador.

    Cada linha e prevista por um modelo que nunca viu aquele jogador. E isso que
    permite ler o residuo como "o mercado paga mais/menos do que o perfil explica";
    residuos dentro da amostra seriam artificialmente pequenos (principalmente em arvores).
    """
    return cross_val_predict(fabrica(), df[colunas], df[ALVO], cv=GroupKFold(n_splits=n_splits), groups=df[GRUPO])


def montar_blocos(colunas, blocos):
    """Blocos de variaveis que so fazem sentido juntas (ex.: idade + idade2) + cada coluna
    restante como bloco proprio. Embaralhar `idade` sem `idade2` criaria combinacoes impossiveis."""
    blocos = {k: [c for c in v if c in colunas] for k, v in blocos.items()}
    blocos = {k: v for k, v in blocos.items() if v}
    usadas = {c for v in blocos.values() for c in v}
    return {**blocos, **{c: [c] for c in colunas if c not in usadas}}


def importancia_permutacao(df: pd.DataFrame, fabrica, colunas, n_splits: int = 5,
                           n_repeats: int = 10, blocos=None) -> pd.DataFrame:
    """Importancia por permutacao (em blocos) no fold de teste de cada GroupKFold.

    Mede quanto o MAE (em log) piora ao embaralhar cada bloco de variaveis em dados que
    o modelo nao viu. Funciona igual para modelos lineares e de arvore.
    """
    y = df[ALVO]
    X = df[colunas]
    blocos = montar_blocos(colunas, blocos or {})
    rng = np.random.default_rng(SEED)
    resultados = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=df[GRUPO])):
        modelo = fabrica().fit(X.iloc[tr], y.iloc[tr])
        X_te, y_te = X.iloc[te], y.iloc[te].to_numpy()
        mae_base = mean_absolute_error(y_te, modelo.predict(X_te))
        piora = {}
        for nome, cols in blocos.items():
            difs = []
            for _ in range(n_repeats):
                X_perm = X_te.copy()
                X_perm[cols] = X_te[cols].to_numpy()[rng.permutation(len(X_te))]
                difs.append(mean_absolute_error(y_te, modelo.predict(X_perm)) - mae_base)
            piora[nome] = np.mean(difs)
        resultados.append(pd.Series(piora, name=fold))
    tabela = pd.concat(resultados, axis=1)
    return pd.DataFrame({"importancia_media": tabela.mean(axis=1), "desvio_entre_folds": tabela.std(axis=1)}) \
        .sort_values("importancia_media", ascending=False)


def classificar_residuo(residuo_log: pd.Series, limiar: float) -> pd.Series:
    """Supervalorizado: mercado acima do previsto. Subvalorizado: abaixo. Limiar em log."""
    return pd.Series(np.select([residuo_log > limiar, residuo_log < -limiar],
                               ["supervalorizado", "subvalorizado"], "dentro_do_esperado"),
                     index=residuo_log.index)
