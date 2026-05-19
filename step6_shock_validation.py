"""
causal_pipeline/step6_shock_validation.py
══════════════════════════════════════════
Шаг 6: Валидация каузальных методов на естественных экспериментах

Методология (по тексту диссертации):
  Шаг 1. Определение Y и горизонта анализа (pre/post периоды).
  Шаг 2. Оценка контрфактуала: обучение на pre-period, прогноз на post-period.
  Шаг 3. Вычисление причинного эффекта: θ̂_t = Y_t^obs − Y_t^cf
  Шаг 4. Экспертная оценка направления: ожидаемый знак θ̂.
  Шаг 5. Проверка: знак, лаг, значимость.
  Шаг 6. Hit Ratio + сравнение точности в периоды шоков vs спокойные периоды.

Методы соответствуют шагам 3 и 4 пайплайна:
  Каузальные (step3): SCM/BSTS, DML, VAR+Granger, IV-2SLS
  Baseline   (step4): ARIMA, Prophet, RandomForest, LSTM

Естественные эксперименты (Таблица 2.5):
  ─ Повышения ставки ФРС: 16.03.22, 04.05.22, 15.06.22
  ─ Начало СВО:           24.02.22
  ─ Сокращение ОПЕК+:     05.10.22
  ─ Крах SVB:             10.03.23
  ─ Халвинг BTC:          11.05.20, 19.04.24

Выходы:
  results/step6_shock_effects.json        — θ̂ по каждому событию и методу
  results/step6_hit_ratio.json            — Hit Ratio по методам
  results/step6_shock_vs_calm.json        — RMSE в шоковые vs спокойные периоды
  figures/shock_{event_id}.png            — факт vs контрфактуал
  figures/shock_hit_ratio.png             — Hit Ratio bar chart
  figures/shock_vs_calm_rmse.png          — сравнение точности шок/спокойствие
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from step1_dag_definitions import ALL_DAGS
from step2_variable_selection import load_dataset

DATASETS_DIR = Path("datasets")
RESULTS_DIR  = Path("results")
FIGURES_DIR  = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОДЫ — соответствие шагам 3 и 4
# ══════════════════════════════════════════════════════════════════════════════
#
# Каузальные (step3): SCM, DML, VAR_Granger, IV_2SLS
# Baseline   (step4): ARIMA, Prophet, RandomForest, LSTM
#
# Для построения контрфактуала используется упрощённая реализация каждого
# метода: обучение на pre-period, прогноз на post-period без обновления.
# Это соответствует логике «что было бы без шока».

CAUSAL_METHODS   = ["SCM", "DML", "VAR_Granger", "IV_2SLS"]
BASELINE_METHODS = ["ARIMA", "Prophet", "RandomForest", "LSTM"]
ALL_METHODS      = CAUSAL_METHODS + BASELINE_METHODS

METHOD_LABELS = {
    "SCM":          "SCM/BSTS",
    "DML":          "DML",
    "VAR_Granger":  "VAR+Granger",
    "IV_2SLS":      "IV-2SLS",
    "ARIMA":        "ARIMA",
    "Prophet":      "Prophet",
    "RandomForest": "RandomForest",
    "LSTM":         "LSTM",
}

# Цвета: 8 максимально контрастных цветов (ColorBrewer Set1 + Dark2)
# Каузальные: насыщенные, тёмные линии
# Baseline: тёплые/нейтральные, более тонкие линии
METHOD_COLORS = {
    "SCM":          "#1B4F72",   # тёмно-синий
    "DML":          "#C0392B",   # тёмно-красный
    "VAR_Granger":  "#1E8449",   # тёмно-зелёный
    "IV_2SLS":      "#7D3C98",   # фиолетовый
    "ARIMA":        "#E67E22",   # оранжевый
    "Prophet":      "#17A589",   # бирюзовый
    "RandomForest": "#2E86C1",   # синий (отличается от тёмно-синего SCM)
    "LSTM":         "#839192",   # серый
}

# Стиль линий: каузальные — сплошные утолщённые, baseline — штриховые
METHOD_LS = {
    "SCM":          "-",
    "DML":          "-",
    "VAR_Granger":  "-",
    "IV_2SLS":      "-",
    "ARIMA":        "--",
    "Prophet":      "-.",
    "RandomForest": (0, (5, 2)),  # пунктир с крупными штрихами
    "LSTM":         ":",
}

# Маркеры на линиях для дополнительной идентификации
METHOD_MARKER = {
    "SCM":          "o",
    "DML":          "s",
    "VAR_Granger":  "^",
    "IV_2SLS":      "D",
    "ARIMA":        "v",
    "Prophet":      "P",
    "RandomForest": "X",
    "LSTM":         "*",
}

# Толщина линий
METHOD_LW = {
    "SCM":          2.2,
    "DML":          2.2,
    "VAR_Granger":  2.2,
    "IV_2SLS":      2.2,
    "ARIMA":        1.5,
    "Prophet":      1.5,
    "RandomForest": 1.5,
    "LSTM":         1.5,
}

METHOD_TYPE = {m: "Causal"   for m in CAUSAL_METHODS}
METHOD_TYPE.update({m: "Baseline" for m in BASELINE_METHODS})


# ══════════════════════════════════════════════════════════════════════════════
# РЕЕСТР ЕСТЕСТВЕННЫХ ЭКСПЕРИМЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShockEvent:
    event_id:      str
    name:          str
    shock_type:    str
    event_date:    str
    targets:       List[str]
    pre_days:      int
    post_days:     int
    expected_sign: Dict[str, int]
    expected_lag:  int
    exogeneity:    str
    notes:         str = ""


SHOCK_EVENTS: List[ShockEvent] = [
    ShockEvent(
        event_id="fed_rate_mar22", name="ФРС +25bp (март 2022)",
        shock_type="monetary_policy", event_date="2022-03-16",
        targets=["EURUSD", "USDJPY", "Gold", "BTC"],
        pre_days=120, post_days=10,
        expected_sign={"EURUSD": +1, "USDJPY": +1, "Gold": +1, "BTC": +1},
        expected_lag=0,
        exogeneity="Решение ФРС обусловлено макро-целями, не зависит от краткосрочных движений активов",
    ),
    ShockEvent(
        event_id="fed_rate_may22", name="ФРС +50bp (май 2022)",
        shock_type="monetary_policy", event_date="2022-05-04",
        targets=["EURUSD", "USDJPY", "Gold", "BTC"],
        pre_days=120, post_days=10,
        expected_sign={"EURUSD": -1, "USDJPY": -1, "Gold": +1, "BTC": -1},
        expected_lag=0,
        exogeneity="Аналогично мартовскому решению ФРС",
    ),
    ShockEvent(
        event_id="fed_rate_jun22", name="ФРС +75bp (июнь 2022)",
        shock_type="monetary_policy", event_date="2022-06-15",
        targets=["EURUSD", "USDJPY", "Gold", "BTC"],
        pre_days=120, post_days=10,
        expected_sign={"EURUSD": -1, "USDJPY": +1, "Gold": +1, "BTC": -1},
        expected_lag=0,
        exogeneity="Наиболее агрессивное повышение цикла",
    ),
    ShockEvent(
        event_id="russia_ukraine_feb22", name="Начало СВО (февраль 2022)",
        shock_type="geopolitical", event_date="2022-02-24",
        targets=["WTI_oil", "NatGas", "Gold", "BTC"],
        pre_days=120, post_days=15,
        expected_sign={"WTI_oil": +1, "NatGas": -1, "Gold": +1, "BTC": +1},
        expected_lag=0,
        exogeneity="Военное вторжение — внешнее событие, не вызванное движением цен",
        notes="Немедленный spike WTI и газа, Gold как safe haven",
    ),
    ShockEvent(
        event_id="opec_cut_oct22", name="Сокращение добычи ОПЕК+ (октябрь 2022)",
        shock_type="supply_shock", event_date="2022-10-05",
        targets=["WTI_oil"],
        pre_days=100, post_days=15,
        expected_sign={"WTI_oil": +1},
        expected_lag=1,
        exogeneity="Решение картеля ОПЕК+, не зависящее от текущей цены WTI",
    ),
    ShockEvent(
        event_id="svb_collapse_mar23", name="Крах Silicon Valley Bank (март 2023)",
        shock_type="liquidity_crisis", event_date="2023-03-10",
        targets=["Gold", "BTC", "ETH"],
        pre_days=120, post_days=10,
        expected_sign={"Gold": +1, "BTC": +1, "ETH": +1},
        expected_lag=0,
        exogeneity="Банкротство банка — экзогенный шок ликвидности",
        notes="BTC рос вопреки risk-off: crypto-specific narrative",
    ),
    ShockEvent(
        event_id="btc_halving_may20", name="Халвинг биткоина (май 2020)",
        shock_type="structural_shift", event_date="2020-05-11",
        targets=["BTC", "ETH"],
        pre_days=180, post_days=90,
        expected_sign={"BTC": +1, "ETH": +1},
        expected_lag=30,
        exogeneity="Заложен в протоколе блокчейна, полностью экзогенен",
        notes="Supply shock: вознаграждение майнеров уменьшается вдвое",
    ),
    ShockEvent(
        event_id="btc_halving_apr24", name="Халвинг биткоина (апрель 2024)",
        shock_type="structural_shift", event_date="2024-04-19",
        targets=["BTC", "ETH"],
        pre_days=180, post_days=90,
        expected_sign={"BTC": +1, "ETH": +1},
        expected_lag=30,
        exogeneity="Четвёртый халвинг, вознаграждение 6.25→3.125 BTC",
    ),
]

SHOCK_INDEX: Dict[str, ShockEvent] = {e.event_id: e for e in SHOCK_EVENTS}


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И РАЗРЕЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def load_shock_data(
    target: str, event_date: str,
    pre_days: int, post_days: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    dag = ALL_DAGS.get(target)
    if dag is None:
        return None, None
    df = load_dataset(dag)
    if df.empty or target not in df.columns:
        return None, None

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    event_dt  = pd.Timestamp(event_date)
    pre_start = event_dt - pd.tseries.offsets.BusinessDay(pre_days)
    pre_end   = event_dt - pd.tseries.offsets.BusinessDay(1)
    post_end  = event_dt + pd.tseries.offsets.BusinessDay(post_days)

    df_pre  = df.loc[(df.index >= pre_start) & (df.index <= pre_end)]
    df_post = df.loc[(df.index >= event_dt)  & (df.index <= post_end)]

    if len(df_pre) < 20 or len(df_post) < 1:
        return None, None

    log.info(f"  {target}: pre={len(df_pre)}  post={len(df_post)}")
    return df_pre, df_post


# ══════════════════════════════════════════════════════════════════════════════
# МЕТРИКА ЭФФЕКТА
# ══════════════════════════════════════════════════════════════════════════════

def metrics_shock(y_obs: np.ndarray, y_cf: np.ndarray) -> Dict:
    """θ̂_t = Y_obs − Y_cf. Возвращает характеристики эффекта."""
    effects  = y_obs - y_cf
    mean_eff = float(np.mean(effects))
    rmse_cf  = float(np.sqrt(np.mean(effects ** 2)))
    return {
        "effects":     effects.tolist(),
        "mean_effect": round(mean_eff, 6),
        "cumulative":  round(float(np.sum(effects)), 6),
        "sign":        int(np.sign(mean_eff)) if mean_eff != 0 else 0,
        "abs_mean":    round(abs(mean_eff), 6),
        "pct_change":  round(mean_eff / (abs(y_obs[0]) + 1e-9) * 100, 4),
        "rmse_cf":     round(rmse_cf, 6),  # RMSE контрфактуала (точность метода)
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════════

def _build_features(df: pd.DataFrame, target: str,
                    controls: List[str], n_lags: int = 5) -> pd.DataFrame:
    avail = [c for c in controls if c in df.columns and c != target]
    feat  = pd.DataFrame(index=df.index)
    for lag in range(1, n_lags + 1):
        feat[f"{target}_lag{lag}"] = df[target].shift(lag)
    for col in avail[:12]:
        for lag in range(0, 3):
            key = col if lag == 0 else f"{col}_lag{lag}"
            feat[key] = df[col].shift(lag)
    return feat.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# КАУЗАЛЬНЫЕ МЕТОДЫ (step3): SCM, DML, VAR+Granger, IV-2SLS
# ══════════════════════════════════════════════════════════════════════════════

def cf_scm(train: pd.DataFrame, test: pd.DataFrame,
           target: str, controls: List[str]) -> Dict:
    """
    SCM/BSTS (step3 — Метод 1).
    Байесовская структурная модель: локальный уровень + регрессоры из controls.
    """
    try:
        import statsmodels.api as sm
        
        y_tr = train[target].dropna()
        if len(y_tr) < 10:
            return {"method": "SCM_FAILED", "metrics": {}}
        
        avail = [c for c in controls if c in train.columns and c != target]
        
        # Подготовка exogenous переменных
        if avail:
            X_tr = train[avail].reindex(y_tr.index).ffill().bfill()
            # Убираем колонки с NaN
            X_tr = X_tr.dropna(axis=1, how='any')
            if X_tr.shape[1] > 0:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr.values)
                
                # Для теста
                X_te = test[avail].reindex(test.index).ffill().bfill()
                X_te = X_te[X_tr.columns].dropna(axis=1, how='any')
                if X_te.shape[1] > 0:
                    X_te_s = scaler.transform(X_te.values)
                else:
                    X_te_s = None
            else:
                X_tr_s, X_te_s = None, None
        else:
            X_tr_s, X_te_s = None, None
        
        # Модель с локальным уровнем
        model = sm.tsa.UnobservedComponents(
            y_tr.values, level="local level", exog=X_tr_s
        )
        res = model.fit(disp=False, maxiter=500)
        
        # Прогноз на post-period
        n_te = len(test[target].dropna())
        if X_te_s is not None:
            y_cf = res.forecast(steps=n_te, exog=X_te_s)
        else:
            y_cf = res.forecast(steps=n_te)
        
        y_obs = test[target].dropna().values[:len(y_cf)]
        y_cf = np.array(y_cf)[:len(y_obs)]
        
        return {
            "method": "SCM/BSTS",
            "metrics": metrics_shock(y_obs, y_cf),
            "y_cf": y_cf.tolist(),
            "y_obs": y_obs.tolist(),
            "dates": test.index[:len(y_obs)].strftime("%Y-%m-%d").tolist(),
        }
    except Exception as e:
        log.warning(f"    SCM ошибка: {e}")
        return {"method": "SCM_FAILED", "metrics": {}}


def cf_dml(train: pd.DataFrame, test: pd.DataFrame,
           target: str, causes: List[str], controls: List[str]) -> Dict:
    """
    DML (step3 — Метод 2).
    Partialling-out: εY = Y − E[Y|Z], εX = X − E[X|Z], θ = OLS(εY ~ εX).
    Контрфактуал = baseline_GB(Z_test) + θ × X_test.
    """
    try:
        full    = pd.concat([train, test])
        feat_full = _build_features(full, target, controls + causes)

        feat_tr = feat_full.reindex(train.index).dropna()
        feat_te = feat_full.reindex(test.index).dropna()
        if feat_tr.empty or feat_te.empty:
            return {"method": "DML_SKIPPED", "metrics": {}}

        y_tr = train[target].reindex(feat_tr.index).dropna()
        feat_tr = feat_tr.reindex(y_tr.index)
        cols = feat_tr.columns.intersection(feat_te.columns).tolist()

        # Разделяем на cause-колонки и control-колонки
        cause_cols = [c for c in cols if any(c.startswith(ca) for ca in causes)]
        ctrl_cols  = [c for c in cols if c not in cause_cols]
        if not cause_cols or not ctrl_cols:
            cause_cols, ctrl_cols = cols[:max(1,len(cols)//2)], cols[max(1,len(cols)//2):]

        from sklearn.linear_model import LinearRegression
        scaler = StandardScaler()
        W_tr = scaler.fit_transform(feat_tr[ctrl_cols].values)
        W_te = scaler.transform(feat_te[ctrl_cols].values)
        T_tr = feat_tr[cause_cols].values
        T_te = feat_te[cause_cols].values

        m_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
        m_y.fit(W_tr, y_tr.values)
        res_y = y_tr.values - m_y.predict(W_tr)

        res_T = np.zeros_like(T_tr, dtype=float)
        mt_list = []
        for j in range(T_tr.shape[1]):
            mt = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=j)
            mt.fit(W_tr, T_tr[:, j])
            res_T[:, j] = T_tr[:, j] - mt.predict(W_tr)
            mt_list.append(mt)

        theta = LinearRegression(fit_intercept=False).fit(res_T, res_y).coef_
        baseline = m_y.predict(W_te)
        res_T_te = T_te - np.column_stack([mt.predict(W_te) for mt in mt_list])
        y_cf  = baseline + res_T_te @ theta
        y_obs = test[target].reindex(feat_te.index).values[:len(y_cf)]
        y_cf  = y_cf[:len(y_obs)]

        return {
            "method": "DML",
            "metrics": metrics_shock(y_obs, y_cf),
            "y_cf": y_cf.tolist(), "y_obs": y_obs.tolist(),
            "dates": feat_te.index[:len(y_obs)].strftime("%Y-%m-%d").tolist(),
        }
    except Exception as e:
        log.warning(f"    DML ошибка: {e}")
        return {"method": "DML_FAILED", "metrics": {}}


def cf_var_granger(train: pd.DataFrame, test: pd.DataFrame,
                   target: str, controls: List[str]) -> Dict:
    """
    VAR+Granger (step3 — Метод 3).
    Granger-тест отбирает значимые переменные, VAR обучается на pre-period,
    forecast без обновления = контрфактуал на post-period.
    """
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
        from statsmodels.tsa.stattools import grangercausalitytests

        avail = [c for c in controls if c in train.columns and c != target]

        # Granger-тест: отбираем значимые контроли
        significant = []
        for col in avail[:6]:
            try:
                pair = train[[target, col]].dropna()
                if len(pair) < 30:
                    continue
                res  = grangercausalitytests(pair, maxlag=4, verbose=False)
                min_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, 5))
                if min_p < 0.05:
                    significant.append(col)
            except Exception:
                pass

        if not significant:
            significant = avail[:3]  # fallback: топ-3

        cols   = [target] + significant[:4]
        sub_tr = train[cols].dropna()
        sub_te = test[cols].dropna()

        if len(sub_tr) < 15 or sub_te.empty:
            return {"method": "VAR_SKIPPED", "metrics": {}}

        try:
            best_lag = max(1, VAR(sub_tr).select_order(maxlags=5).aic)
        except Exception:
            best_lag = 2

        fitted = VAR(sub_tr).fit(maxlags=best_lag, ic=None)
        fc     = fitted.forecast(sub_tr.values[-best_lag:], steps=len(sub_te))
        y_cf   = fc[:, 0]
        y_obs  = sub_te[target].values[:len(y_cf)]
        y_cf   = y_cf[:len(y_obs)]

        return {
            "method": f"VAR+Granger(lag={best_lag})",
            "metrics": metrics_shock(y_obs, y_cf),
            "y_cf": y_cf.tolist(), "y_obs": y_obs.tolist(),
            "dates": sub_te.index[:len(y_obs)].strftime("%Y-%m-%d").tolist(),
            "significant_vars": significant,
        }
    except Exception as e:
        log.warning(f"    VAR+Granger ошибка: {e}")
        return {"method": "VAR_FAILED", "metrics": {}}


def cf_iv2sls(train: pd.DataFrame, test: pd.DataFrame,
              target: str, causes: List[str],
              instruments: List[str], controls: List[str]) -> Dict:
    """
    IV-2SLS (step3 — Метод 4).
    Стадия 1: X̂ = f(instruments, controls).
    Стадия 2: Y = g(X̂, controls).
    Контрфактуал: прогноз из стадии 2 на post-period.
    """
    try:
        from sklearn.linear_model import LinearRegression

        full = pd.concat([train, test])
        feat_full = _build_features(full, target, causes + controls + instruments)

        feat_tr = feat_full.reindex(train.index).dropna()
        feat_te = feat_full.reindex(test.index).dropna()
        if feat_tr.empty or feat_te.empty:
            return {"method": "IV_SKIPPED", "metrics": {}}

        y_tr  = train[target].reindex(feat_tr.index).dropna()
        feat_tr = feat_tr.reindex(y_tr.index)
        cols  = feat_tr.columns.intersection(feat_te.columns).tolist()

        endog_cols = [c for c in cols if any(c.startswith(ca) for ca in causes)][:3]
        inst_cols  = [c for c in cols if any(c.startswith(iv) for iv in instruments)][:3]
        exog_cols  = [c for c in cols if c not in endog_cols][:8]

        if not endog_cols:
            endog_cols = cols[:2]
        if not inst_cols:
            inst_cols = endog_cols

        scaler = StandardScaler()
        W_tr   = scaler.fit_transform(feat_tr[exog_cols].values) if exog_cols else np.ones((len(feat_tr), 1))
        W_te   = scaler.transform(feat_te[exog_cols].values)     if exog_cols else np.ones((len(feat_te), 1))
        T_tr   = feat_tr[endog_cols].values
        T_te   = feat_te[endog_cols].values
        Z_tr   = feat_tr[inst_cols].values

        # Стадия 1
        s1 = LinearRegression().fit(np.hstack([Z_tr, W_tr]), T_tr)
        T_hat_tr = s1.predict(np.hstack([Z_tr, W_tr]))
        Z_te     = feat_te[inst_cols].values
        T_hat_te = s1.predict(np.hstack([Z_te, W_te]))

        # Стадия 2
        s2 = LinearRegression().fit(np.hstack([T_hat_tr, W_tr]), y_tr.values)
        y_cf  = s2.predict(np.hstack([T_hat_te, W_te]))
        y_obs = test[target].reindex(feat_te.index).values[:len(y_cf)]
        y_cf  = y_cf[:len(y_obs)]

        return {
            "method": "IV-2SLS",
            "metrics": metrics_shock(y_obs, y_cf),
            "y_cf": y_cf.tolist(), "y_obs": y_obs.tolist(),
            "dates": feat_te.index[:len(y_obs)].strftime("%Y-%m-%d").tolist(),
        }
    except Exception as e:
        log.warning(f"    IV-2SLS ошибка: {e}")
        return {"method": "IV_FAILED", "metrics": {}}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE МЕТОДЫ (step4): ARIMA, Prophet, RandomForest, LSTM
# ══════════════════════════════════════════════════════════════════════════════

def cf_arima(train: pd.DataFrame, test: pd.DataFrame, target: str) -> Dict:
    """ARIMA (step4 — Baseline 1). Авторегрессионный контрфактуал без контролей."""
    try:
        from statsmodels.tsa.stattools import adfuller
        import statsmodels.tsa.arima.model as sm_arima

        s_tr = train[target].dropna()
        s_te = test[target].dropna()
        if len(s_tr) < 20 or s_te.empty:
            return {"method": "ARIMA_SKIPPED", "metrics": {}}

        try:
            d = 0 if adfuller(s_tr, autolag="AIC")[1] < 0.05 else 1
        except Exception:
            d = 1

        try:
            from pmdarima import auto_arima
            model = auto_arima(s_tr, d=d if d == 0 else None,
                               max_d=2, max_p=4, max_q=4,
                               seasonal=False, stepwise=True,
                               error_action="ignore", suppress_warnings=True)
            fc = model.predict(n_periods=len(s_te))
        except Exception:
            res = sm_arima.ARIMA(s_tr, order=(2, d, 2)).fit(
                method_kwargs={"warn_convergence": False})
            fc = res.forecast(len(s_te)).values

        y_cf  = np.array(fc)[:len(s_te)]
        y_obs = s_te.values[:len(y_cf)]
        return {
            "method": "ARIMA",
            "metrics": metrics_shock(y_obs, y_cf),
            "y_cf": y_cf.tolist(), "y_obs": y_obs.tolist(),
            "dates": s_te.index[:len(y_obs)].strftime("%Y-%m-%d").tolist(),
        }
    except Exception as e:
        log.warning(f"    ARIMA ошибка: {e}")
        return {"method": "ARIMA_FAILED", "metrics": {}}


def cf_prophet(train: pd.DataFrame, test: pd.DataFrame,
               target: str, controls: List[str]) -> Dict:
    """Prophet (step4 — Baseline 2). Тренд + сезонность + регрессоры."""
    try:
        try:
            from prophet import Prophet
        except ImportError:
            from fbprophet import Prophet

        s_tr = train[target].dropna()
        s_te = test[target].dropna()

        df_tr = pd.DataFrame({"ds": s_tr.index, "y": s_tr.values}).reset_index(drop=True)
        avail = [c for c in controls if c in train.columns and c != target][:3]
        for col in avail:
            df_tr[col] = train[col].reindex(s_tr.index).ffill().values

        m = Prophet(seasonality_mode="additive",
                    daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, interval_width=0.95)
        for col in avail:
            m.add_regressor(col)
        m.fit(df_tr)

        future = m.make_future_dataframe(periods=len(s_te), freq="B", include_history=False)
        for col in avail:
            future[col] = test[col].reindex(future["ds"]).ffill().values[:len(future)]

        fc    = m.predict(future)["yhat"].values[:len(s_te)]
        y_obs = s_te.values[:len(fc)]
        return {
            "method": "Prophet",
            "metrics": metrics_shock(y_obs, fc),
            "y_cf": fc.tolist(), "y_obs": y_obs.tolist(),
            "dates": s_te.index[:len(y_obs)].strftime("%Y-%m-%d").tolist(),
        }
    except Exception as e:
        log.warning(f"    Prophet ошибка: {e}")
        return {"method": "Prophet_FAILED", "metrics": {}}


def cf_random_forest(train: pd.DataFrame, test: pd.DataFrame,
                     target: str, controls: List[str]) -> Dict:
    """RandomForest (step4 — Baseline 3). ML контрфактуал без каузальной коррекции."""
    avail = [c for c in controls if c in train.columns and c != target]

    def make_feat(df_in):
        feat = pd.DataFrame(index=df_in.index)
        for lag in range(1, 6):
            feat[f"{target}_lag{lag}"] = df_in[target].shift(lag)
        for col in avail[:15]:
            for lag in range(0, 3):
                feat[f"{col}_lag{lag}"] = df_in[col].shift(lag)
        return feat.dropna()

    feat_tr = make_feat(train)
    y_tr    = train[target].reindex(feat_tr.index).dropna()
    feat_tr = feat_tr.reindex(y_tr.index)
    if len(feat_tr) < 20:
        return {"method": "RF_SKIPPED", "metrics": {}}

    full    = pd.concat([train, test])
    feat_te = make_feat(full).reindex(test.index).dropna()
    if feat_te.empty:
        return {"method": "RF_SKIPPED", "metrics": {}}

    cols = feat_tr.columns.intersection(feat_te.columns).tolist()
    rf   = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(feat_tr[cols].values, y_tr.values)
    y_cf  = rf.predict(feat_te[cols].values)
    y_obs = test[target].reindex(feat_te.index).values[:len(y_cf)]
    y_cf  = y_cf[:len(y_obs)]
    return {
        "method": "RandomForest",
        "metrics": metrics_shock(y_obs, y_cf),
        "y_cf": y_cf.tolist(), "y_obs": y_obs.tolist(),
        "dates": feat_te.index[:len(y_obs)].strftime("%Y-%m-%d").tolist(),
    }


def cf_lstm(train: pd.DataFrame, test: pd.DataFrame,
            target: str, controls: List[str], seq_len: int = 20) -> Dict:
    """LSTM (step4 — Baseline 4). Нейросетевой контрфактуал."""
    try:
        import torch
        import torch.nn as nn

        avail = [c for c in controls if c in train.columns and c != target]
        cols  = [target] + avail[:7]
        full  = pd.concat([train, test])
        sub   = full[cols].dropna()
        n_tr  = len(train[cols].dropna())

        scaler = MinMaxScaler()
        data   = scaler.fit_transform(sub.values)

        # Обучающие последовательности из pre-period
        X_s, y_s = [], []
        for i in range(seq_len, n_tr):
            X_s.append(data[i - seq_len:i])
            y_s.append(data[i, 0])
        if len(X_s) < 10:
            return {"method": "LSTM_SKIPPED", "metrics": {}}

        X_t = torch.FloatTensor(np.array(X_s))
        y_t = torch.FloatTensor(np.array(y_s))

        class LSTMModel(nn.Module):
            def __init__(self, n_feat):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, 32, 1, batch_first=True)
                self.fc   = nn.Linear(32, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        model = LSTMModel(len(cols))
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(20):
            optim.zero_grad()
            loss = nn.MSELoss()(model(X_t), y_t)
            loss.backward()
            optim.step()

        # Контрфактуал: прогноз на post-period
        model.eval()
        preds = []
        context = data[n_tr - seq_len:n_tr].copy()
        n_post = len(test[cols].dropna())
        for _ in range(n_post):
            inp = torch.FloatTensor(context[-seq_len:]).unsqueeze(0)
            with torch.no_grad():
                pred_sc = model(inp).item()
            preds.append(pred_sc)
            new_row = context[-1].copy()
            new_row[0] = pred_sc
            context = np.vstack([context, new_row])

        pad = np.zeros((len(preds), len(cols)))
        pad[:, 0] = preds
        y_cf = scaler.inverse_transform(pad)[:, 0]

        y_obs = test[target].dropna().values[:len(y_cf)]
        y_cf  = y_cf[:len(y_obs)]
        dates = test.index[:len(y_obs)].strftime("%Y-%m-%d").tolist()

        return {
            "method": "LSTM",
            "metrics": metrics_shock(y_obs, y_cf),
            "y_cf": y_cf.tolist(), "y_obs": y_obs.tolist(), "dates": dates,
        }
    except Exception as e:
        log.warning(f"    LSTM ошибка: {e}")
        return {"method": "LSTM_FAILED", "metrics": {}}


# ══════════════════════════════════════════════════════════════════════════════
# АНАЛИЗ ОДНОГО СОБЫТИЯ (Шаги 1–5)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_shock(event: ShockEvent, var_selection: Optional[Dict] = None) -> Dict:
    log.info(f"\n{'─'*60}")
    log.info(f"  {event.name}  [{event.event_date}]  тип={event.shock_type}")
    log.info(f"{'─'*60}")

    event_results = {
        "event_id": event.event_id, "name": event.name,
        "shock_type": event.shock_type, "event_date": event.event_date,
        "exogeneity": event.exogeneity, "targets": {},
    }

    for target in event.targets:
        log.info(f"\n  ▸ {target}")
        df_pre, df_post = load_shock_data(
            target, event.event_date, event.pre_days, event.post_days)
        if df_pre is None:
            event_results["targets"][target] = {"error": "no_data"}
            continue

        controls, causes, instruments = [], [], []
        if var_selection and target in var_selection:
            info        = var_selection[target]
            controls    = info.get("causal_union", info.get("causal_expert", []))
            causes      = info.get("causes", [])
            instruments = info.get("instruments", [])

        expected = event.expected_sign.get(target, 0)

        # ── Запуск всех 8 методов ────────────────────────────────────────────
        methods_results = {
            "SCM":          cf_scm(df_pre, df_post, target, controls),
            "DML":          cf_dml(df_pre, df_post, target, causes, controls),
            "VAR_Granger":  cf_var_granger(df_pre, df_post, target, controls),
            "IV_2SLS":      cf_iv2sls(df_pre, df_post, target, causes, instruments, controls),
            "ARIMA":        cf_arima(df_pre, df_post, target),
            "Prophet":      cf_prophet(df_pre, df_post, target, controls),
            "RandomForest": cf_random_forest(df_pre, df_post, target, controls),
            "LSTM":         cf_lstm(df_pre, df_post, target, controls),
        }

        target_summary = {
            "expected_sign": expected, "expected_lag": event.expected_lag,
            "n_pre": len(df_pre), "n_post": len(df_post),
            "methods": {},
        }

        for mname, res in methods_results.items():
            m = res.get("metrics", {})
            if not m:
                target_summary["methods"][mname] = {"status": "skipped"}
                continue
            sign_ok = (m.get("sign", 0) == expected) if expected != 0 else None
            eff     = m.get("mean_effect", 0)
            cum     = m.get("cumulative", 0)
            target_summary["methods"][mname] = {
                "method_label": res.get("method", mname),
                "mean_effect":  m.get("mean_effect"),
                "cumulative":   m.get("cumulative"),
                "pct_change":   m.get("pct_change"),
                "sign":         m.get("sign"),
                "sign_correct": sign_ok,
                "rmse_cf":      m.get("rmse_cf"),
                "dates":        res.get("dates", []),
                "y_obs":        res.get("y_obs", []),
                "y_cf":         res.get("y_cf",  []),
            }
            ok_str = "✓" if sign_ok else ("✗" if sign_ok is False else "?")
            tp     = METHOD_TYPE.get(mname, "?")
            log.info(f"    [{tp:<8}] {mname:<14} "
                     f"θ̂={eff:+.4f}  sign={'+'if m.get('sign',0)>0 else '-'}  "
                     f"exp={'+'if expected>0 else'-' if expected<0 else'0'}  {ok_str}")

        event_results["targets"][target] = target_summary

    return event_results


# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 6А: HIT RATIO
# ══════════════════════════════════════════════════════════════════════════════

def compute_hit_ratio(all_shock_results: List[Dict]) -> Dict:
    """
    Hit Ratio = правильных знаков / всего случаев.
    Разбивка по типам шоков и по каузальный/baseline.
    """
    results = {m: {"correct": 0, "total": 0, "effects": [], "by_type": {}}
               for m in ALL_METHODS}

    for event_res in all_shock_results:
        shock_type = event_res.get("shock_type", "unknown")
        for target, tdata in event_res.get("targets", {}).items():
            if "error" in tdata:
                continue
            expected = tdata.get("expected_sign", 0)
            if expected == 0:
                continue
            for mname in ALL_METHODS:
                m_res = tdata.get("methods", {}).get(mname, {})
                if m_res.get("status") == "skipped":
                    continue
                sign_ok = m_res.get("sign_correct")
                if sign_ok is None:
                    continue
                results[mname]["total"] += 1
                if sign_ok:
                    results[mname]["correct"] += 1
                eff = m_res.get("mean_effect")
                if eff is not None:
                    results[mname]["effects"].append(abs(eff))
                if shock_type not in results[mname]["by_type"]:
                    results[mname]["by_type"][shock_type] = {"correct": 0, "total": 0}
                results[mname]["by_type"][shock_type]["total"] += 1
                if sign_ok:
                    results[mname]["by_type"][shock_type]["correct"] += 1

    summary = {}
    for mname, data in results.items():
        t = data["total"]
        c = data["correct"]
        summary[mname] = {
            "method_type":     METHOD_TYPE.get(mname, "?"),
            "hit_ratio":       round(c / t, 4) if t > 0 else None,
            "correct": c, "total": t,
            "mean_abs_effect": round(np.mean(data["effects"]), 6) if data["effects"] else None,
            "by_shock_type":   {st: round(v["correct"]/v["total"], 4)
                                for st, v in data["by_type"].items() if v["total"] > 0},
        }

    log.info("\n" + "═"*60)
    log.info("  HIT RATIO по методам (каузальные vs baseline)")
    log.info("═"*60)
    for tp in ["Causal", "Baseline"]:
        log.info(f"  {'─'*30} {tp} {'─'*10}")
        for mname, s in [(m, summary[m]) for m in ALL_METHODS
                         if summary[m]["method_type"] == tp]:
            hr = s["hit_ratio"]
            log.info(f"  {mname:<16}  HR={hr:.3f}  ({s['correct']}/{s['total']})")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 6Б: СРАВНЕНИЕ ТОЧНОСТИ ШОКОВЫЕ vs СПОКОЙНЫЕ ПЕРИОДЫ
# ══════════════════════════════════════════════════════════════════════════════

def compute_shock_vs_calm(
    all_shock_results: List[Dict],
    var_selection: Optional[Dict] = None,
    calm_window_days: int = 60,
) -> Dict:
    """
    Сравнивает RMSE контрфактуала в шоковые и спокойные периоды.
    """
    log.info("\n" + "═"*60)
    log.info("  СРАВНЕНИЕ: точность в шоковые vs спокойные периоды")
    log.info("═"*60)

    results_by_method: Dict[str, Dict] = {m: {
        "rmse_shock": [], "rmse_calm": [], "ratio": [],
        "method_type": METHOD_TYPE.get(m, "?"),
    } for m in ALL_METHODS}

    per_event_detail = []

    for event_res in all_shock_results:
        event_id   = event_res["event_id"]
        event_date = event_res["event_date"]
        event      = SHOCK_INDEX.get(event_id)
        if event is None:
            continue

        for target, tdata in event_res.get("targets", {}).items():
            if "error" in tdata:
                continue

            # Загружаем полный датасет для calm-period
            dag = ALL_DAGS.get(target)
            if dag is None:
                continue
            df_full = load_dataset(dag)
            if df_full.empty or target not in df_full.columns:
                continue
            df_full.index = pd.to_datetime(df_full.index)
            df_full = df_full.sort_index()

            controls, causes, instruments = [], [], []
            if var_selection and target in var_selection:
                info        = var_selection[target]
                controls    = info.get("causal_union", info.get("causal_expert", []))
                causes      = info.get("causes", [])
                instruments = info.get("instruments", [])

            event_dt = pd.Timestamp(event_date)

            # Calm-period: за calm_window_days дней до начала pre-period
            pre_start    = event_dt - pd.tseries.offsets.BusinessDay(event.pre_days)
            calm_end     = pre_start - pd.tseries.offsets.BusinessDay(1)
            calm_start   = calm_end - pd.tseries.offsets.BusinessDay(calm_window_days)
            calm_pre_end = calm_start - pd.tseries.offsets.BusinessDay(1)
            calm_pre_st  = calm_pre_end - pd.tseries.offsets.BusinessDay(event.pre_days)

            df_calm_train = df_full.loc[(df_full.index >= calm_pre_st) &
                                        (df_full.index <= calm_pre_end)]
            df_calm_test  = df_full.loc[(df_full.index >= calm_start) &
                                        (df_full.index <= calm_end)]

            if len(df_calm_train) < 20 or df_calm_test.empty:
                continue

            event_row = {"event_id": event_id, "target": target, "methods": {}}

            for mname in ALL_METHODS:
                # RMSE в шоковый период (из уже вычисленных результатов)
                m_shock = tdata.get("methods", {}).get(mname, {})
                rmse_sh = m_shock.get("rmse_cf")

                # RMSE в спокойный период: запускаем тот же метод на calm данных
                try:
                    if mname == "SCM":
                        r_calm = cf_scm(df_calm_train, df_calm_test, target, controls)
                    elif mname == "DML":
                        r_calm = cf_dml(df_calm_train, df_calm_test, target, causes, controls)
                    elif mname == "VAR_Granger":
                        r_calm = cf_var_granger(df_calm_train, df_calm_test, target, controls)
                    elif mname == "IV_2SLS":
                        r_calm = cf_iv2sls(df_calm_train, df_calm_test, target,
                                           causes, instruments, controls)
                    elif mname == "ARIMA":
                        r_calm = cf_arima(df_calm_train, df_calm_test, target)
                    elif mname == "Prophet":
                        r_calm = cf_prophet(df_calm_train, df_calm_test, target, controls)
                    elif mname == "RandomForest":
                        r_calm = cf_random_forest(df_calm_train, df_calm_test, target, controls)
                    elif mname == "LSTM":
                        r_calm = cf_lstm(df_calm_train, df_calm_test, target, controls)
                    else:
                        r_calm = {"metrics": {}}
                    rmse_cl = r_calm.get("metrics", {}).get("rmse_cf")
                except Exception as e:
                    log.warning(f"    calm {mname} {target}: {e}")
                    rmse_cl = None

                ratio = None
                if rmse_sh is not None and rmse_cl is not None and rmse_cl > 1e-9:
                    ratio = round(rmse_sh / rmse_cl, 4)
                    results_by_method[mname]["rmse_shock"].append(rmse_sh)
                    results_by_method[mname]["rmse_calm"].append(rmse_cl)
                    results_by_method[mname]["ratio"].append(ratio)

                event_row["methods"][mname] = {
                    "rmse_shock": rmse_sh, "rmse_calm": rmse_cl, "ratio": ratio
                }
                
                # ИСПРАВЛЕНО: безопасное форматирование строки
                rmse_sh_str = f"{rmse_sh:.4f}" if rmse_sh is not None else "N/A"
                rmse_cl_str = f"{rmse_cl:.4f}" if rmse_cl is not None else "N/A"
                ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
                
                log.info(f"    {mname:<14} {target:<10} "
                         f"RMSE_shock={rmse_sh_str:<10}  "
                         f"RMSE_calm={rmse_cl_str:<10}  "
                         f"ratio={ratio_str}")

            per_event_detail.append(event_row)

    # Агрегируем по методам
    aggregated = {}
    for mname, data in results_by_method.items():
        rs = data["rmse_shock"]
        rc = data["rmse_calm"]
        ra = data["ratio"]
        aggregated[mname] = {
            "method_type":     data["method_type"],
            "mean_rmse_shock": round(np.mean(rs), 6) if rs else None,
            "mean_rmse_calm":  round(np.mean(rc), 6) if rc else None,
            "mean_ratio":      round(np.mean(ra), 4) if ra else None,
            "n_observations":  len(ra),
        }

    log.info("\n  Итог: средний Ratio RMSE_shock/RMSE_calm")
    log.info("  (> 1 = хуже при шоке, ≈ 1 = устойчив, < 1 = редкий артефакт)")
    for tp in ["Causal", "Baseline"]:
        log.info(f"  {'─'*20} {tp}")
        for mname, s in [(m, aggregated[m]) for m in ALL_METHODS
                         if aggregated[m]["method_type"] == tp]:
            r = s["mean_ratio"]
            r_str = f"{r:.3f}" if r is not None else "N/A"
            log.info(f"  {mname:<16}  Ratio={r_str}  "
                     f"n={s['n_observations']}")

    return {"aggregated": aggregated, "per_event": per_event_detail}

# ══════════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ (белый фон)
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax_white(ax, title="", xlabel="", ylabel=""):
    """Стиль осей с белым фоном."""
    ax.set_facecolor("white")
    ax.set_title(title, color="#1A1A2E", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color="#444444", fontsize=9)
    ax.set_ylabel(ylabel, color="#444444", fontsize=9)
    ax.tick_params(colors="#333333", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
        sp.set_linewidth(0.8)
    ax.grid(color="#EEEEEE", linewidth=0.6, linestyle="--", alpha=0.8)


def plot_shock(event: ShockEvent, event_res: Dict):
    """Факт vs контрфактуал для каждого target события. Белый фон."""
    targets_data = {t: d for t, d in event_res.get("targets", {}).items()
                    if "error" not in d}
    n = len(targets_data)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{event.name}  |  {event.event_date}  |  {event.shock_type}",
        color="#1A1A2E", fontsize=13, fontweight="bold", y=1.02,
    )

    event_dt = pd.Timestamp(event.event_date)

    for ax_idx, (target, tdata) in enumerate(targets_data.items()):
        ax = axes[0][ax_idx]
        expected = tdata.get("expected_sign", 0)
        exp_str  = "↑ ожидается рост" if expected > 0 else "↓ ожидается снижение"
        _style_ax_white(ax, title=f"{target}  |  {exp_str}",
                        xlabel="Дата", ylabel="Цена / уровень")

        any_obs = None
        for mname in ALL_METHODS:
            m_res = tdata.get("methods", {}).get(mname, {})
            if m_res.get("status") == "skipped" or not m_res.get("y_cf"):
                continue
            dates = pd.to_datetime(m_res.get("dates", []))
            y_cf  = m_res.get("y_cf", [])
            y_obs = m_res.get("y_obs", [])
            if dates.empty or len(y_cf) != len(dates):
                continue

            color    = METHOD_COLORS[mname]
            ls       = METHOD_LS[mname]
            lw       = METHOD_LW[mname]
            marker   = METHOD_MARKER[mname]
            sign_ok  = m_res.get("sign_correct")
            # При неправильном знаке — снижаем alpha
            alpha    = 0.90 if sign_ok else 0.50
            label    = f"{METHOD_LABELS[mname]} {'✓' if sign_ok else '✗'}"

            # Маркеры ставим через каждые N точек чтобы не перегружать
            n_pts    = len(dates)
            every    = max(1, n_pts // 6)

            ax.plot(dates, y_cf,
                    color=color, linewidth=lw, linestyle=ls,
                    marker=marker, markevery=every, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    alpha=alpha, label=label, zorder=4)

            if any_obs is None and len(y_obs) > 0:
                any_obs = (dates, np.array(y_obs))

        # Наблюдаемый ряд
        if any_obs is not None:
            dates_obs, y_obs_arr = any_obs
            ax.plot(dates_obs, y_obs_arr, color="#1A1A2E", linewidth=2.5,
                    zorder=10, label="Факт (Y_obs)")
            # Заливка θ̂ относительно первого контрфактуала
            for m_res in tdata.get("methods", {}).values():
                if m_res.get("status") != "skipped" and m_res.get("y_cf"):
                    first_cf = np.array(m_res["y_cf"])[:len(y_obs_arr)]
                    fill_col = "#2E7D32" if expected > 0 else "#C62828"
                    ax.fill_between(
                        dates_obs[:len(first_cf)], y_obs_arr[:len(first_cf)],
                        first_cf, alpha=0.10, color=fill_col,
                    )
                    break

        # Вертикальная линия события
        ax.axvline(event_dt, color="#D32F2F", linewidth=1.8,
                   linestyle="-", alpha=0.9, label="Событие")

        # Легенда: Line2D — показывает и цвет, и стиль, и маркер
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], color="#1A1A2E", linewidth=2.5,
                   label="Факт (Y_obs)"),
            Line2D([0], [0], color="#D32F2F", linewidth=1.8,
                   linestyle="-", label="Событие"),
        ]

        # Разделитель каузальные
        legend_handles.append(
            Line2D([0], [0], color="none", label="─── Каузальные ───")
        )
        for m in CAUSAL_METHODS:
            mdata = tdata.get("methods", {}).get(m, {})
            if not mdata.get("y_cf"):
                continue
            sign_ok = mdata.get("sign_correct")
            suffix  = " ✓" if sign_ok else " ✗"
            legend_handles.append(Line2D(
                [0], [0],
                color=METHOD_COLORS[m],
                linewidth=METHOD_LW[m],
                linestyle=METHOD_LS[m],
                marker=METHOD_MARKER[m],
                markersize=6,
                markeredgecolor="white", markeredgewidth=0.5,
                label=METHOD_LABELS[m] + suffix,
            ))

        # Разделитель baseline
        legend_handles.append(
            Line2D([0], [0], color="none", label="─── Baseline ───")
        )
        for m in BASELINE_METHODS:
            mdata = tdata.get("methods", {}).get(m, {})
            if not mdata.get("y_cf"):
                continue
            sign_ok = mdata.get("sign_correct")
            suffix  = " ✓" if sign_ok else " ✗"
            legend_handles.append(Line2D(
                [0], [0],
                color=METHOD_COLORS[m],
                linewidth=METHOD_LW[m],
                linestyle=METHOD_LS[m],
                marker=METHOD_MARKER[m],
                markersize=6,
                markeredgecolor="white", markeredgewidth=0.5,
                label=METHOD_LABELS[m] + suffix,
            ))

        ax.legend(handles=legend_handles, fontsize=7, loc="best",
                  facecolor="white", edgecolor="#BBBBBB",
                  framealpha=0.95, ncol=2,
                  handlelength=2.5, handleheight=1.2)

    plt.tight_layout()
    fname = FIGURES_DIR / f"shock_{event.event_id}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    log.info(f"  ✓ {fname.name}")
    plt.close()


def plot_hit_ratio(hit_ratio: Dict):
    """Hit Ratio: каузальные vs baseline, общий и по типам шоков."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Hit Ratio: доля правильных знаков θ̂ = Y_obs − Y_cf",
                 color="#1A1A2E", fontsize=14, fontweight="bold")

    # График 1: общий Hit Ratio
    ax1 = axes[0]
    _style_ax_white(ax1, title="Общий Hit Ratio по методам",
                    ylabel="Hit Ratio (доля правильных знаков)")

    causal_methods  = [m for m in CAUSAL_METHODS
                       if hit_ratio.get(m, {}).get("hit_ratio") is not None]
    baseline_methods = [m for m in BASELINE_METHODS
                        if hit_ratio.get(m, {}).get("hit_ratio") is not None]
    all_m = causal_methods + baseline_methods

    if not all_m:
        ax1.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                 transform=ax1.transAxes, color="#666666")
    else:
        x      = np.arange(len(all_m))
        colors = [METHOD_COLORS[m] for m in all_m]
        hrs    = [hit_ratio[m]["hit_ratio"] or 0 for m in all_m]

        bars = ax1.bar(x, hrs, color=colors, edgecolor="#FFFFFF",
                       linewidth=0.5, alpha=0.85, width=0.6)
        ax1.axhline(0.5, color="#D32F2F", linewidth=1.5, linestyle="--",
                    label="Случайное угадывание (0.5)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([METHOD_LABELS[m] for m in all_m],
                            rotation=30, ha="right", color="#333333", fontsize=9)
        ax1.set_ylim(0, 1.05)

        for bar, hr, mname in zip(bars, hrs, all_m):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{hr:.2f}", ha="center", va="bottom",
                     color="#1A1A2E", fontsize=9, fontweight="bold")

        # Разделитель каузальные/baseline
        if causal_methods and baseline_methods:
            sep = len(causal_methods) - 0.5
            ax1.axvline(sep, color="#666666", linewidth=1.2, linestyle=":")
            ax1.text(sep / 2, 1.01, "Каузальные", ha="center",
                     color="#1565C0", fontsize=9, fontweight="bold",
                     transform=ax1.get_xaxis_transform())
            ax1.text(sep + (len(baseline_methods)) / 2, 1.01, "Baseline",
                     ha="center", color="#E65100", fontsize=9, fontweight="bold",
                     transform=ax1.get_xaxis_transform())

        ax1.legend(fontsize=9, facecolor="white", edgecolor="#CCCCCC")

    # График 2: Hit Ratio по типам шоков
    ax2 = axes[1]
    _style_ax_white(ax2, title="Hit Ratio по типам шоков",
                    ylabel="Hit Ratio")

    shock_types = sorted({
        st for m in all_m
        for st in hit_ratio.get(m, {}).get("by_shock_type", {}).keys()
    })
    if not shock_types:
        ax2.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                 transform=ax2.transAxes, color="#666666")
    else:
        x2     = np.arange(len(shock_types))
        n_meth = len(all_m)
        width  = 0.8 / n_meth if n_meth > 0 else 0.1

        for i, mname in enumerate(all_m):
            vals   = [hit_ratio[mname].get("by_shock_type", {}).get(st, 0) or 0
                      for st in shock_types]
            offset = (i - n_meth / 2 + 0.5) * width
            ax2.bar(x2 + offset, vals, width * 0.92,
                    color=METHOD_COLORS[mname],
                    label=METHOD_LABELS[mname],
                    edgecolor="#FFFFFF", linewidth=0.3, alpha=0.85)

        ax2.set_xticks(x2)
        shock_labels = {
            "monetary_policy": "Монетарная\nполитика",
            "geopolitical":    "Геополитика",
            "supply_shock":    "Шок\nпредложения",
            "liquidity_crisis":"Кризис\nликвидности",
            "structural_shift":"Структурный\nсдвиг",
        }
        ax2.set_xticklabels(
            [shock_labels.get(st, st) for st in shock_types],
            rotation=0, ha="center", color="#333333", fontsize=8,
        )
        ax2.set_ylim(0, 1.05)
        ax2.axhline(0.5, color="#D32F2F", linewidth=1.2, linestyle="--")
        ax2.legend(fontsize=7, facecolor="white", edgecolor="#CCCCCC",
                   ncol=2, loc="upper right")

    plt.tight_layout()
    fname = FIGURES_DIR / "shock_hit_ratio.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    log.info(f"  ✓ {fname.name}")
    plt.close()


def plot_shock_vs_calm(shock_vs_calm: Dict):
    """
    Сравнение RMSE в шоковые vs спокойные периоды.
    Два графика: (1) абсолютные RMSE, (2) Ratio шок/спокойствие.
    Белый фон.
    """
    agg  = shock_vs_calm.get("aggregated", {})
    valid = {m: d for m, d in agg.items()
             if d.get("mean_ratio") is not None and d["n_observations"] > 0}
    if not valid:
        log.warning("  Нет данных для shock_vs_calm графика")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Точность методов: шоковые vs спокойные периоды\n"
        "RMSE контрфактуала (меньше = лучше)",
        color="#1A1A2E", fontsize=13, fontweight="bold",
    )

    methods_ord = [m for m in ALL_METHODS if m in valid]
    colors      = [METHOD_COLORS[m] for m in methods_ord]
    x           = np.arange(len(methods_ord))
    w           = 0.35

    # ── График 1: абсолютные RMSE ─────────────────────────────────────────────
    ax1 = axes[0]
    _style_ax_white(ax1, title="Средний RMSE: шок vs спокойствие",
                    ylabel="RMSE (средний по событиям и активам)")

    rmse_shock = [valid[m]["mean_rmse_shock"] or 0 for m in methods_ord]
    rmse_calm  = [valid[m]["mean_rmse_calm"]  or 0 for m in methods_ord]

    b1 = ax1.bar(x - w/2, rmse_shock, w, label="Шоковый период",
                 color=colors, edgecolor="white", linewidth=0.4, alpha=0.85)
    b2 = ax1.bar(x + w/2, rmse_calm,  w, label="Спокойный период",
                 color=colors, edgecolor="white", linewidth=0.4,
                 alpha=0.45, hatch="//")

    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_LABELS[m] for m in methods_ord],
                        rotation=30, ha="right", color="#333333", fontsize=9)

    # Разделитель каузальные/baseline
    nc = sum(1 for m in methods_ord if m in CAUSAL_METHODS)
    if nc > 0 and nc < len(methods_ord):
        ax1.axvline(nc - 0.5, color="#666666", linewidth=1.2, linestyle=":")
        ax1.text(nc/2 - 0.5, ax1.get_ylim()[1]*0.97, "Каузальные",
                 ha="center", color="#1565C0", fontsize=9, fontweight="bold")
        ax1.text(nc + (len(methods_ord)-nc)/2 - 0.5, ax1.get_ylim()[1]*0.97,
                 "Baseline", ha="center", color="#E65100",
                 fontsize=9, fontweight="bold")

    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor="#888888", label="Шоковый период"),
        Patch(facecolor="#888888", alpha=0.45, hatch="//", label="Спокойный период"),
    ], fontsize=9, facecolor="white", edgecolor="#CCCCCC")

    # ── График 2: Ratio RMSE_shock / RMSE_calm ───────────────────────────────
    ax2 = axes[1]
    _style_ax_white(ax2,
                    title="Ratio = RMSE_shock / RMSE_calm\n"
                          "(> 1 = хуже при шоке, ≈ 1 = устойчив)",
                    ylabel="Ratio")

    ratios = [valid[m]["mean_ratio"] or 0 for m in methods_ord]
    bar_colors = [
        "#C62828" if r > 1.2 else
        "#F57C00" if r > 1.0 else
        "#2E7D32"
        for r in ratios
    ]

    bars2 = ax2.bar(x, ratios, 0.6, color=bar_colors,
                    edgecolor="white", linewidth=0.4, alpha=0.85)
    ax2.axhline(1.0, color="#1A1A2E", linewidth=1.5, linestyle="-",
                label="Ratio = 1 (одинакова точность)")
    ax2.axhline(1.2, color="#D32F2F", linewidth=1.0, linestyle="--",
                label="Ratio = 1.2 (порог ухудшения)")

    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_LABELS[m] for m in methods_ord],
                        rotation=30, ha="right", color="#333333", fontsize=9)

    for bar, ratio, mname in zip(bars2, ratios, methods_ord):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{ratio:.2f}", ha="center", va="bottom",
                 color="#1A1A2E", fontsize=9, fontweight="bold")

    if nc > 0 and nc < len(methods_ord):
        ax2.axvline(nc - 0.5, color="#666666", linewidth=1.2, linestyle=":")

    ax2.legend(fontsize=9, facecolor="white", edgecolor="#CCCCCC")

    # Аннотация интерпретации
    fig.text(0.5, -0.04,
             "Ratio < 1.0 (зелёный): метод точнее в шоковые периоды  |  "
             "1.0–1.2 (оранжевый): умеренное ухудшение  |  "
             "> 1.2 (красный): существенное ухудшение точности при шоках",
             ha="center", fontsize=9, color="#555555",
             transform=fig.transFigure)

    plt.tight_layout()
    fname = FIGURES_DIR / "shock_vs_calm_rmse.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    log.info(f"  ✓ {fname.name}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def run_shock_validation(
    events:            Optional[List[str]] = None,
    var_selection:     Optional[Dict]      = None,
    plot:              bool                = True,
    calm_window_days:  int                 = 60,
) -> Dict:
    """
    Запускает полную валидацию: шаги 1–6 + сравнение шок/спокойствие.

    Параметры
    ----------
    events           : список event_id; None = все 8 событий
    var_selection    : результат step2; None = загружается автоматически
    plot             : строить графики
    calm_window_days : длина спокойного периода для сравнения (торговых дней)
    """
    if var_selection is None:
        sel_path = RESULTS_DIR / "step2_variable_selection.json"
        if sel_path.exists():
            with open(sel_path, encoding="utf-8") as f:
                var_selection = json.load(f)
            log.info(f"Загружен step2: {sel_path}")
        else:
            log.warning("step2 не найден — контроли не используются")
            var_selection = {}

    selected = [e for e in SHOCK_EVENTS
                if events is None or e.event_id in events]

    log.info(f"\n{'═'*60}")
    log.info(f"  ШАГ 6: Валидация на {len(selected)} естественных экспериментах")
    log.info(f"  Методы: {CAUSAL_METHODS} + {BASELINE_METHODS}")
    log.info(f"{'═'*60}")

    # Шаги 1–5
    all_results = []
    for event in selected:
        res = analyze_shock(event, var_selection)
        all_results.append(res)
        if plot:
            plot_shock(event, res)

    # Шаг 6А: Hit Ratio
    hit_ratio = compute_hit_ratio(all_results)
    if plot:
        plot_hit_ratio(hit_ratio)

    # Шаг 6Б: Шок vs спокойный период
    shock_vs_calm = compute_shock_vs_calm(all_results, var_selection, calm_window_days)
    if plot:
        plot_shock_vs_calm(shock_vs_calm)

    # Сериализация
    def _clean(obj):
        if isinstance(obj, dict):   return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
        if isinstance(obj, bool): return obj
        return obj

    with open(RESULTS_DIR / "step6_shock_effects.json", "w", encoding="utf-8") as f:
        json.dump(_clean(all_results), f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "step6_hit_ratio.json", "w", encoding="utf-8") as f:
        json.dump(_clean(hit_ratio), f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "step6_shock_vs_calm.json", "w", encoding="utf-8") as f:
        json.dump(_clean(shock_vs_calm), f, ensure_ascii=False, indent=2)

    log.info(f"\n✓ Сохранено: step6_shock_effects.json, step6_hit_ratio.json, "
             f"step6_shock_vs_calm.json")
    return {"shock_results": all_results, "hit_ratio": hit_ratio,
            "shock_vs_calm": shock_vs_calm}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Шаг 6: Валидация на естественных экспериментах"
    )
    parser.add_argument("--events", nargs="+", default=None,
                        help="event_id для запуска (None = все)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--calm-days", type=int, default=60,
                        help="Длина спокойного периода (торг. дней, по умолч. 60)")
    parser.add_argument("--list", action="store_true",
                        help="Показать список событий и выйти")
    args = parser.parse_args()

    if args.list:
        print("\nДоступные события:")
        for e in SHOCK_EVENTS:
            print(f"  {e.event_id:<28} {e.event_date}  {e.name}")
        print()
    else:
        results = run_shock_validation(
            events=args.events,
            plot=not args.no_plot,
            calm_window_days=args.calm_days,
        )
        print("\n" + "═"*60)
        print("  ИТОГ: Hit Ratio и устойчивость к шокам")
        print("═"*60)
        hr = results["hit_ratio"]
        svc = results["shock_vs_calm"]["aggregated"]
        print(f"\n  {'Метод':<16} {'Тип':<10} {'Hit Ratio':<12} {'Ratio шок/спок'}")
        print(f"  {'─'*55}")
        for tp in ["Causal", "Baseline"]:
            for m in ALL_METHODS:
                if hr.get(m, {}).get("method_type") != tp:
                    continue
                h  = hr[m]["hit_ratio"]
                r  = svc.get(m, {}).get("mean_ratio")
                print(f"  {METHOD_LABELS[m]:<16} {tp:<10} "
                      f"{'N/A' if h is None else f'{h:.3f}':<12} "
                      f"{'N/A' if r is None else f'{r:.3f}'}")
