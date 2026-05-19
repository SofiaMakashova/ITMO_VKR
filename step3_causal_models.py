"""
causal_pipeline/step3_causal_models.py
════════════════════════════════════════
Шаг 3: Каузальные методы прогнозирования

Метод 1: CausalImpact  — байесовская структурная модель временных рядов
                          (Google's causalimpact / tfcausalimpact)
Метод 2: DML           — Double Machine Learning (EconML)
                          residual-on-residual регрессия с RF/XGBoost
Метод 3: VAR+Granger   — Vector Autoregression с ограничениями
                          по значимости Granger-тестов
Метод 4: IV-2SLS       — Instrumental Variables (2-stage least squares)
                          для идентификации каузального эффекта

Каждый метод:
  - Обучается на train (80%)
  - Прогнозирует на test (20%)
  - Возвращает: predictions, metrics (RMSE, MAE, MAPE, R²), diagnostics
"""

import warnings
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step1_dag_definitions import ALL_DAGS
from step2_variable_selection import (
    load_dataset, prepare_stationary, run_variable_selection
)

TRAIN_RATIO = 0.80

# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str = "") -> Dict[str, float]:
    """Вычисляет RMSE, MAE, MAPE, R²."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae  = np.mean(np.abs(y_true - y_pred))
    denominator = (np.abs(y_t) + np.abs(y_p)) / 2
    denominator = np.maximum(denominator, 1e-8)  # защита от нуля
    mape = 100 * np.mean(np.abs(y_t - y_p) / denominator)
    r2   = r2_score(y_t, y_p)
    metrics = {"RMSE": round(rmse, 6), "MAE": round(mae, 6),
               "MAPE": round(mape, 4), "R2": round(r2, 4)}
    if label:
        log.info(f"    {label}: RMSE={rmse:.4f}  MAE={mae:.4f}  "
                 f"MAPE={mape:.2f}%  R²={r2:.4f}")
    return metrics


def prepare_ml_data(df: pd.DataFrame, target: str,
                    controls: List[str],
                    causes: List[str],
                    horizon: int = 1,
                    n_lags: int = 5
                    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Формирует матрицу признаков X и целевой вектор y.
    Включает лаги целевого ряда + контрольные переменные.
    """
    feature_cols = list(dict.fromkeys(causes + controls))
    available = [c for c in feature_cols if c in df.columns]

    feat = pd.DataFrame(index=df.index)

    # Лаги целевого ряда
    for lag in range(1, n_lags + 1):
        feat[f"{target}_lag{lag}"] = df[target].shift(lag)

    # Лаги предикторов
    for col in available:
        for lag in range(0, 3):
            key = f"{col}_lag{lag}" if lag > 0 else col
            if key in df.columns:
                feat[key] = df[key]
            else:
                feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    y = df[target].shift(-horizon)  # прогноз на horizon шагов вперёд
    combined = feat.join(y.rename("__target__")).dropna()
    X = combined.drop("__target__", axis=1)
    y_clean = combined["__target__"]
    return X, y_clean

def prepare_ml_data_causal(df: pd.DataFrame, target: str,
                           controls: List[str],
                           causes: List[str],
                           horizon: int = 1,
                           n_lags: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Формирует матрицу признаков X и целевой вектор y.
    Только для каузальных моделей (БЕЗ лагов целевого ряда).
    
    Аргументы:
        df: исходный DataFrame
        target: целевая переменная
        controls: контрольные переменные (Z)
        causes: причинные переменные (X)
        horizon: горизонт прогноза
        n_lags: количество лагов для признаков (игнорируется, фиксировано 3)
    """
    feature_cols = list(dict.fromkeys(causes + controls))
    available = [c for c in feature_cols if c in df.columns]
    
    feat = pd.DataFrame(index=df.index)
    
    # Только лаги предикторов (БЕЗ лагов целевого ряда!)
    for col in available:
        for lag in range(0, 3):  # фиксированно 3 лага
            if lag == 0:
                feat[f"{col}_lag0"] = df[col]
            else:
                feat[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Целевая переменная со сдвигом на horizon вперёд
    y = df[target].shift(-horizon)
    combined = feat.join(y.rename("__target__")).dropna()
    
    X = combined.drop("__target__", axis=1)
    y_clean = combined["__target__"]
    
    # Удаляем колонки с нулевой дисперсией (чтобы избежать проблем в DML)
    X = X.loc[:, X.std() > 1e-8]
    
    return X, y_clean

def train_test_split_ts(X: pd.DataFrame, y: pd.Series,
                        ratio: float = TRAIN_RATIO):
    """Временной split без перемешивания."""
    n = len(X)
    cut = int(n * ratio)
    return (X.iloc[:cut], X.iloc[cut:],
            y.iloc[:cut], y.iloc[cut:])


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 1: SCM-прогноз через DoWhy (Structural Causal Model)
# ══════════════════════════════════════════════════════════════════════════════

def run_scm_dowhy(df: pd.DataFrame, target: str,
                  causes: List[str],
                  controls: List[str],
                  dag: object = None,
                  horizon: int = 1) -> Dict:
    """
    SCM-прогноз (Structural Causal Model) через DoWhy + Ridge/GBM.
    """
    log.info(f"  [SCM-DoWhy] {target}  h={horizon}")
    
    # ── Подготовка данных (каузальная, без лагов целевого ряда) ─────────────
    avail_causes = [c for c in causes if c in df.columns]
    avail_controls = [c for c in controls if c in df.columns]
    
    if not avail_causes:
        avail_causes = [c for c in df.columns 
                        if c != target and not c.startswith(f"{target}_")][:3]
    
    all_features = list(dict.fromkeys(avail_causes + avail_controls))
    if not all_features:
        return {"method": "SCM_SKIPPED", "metrics": {}, "predictions": {}}
    
    # ── Матрица признаков (БЕЗ лагов целевого ряда!) ─────────────────────────
    feat = pd.DataFrame(index=df.index)
    
    # Лаги причин (X) и контролей (Z) — только 2 лага для простоты
    for col in all_features:
        # Текущее значение (lag 0)
        feat[f"{col}"] = df[col]
        # Лаг 1
        feat[f"{col}_lag1"] = df[col].shift(1)
        # Лаг 2
        feat[f"{col}_lag2"] = df[col].shift(2)
    
    # Целевой вектор: прямой h-step прогноз
    feat["__y_h__"] = df[target].shift(-horizon)
    
    # Удаляем строки с NaN
    feat = feat.dropna()
    
    if len(feat) < 60:
        log.warning(f"    Недостаточно данных: {len(feat)} строк")
        return {"method": "SCM_SKIPPED_insufficient_data", "metrics": {}}
    
    # ── Разделение на train/test ─────────────────────────────────────────────
    cut = int(len(feat) * TRAIN_RATIO)
    
    pred_cols = [c for c in feat.columns if c != "__y_h__"]
    X_all = feat[pred_cols].values  # преобразуем в numpy
    y_all = feat["__y_h__"].values
    
    X_tr, X_te = X_all[:cut], X_all[cut:]
    y_tr, y_te = y_all[:cut], y_all[cut:]
    
    # ── Определяем колонки по каузальным ролям ───────────────────────────────
    cause_feat_names = []
    for ca in avail_causes:
        for suffix in ['', '_lag1', '_lag2']:
            name = f"{ca}{suffix}"
            if name in pred_cols:
                cause_feat_names.append(name)
    
    control_feat_names = []
    for co in avail_controls:
        for suffix in ['', '_lag1', '_lag2']:
            name = f"{co}{suffix}"
            if name in pred_cols and name not in cause_feat_names:
                control_feat_names.append(name)
    
    # Backdoor-корректный набор: причинные + контрольные переменные
    backdoor_names = list(dict.fromkeys(cause_feat_names + control_feat_names))
    
    if not backdoor_names:
        backdoor_names = pred_cols[:min(10, len(pred_cols))]
    
    # Находим индексы колонок для backdoor набора
    bd_idx = [pred_cols.index(name) for name in backdoor_names if name in pred_cols]
    
    if not bd_idx:
        bd_idx = list(range(min(10, X_tr.shape[1])))
    
    # ── Масштабирование ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_bd_tr = scaler.fit_transform(X_tr[:, bd_idx])
    X_bd_te = scaler.transform(X_te[:, bd_idx])
    
    # ── Прогнозные модели ────────────────────────────────────────────────────
    preds_list = []
    
    # Модель 1: RidgeCV
    try:
        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)
        ridge.fit(X_bd_tr, y_tr)
        preds_list.append(ridge.predict(X_bd_te))
        log.info(f"    Ridge alpha={ridge.alpha_:.3f}")
    except Exception as e:
        log.warning(f"    Ridge ошибка: {e}")
        # Fallback: простое среднее
        preds_list.append(np.ones(len(X_bd_te)) * np.mean(y_tr))
    
    # Модель 2: Random Forest (более стабильный, чем GBM)
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, 
                                    random_state=42, n_jobs=-1)
        rf.fit(X_bd_tr, y_tr)
        preds_list.append(rf.predict(X_bd_te))
        log.info(f"    RF n_features={X_bd_tr.shape[1]}")
    except Exception as e:
        log.warning(f"    RF ошибка: {e}")
        preds_list.append(preds_list[0])  # дублируем Ridge
    
    # Усреднение предсказаний
    y_pred = np.mean(preds_list, axis=0)
    
    # ── Вычисление метрик (только на валидных значениях) ────────────────────
    valid_mask = ~(np.isnan(y_te) | np.isnan(y_pred) | 
                   np.isinf(y_te) | np.isinf(y_pred))
    y_te_clean = y_te[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_te_clean) == 0:
        metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}
    else:
        rmse = np.sqrt(np.mean((y_te_clean - y_pred_clean) ** 2))
        mae = np.mean(np.abs(y_te_clean - y_pred_clean))
        
        # sMAPE
        denominator = (np.abs(y_te_clean) + np.abs(y_pred_clean)) / 2
        denominator = np.maximum(denominator, 1e-8)
        smape = 100 * np.mean(np.abs(y_te_clean - y_pred_clean) / denominator)
        
        # R²
        ss_res = np.sum((y_te_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_te_clean - np.mean(y_te_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {"RMSE": round(rmse, 6), "MAE": round(mae, 6),
                   "MAPE": round(smape, 4), "R2": round(r2, 4)}
    
    log.info(f"    SCM_Ensemble(h={horizon}): RMSE={metrics['RMSE']:.4f}  "
             f"MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE']:.2f}%  "
             f"R²={metrics['R2']:.4f}")
    
    # Формируем предсказания с правильными датами
    test_dates = feat.index[cut:][valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    return {
        "method": "SCM_Ensemble",
        "metrics": metrics,
        "predictions": dict(zip(
            test_dates.strftime("%Y-%m-%d").tolist(),
            [round(float(x), 4) for x in y_pred_valid]
        )),
        "backdoor_n_features": len(bd_idx),
    }

# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 2: Double Machine Learning (DML)
# ══════════════════════════════════════════════════════════════════════════════

def run_dml(df: pd.DataFrame, target: str,
            causes: List[str],
            controls: List[str],
            horizon: int = 1) -> Dict:
    """
    Double Machine Learning.
    """
    log.info(f"  [DML] {target} | horizon={horizon}")
    
    # ── Подготовка данных (каузальная) ──────────────────────────────────────
    avail_causes = [c for c in causes if c in df.columns]
    avail_controls = [c for c in controls if c in df.columns]
    
    if not avail_causes:
        avail_causes = [c for c in df.columns 
                        if c != target and not c.startswith(f"{target}_")][:3]
    
    all_features = list(dict.fromkeys(avail_causes + avail_controls))
    if not all_features:
        return {"method": "DML_SKIPPED", "metrics": {}}
    
    # Создаём признаки с лагами
    feat = pd.DataFrame(index=df.index)
    for col in all_features:
        feat[col] = df[col]
        feat[f"{col}_lag1"] = df[col].shift(1)
        feat[f"{col}_lag2"] = df[col].shift(2)
    
    # Целевая переменная со сдвигом
    y = df[target].shift(-horizon)
    
    # Объединяем и удаляем NaN
    combined = feat.join(y.rename("__target__")).dropna()
    
    if len(combined) < 60:
        return {"method": "DML_SKIPPED", "metrics": {}}
    
    X = combined.drop("__target__", axis=1)
    y_vec = combined["__target__"]
    
    # Разделение
    cut = int(len(X) * TRAIN_RATIO)
    X_tr = X.iloc[:cut]
    X_te = X.iloc[cut:]
    y_tr = y_vec.iloc[:cut]
    y_te = y_vec.iloc[cut:]
    
    # Определяем колонки причин
    all_cols = X.columns.tolist()
    cause_cols = []
    for base in avail_causes:
        for cand in [base, f"{base}_lag1", f"{base}_lag2"]:
            if cand in all_cols and cand not in cause_cols:
                cause_cols.append(cand)
                break
    
    if not cause_cols:
        cause_cols = all_cols[:min(3, len(all_cols))]
    
    control_cols = [c for c in all_cols if c not in cause_cols]
    
    # Преобразуем в numpy
    T_tr = X_tr[cause_cols].values if cause_cols else np.ones((len(X_tr), 1))
    W_tr = X_tr[control_cols].values if control_cols else np.ones((len(X_tr), 1))
    T_te = X_te[cause_cols].values if cause_cols else np.ones((len(X_te), 1))
    W_te = X_te[control_cols].values if control_cols else np.ones((len(X_te), 1))
    y_tr_np = y_tr.values
    y_te_np = y_te.values
    
    # Ручная реализация DML
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        
        # Step 1: partialling out Y
        m_y = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)
        m_y.fit(W_tr, y_tr_np)
        res_y_tr = y_tr_np - m_y.predict(W_tr)
        
        # Step 2: partialling out T
        res_T_tr = np.zeros_like(T_tr, dtype=float)
        m_t_list = []
        for j in range(T_tr.shape[1]):
            m_t = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=j)
            m_t.fit(W_tr, T_tr[:, j])
            res_T_tr[:, j] = T_tr[:, j] - m_t.predict(W_tr)
            m_t_list.append(m_t)
        
        # Step 3: OLS residual-on-residual
        theta_model = LinearRegression(fit_intercept=False)
        theta_model.fit(res_T_tr, res_y_tr)
        
        # Step 4: Prediction
        baseline = m_y.predict(W_te)
        res_T_te = T_te - np.column_stack([m_t.predict(W_te) for m_t in m_t_list])
        y_pred = baseline + res_T_te @ theta_model.coef_
        
        metrics = compute_metrics(y_te_np, y_pred, f"DML_manual_h{horizon}")
        
        return {
            "method": "DML_manual",
            "metrics": metrics,
            "predictions": dict(zip(
                y_te.index.strftime("%Y-%m-%d").tolist(),
                y_pred.tolist()
            )),
        }
        
    except Exception as e:
        log.error(f"    DML ошибка: {e}")
        return {"method": "DML_FAILED", "metrics": {}}


def _manual_dml(X_tr, X_te, y_tr, y_te,
                cause_cols: List[str],
                control_cols: List[str]) -> Dict:
    """
    Ручной DML:
      Step 1: ε_Y = Y - E[Y|Z]  (partialling out Y)
      Step 2: ε_T = T - E[T|Z]  (partialling out each treatment)
      Step 3: OLS ε_Y ~ ε_T     (causal coefficient θ)
      Step 4: Forecast = θ*T_test + RF(Z_test)
    """
    from sklearn.linear_model import LinearRegression, RidgeCV

    W_tr = X_tr[control_cols].values
    W_te = X_te[control_cols].values
    T_tr = X_tr[cause_cols].values
    T_te = X_te[cause_cols].values

    # Step 1: partialling out Y
    m_y = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0)
    m_y.fit(W_tr, y_tr.values)
    res_y_tr = y_tr.values - m_y.predict(W_tr)

    # Step 2: partialling out T (каждый treatment по отдельности)
    res_T_tr = np.zeros_like(T_tr, dtype=float)
    m_t_list = []
    for j in range(T_tr.shape[1]):
        m_t = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=j)
        m_t.fit(W_tr, T_tr[:, j])
        res_T_tr[:, j] = T_tr[:, j] - m_t.predict(W_tr)
        m_t_list.append(m_t)

    # Step 3: OLS residual-on-residual
    theta_model = LinearRegression(fit_intercept=False)
    theta_model.fit(res_T_tr, res_y_tr)
    theta = theta_model.coef_

    # Step 4: Prediction
    baseline = m_y.predict(W_te)
    res_T_te = T_te - np.column_stack(
        [m_t.predict(W_te) for m_t in m_t_list]
    )
    y_pred = baseline + res_T_te @ theta

    metrics = compute_metrics(y_te.values, y_pred, "DML_manual")
    return {
        "method":    "DML_manual",
        "metrics":   metrics,
        "predictions": dict(zip(
            y_te.index.strftime("%Y-%m-%d").tolist(),
            y_pred.tolist()
        )),
        "theta": dict(zip(cause_cols, theta.tolist())),
    }


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 3: VAR с ограничениями Granger
# ══════════════════════════════════════════════════════════════════════════════

def run_var_granger(df: pd.DataFrame, target: str,
                    causes: List[str],
                    controls: List[str],
                    max_lags: int = 6,
                    horizon: int = 1) -> Dict:
    """
    VAR + Granger causality:
      1. Тест Гренджера для каждого кандидата X → Y
      2. Оставляем только статистически значимые (p < 0.05)
      3. Подбираем лаг VAR по AIC/BIC
      4. Прогноз на горизонт horizon
    """
    log.info(f"  [VAR+Granger] {target}")

    import statsmodels.api as sm
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.stattools import grangercausalitytests

    candidates = list(dict.fromkeys(causes + controls))
    available  = [c for c in candidates if c in df.columns]

    sub = df[[target] + available].dropna()
    n   = len(sub)
    if n < 50:
        return {"method": "VAR_SKIPPED", "metrics": {}}

    cut = int(n * TRAIN_RATIO)
    train = sub.iloc[:cut]
    test  = sub.iloc[cut:]

    # ── Granger тест ──────────────────────────────────────────────────────────
    significant_vars = []
    granger_pvalues  = {}
    for col in available:
        try:
            pair = train[[target, col]].dropna()
            if len(pair) < 30:
                continue
            res = grangercausalitytests(pair, maxlag=min(max_lags, 4),
                                        verbose=False)
            # Минимальный p-value по всем лагам (F-test)
            p_vals = [res[lag][0]["ssr_ftest"][1]
                      for lag in range(1, min(max_lags, 4) + 1)]
            min_p = min(p_vals)
            granger_pvalues[col] = round(min_p, 4)
            if min_p < 0.05:
                significant_vars.append(col)
        except Exception:
            pass

    log.info(f"    Granger значимых: {len(significant_vars)}/{len(available)}")
    log.info(f"    {dict(list(granger_pvalues.items())[:5])}")

    # Если ничего не прошло — берём топ-3 по p-value
    if not significant_vars:
        significant_vars = sorted(granger_pvalues, key=granger_pvalues.get)[:3]
        log.info(f"    Используем топ-3 по p-value: {significant_vars}")

    # ── Подбор лага VAR по AIC ────────────────────────────────────────────────
    var_cols = [target] + significant_vars[:5]  # VAR до 6 переменных
    var_data = train[var_cols].dropna()

    try:
        model = VAR(var_data)
        lag_order = model.select_order(maxlags=max_lags)
        best_lag  = lag_order.aic
        if best_lag == 0:
            best_lag = 1
    except Exception:
        best_lag = 2

    # ── Обучение и прогноз ───────────────────────────────────────────────────
    try:
        fitted = VAR(var_data).fit(maxlags=best_lag, ic=None)

        # Прогнозируем шаг за шагом (rolling forecast)
        history  = var_data.values.copy()
        preds    = []
        test_var = test[var_cols].dropna()

        for i in range(len(test_var)):
            fc = fitted.forecast(history[-best_lag:], steps=horizon)
            preds.append(fc[horizon - 1, 0])  # индекс 0 = target
            # Обновляем историю реальными данными
            history = np.vstack([history, test_var.iloc[i].values])

        preds = np.array(preds)
        y_true = test_var[target].values[:len(preds)]
        metrics = compute_metrics(y_true, preds, f"VAR(lag={best_lag})")

        return {
            "method":  f"VAR_Granger(lag={best_lag})",
            "metrics": metrics,
            "predictions": dict(zip(
                test_var.index[:len(preds)].strftime("%Y-%m-%d").tolist(),
                preds.tolist()
            )),
            "granger_pvalues":  granger_pvalues,
            "significant_vars": significant_vars,
            "best_lag": best_lag,
        }
    except Exception as e:
        log.error(f"    VAR ошибка: {e}")
        return {"method": "VAR_FAILED", "metrics": {}}


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 4: IV-2SLS (инструментальные переменные)
# ══════════════════════════════════════════════════════════════════════════════

def run_iv_2sls(df: pd.DataFrame, target: str,
                causes: List[str],
                controls: List[str],
                instruments: List[str],
                horizon: int = 1) -> Dict:
    """
    Two-Stage Least Squares:
      Stage 1: X̂ = Z + controls  (инструменты предсказывают причину)
      Stage 2: Y = X̂ + controls  (используем X̂ вместо X)

    Устраняет endogeneity (simultaneity bias).
    """
    log.info(f"  [IV-2SLS] {target}")

    try:
        from linearmodels.iv import IV2SLS as lmIV
        use_linearmodels = True
    except ImportError:
        use_linearmodels = False

    # Формируем данные
    X_mat, y_vec = prepare_ml_data_causal(df, target, controls, 
                                           causes + instruments,
                                           horizon=horizon)
    if len(X_mat) < 60:
        return {"method": "IV_SKIPPED", "metrics": {}}
    
    X_tr, X_te, y_tr, y_te = train_test_split_ts(X_mat, y_vec)

    # Находим доступные инструменты и причины в матрице
    all_iv_cols = X_mat.columns.tolist()

    def _find_best(base_names, col_pool, max_n=5):
        """Ищет представителей base_names в col_pool: точное имя → _lag1 → startswith."""
        found = []
        for base in base_names:
            for cand in [base, f"{base}_lag1",
                         next((c for c in col_pool if c.startswith(base + "_")), None)]:
                if cand and cand in col_pool and cand not in found:
                    found.append(cand)
                    break
        return found[:max_n]

    inst_cols  = _find_best(instruments, all_iv_cols, max_n=3)
    endog_cols = _find_best(causes,      all_iv_cols, max_n=5)
    exog_cols  = [c for c in X_mat.columns
                  if c not in inst_cols + endog_cols][:8]

    if not inst_cols or not endog_cols:
        log.warning(f"    2SLS: нет инструментов или эндогенных переменных")
        return _manual_2sls(X_tr, X_te, y_tr, y_te,
                            endog_cols or X_mat.columns[:2].tolist(),
                            exog_cols)

    if use_linearmodels:
        try:
            import statsmodels.api as sm
            # linearmodels ожидает формат: endog, exog, dependent, instruments
            train_df = pd.concat([y_tr.rename("__y__"), X_tr], axis=1).dropna()
            endog = train_df[endog_cols]
            exog  = sm.add_constant(train_df[exog_cols])
            instruments_df = train_df[inst_cols]

            model = lmIV(train_df["__y__"], exog, endog, instruments_df)
            res   = model.fit(cov_type="robust")

            # Прогноз: вычисляем вручную через коэффициенты
            test_df = pd.concat([y_te.rename("__y__"), X_te], axis=1).dropna()
            X_test_exog = sm.add_constant(test_df[exog_cols])
            X_test_endog = test_df[endog_cols]

            coef_exog  = res.params[[c for c in res.params.index
                                     if c in X_test_exog.columns]]
            coef_endog = res.params[[c for c in res.params.index
                                     if c in X_test_endog.columns]]

            pred = (X_test_exog[coef_exog.index].values @ coef_exog.values +
                    X_test_endog[coef_endog.index].values @ coef_endog.values)

            metrics = compute_metrics(test_df["__y__"].values, pred, "IV-2SLS")
            return {
                "method":  "IV_2SLS",
                "metrics": metrics,
                "predictions": dict(zip(
                    test_df.index.strftime("%Y-%m-%d").tolist(),
                    pred.tolist()
                )),
                "first_stage_f": float(res.first_stage.diagnostics["f.stat"].mean())
                                  if hasattr(res, "first_stage") else None,
            }
        except Exception as e:
            log.warning(f"    linearmodels 2SLS ошибка: {e}")

    return _manual_2sls(X_tr, X_te, y_tr, y_te, endog_cols, exog_cols,
                        inst_cols)


def _manual_2sls(X_tr, X_te, y_tr, y_te,
                 endog_cols, exog_cols,
                 inst_cols: Optional[List[str]] = None) -> Dict:
    """Ручная реализация 2SLS через OLS."""
    from sklearn.linear_model import LinearRegression

    if not endog_cols:
        endog_cols = X_tr.columns[:1].tolist()

    W_tr  = X_tr[exog_cols].values if exog_cols else np.ones((len(X_tr), 1))
    W_te  = X_te[exog_cols].values if exog_cols else np.ones((len(X_te), 1))
    T_tr  = X_tr[endog_cols].values
    T_te  = X_te[endog_cols].values

    if inst_cols:
        Z_tr = X_tr[inst_cols].values
        # Stage 1
        stage1 = LinearRegression()
        Z_W_tr = np.hstack([Z_tr, W_tr])
        stage1.fit(Z_W_tr, T_tr)
        T_hat_tr = stage1.predict(Z_W_tr)

        Z_te = X_te[inst_cols].values
        Z_W_te = np.hstack([Z_te, W_te])
        T_hat_te = stage1.predict(Z_W_te)
    else:
        T_hat_tr = T_tr
        T_hat_te = T_te

    # Stage 2
    stage2 = LinearRegression()
    stage2.fit(np.hstack([T_hat_tr, W_tr]), y_tr.values)
    y_pred = stage2.predict(np.hstack([T_hat_te, W_te]))

    metrics = compute_metrics(y_te.values, y_pred, "2SLS_manual")
    return {
        "method":  "2SLS_manual",
        "metrics": metrics,
        "predictions": dict(zip(
            y_te.index.strftime("%Y-%m-%d").tolist(),
            y_pred.tolist()
        )),
    }


# ══════════════════════════════════════════════════════════════════════════════
# АДАПТИВНЫЙ ГОРИЗОНТ ПРОГНОЗА
# ══════════════════════════════════════════════════════════════════════════════

# Горизонты прогноза по частоте данных
HORIZONS = {
    "D":  [1, 5, 21],   # дневные:  1 день, 1 неделя, 1 месяц
    "B":  [1, 5, 21],   # рабочие дни — то же
    "ME": [1, 3, 6],    # месячные: 1 мес, квартал, полгода
    "MS": [1, 3, 6],
    "M":  [1, 3, 6],
    "W":  [1, 4, 13],   # недельные: 1 нед, 1 мес, квартал
}

def get_horizons(freq: str) -> List[int]:
    return HORIZONS.get(freq, [1, 5, 21])


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК ОДНОГО АКТИВА С ТРЕМЯ НАБОРАМИ КОНТРОЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_control_set(df_stat: pd.DataFrame,
                         target: str,
                         causes: List[str],
                         controls: List[str],
                         instruments: List[str],
                         horizons: List[int],
                         label: str) -> Dict:
    """
    Запускает все 4 каузальных метода для одного набора контролей Z
    и по всем горизонтам прогноза.
    Возвращает: {horizon: {method: result}}
    """
    result_by_horizon = {}
    for h in horizons:
        log.info(f"    horizon={h} | controls={label}")
        h_res = {}

        h_res["causal_impact"] = run_scm_dowhy(
            df_stat, target, causes, controls,
            dag=ALL_DAGS.get(target), horizon=h
        )

        h_res["dml"] = run_dml(df_stat, target, causes, controls,
                               horizon=h)

        h_res["var_granger"] = run_var_granger(df_stat, target, causes,
                                               controls, horizon=h)

        h_res["iv_2sls"] = run_iv_2sls(df_stat, target, causes, controls,
                                       instruments, horizon=h)

        result_by_horizon[f"h{h}"] = h_res

    return result_by_horizon


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК ВСЕХ КАУЗАЛЬНЫХ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def run_all_causal_models(var_selection: Dict) -> Dict:
    """
    Для каждого из 10 активов запускает 4 каузальных метода ×
    3 варианта контролей (heuristic / causal_expert / causal_pc) ×
    N горизонтов прогноза (адаптивно по частоте данных).

    Структура результата:
      {target: {
          "controls_heuristic":     {h1: {method: result}, ...},
          "controls_causal_expert": {...},
          "controls_causal_pc":     {...},
          "freq": "D" | "ME",
          "horizons": [1, 5, 21]
       }}
    """
    all_results = {}

    for target, info in var_selection.items():
        dag = ALL_DAGS.get(target)
        if dag is None:
            continue

        log.info(f"\n{'═'*60}")
        log.info(f"  КАУЗАЛЬНЫЕ МОДЕЛИ: {dag.name} ({target})  "
                 f"[freq={dag.freq}]")
        log.info(f"{'═'*60}")

        df_raw  = load_dataset(dag)
        if df_raw.empty or target not in df_raw.columns:
            log.warning(f"  Пропускаем {target}: нет данных")
            continue
        df_stat = prepare_stationary(df_raw, target)

        causes    = info.get("causes",      dag.causes)
        instrs    = info.get("instruments", dag.instruments)
        horizons  = get_horizons(dag.freq)

        # Три варианта наборов контролей
        control_sets = {
            "controls_heuristic":     info.get("heuristic",     []),
            "controls_causal_expert": info.get("causal_expert", []),
            "controls_causal_pc":     info.get("causal_pc",
                                               info.get("causal_expert", [])),
        }

        target_results = {
            "freq":    dag.freq,
            "horizons": horizons,
            "asset":   dag.name,
        }

        for ctrl_label, ctrl_vars in control_sets.items():
            log.info(f"\n  ▸ {ctrl_label}: {ctrl_vars}")
            target_results[ctrl_label] = _run_one_control_set(
                df_stat, target,
                causes, ctrl_vars, instrs,
                horizons, label=ctrl_label
            )

        all_results[target] = target_results

    # ── сериализация ──────────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    out_path = RESULTS_DIR / "step3_causal_models.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_clean(all_results), f, ensure_ascii=False, indent=2)
    log.info(f"\n✓ Результаты step3 сохранены: {out_path}")
    return all_results


if __name__ == "__main__":
    log.info("Загружаем отбор переменных (step2)...")
    sel_path = RESULTS_DIR / "step2_variable_selection.json"
    if sel_path.exists():
        with open(sel_path) as f:
            var_sel = json.load(f)
    else:
        log.info("Запускаем step2...")
        var_sel = run_variable_selection(use_pcmci=True)

    run_all_causal_models(var_sel)