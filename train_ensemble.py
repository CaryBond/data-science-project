import argparse
import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from joblib import dump, load

# LightGBM（老版本无 callbacks 也可运行）
try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except Exception:  # 老版本兜底
    LGBMRegressor = None
    early_stopping = None
    log_evaluation = None

# CatBoost
try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    CatBoostRegressor = None
    Pool = None

# SciPy（可选）
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)

@dataclass
class TrainConfig:
    data_path: str = "/mnt/data/features_with_ids_new_with_zeta.csv"
    target: Optional[str] = None
    n_splits: int = 5
    random_state: int = 42
    max_gpr_samples_per_fold: int = 3000
    lgbm_params: dict = None
    cat_params: dict = None
    gpr_kernel_length_scale: float = 10.0
    ridge_alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    artifacts_dir: str = "/mnt/data"
    use_gpu: bool = False
    verbose: int = 1

# —— 默认参数（保守，兼容旧版）——
DEFAULT_LGBM = dict(
    n_estimators=10000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_samples=20,
    objective="regression",
    n_jobs=-1,
    verbosity=-1,  # 抑制原生日志
)

DEFAULT_CAT = dict(
    iterations=10000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    od_type="Iter",
    od_wait=200,
    allow_writing_files=False,
    verbose=False,
)

ID_LIKE = {"id", "ID", "Id", "material_id", "uids", "index"}

# —— 工具函数 ——


def existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def guess_target(df: pd.DataFrame, feature_cols: List[str], user_target: Optional[str]) -> str:
    if user_target:
        if user_target not in df.columns:
            raise ValueError(f"--target 指定列不存在：{user_target}")
        return user_target
    candidates = [c for c in df.columns if c not in feature_cols and c not in ID_LIKE]
    candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if not candidates:
        raise ValueError("无法自动识别target列，请使用 --target 明确指定。")
    for p in ["target", "label", "y", "y_true", "property", "value"]:
        if p in candidates:
            return p
    return candidates[-1]


def detect_categorical_indices(df: pd.DataFrame, feature_cols: List[str]) -> List[int]:
    # 只在 CatBoost 用；老版 pandas 用 isinstance(dtype, pd.CategoricalDtype)
    cats = []
    for i, c in enumerate(feature_cols):
        if is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype):
            cats.append(i)
    return cats


def mse_np(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    # 兼容旧版 sklearn：无 squared 参数则自己开根号
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mse_np(y_true, y_pred)))


def kmeans_subset(X: np.ndarray, y: np.ndarray, k: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n <= k:
        return X, y
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)  # 兼容旧版
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    idxs = []
    for j in range(k):
        mask = np.where(labels == j)[0]
        d = np.linalg.norm(X[mask] - centers[j], axis=1)
        idxs.append(mask[np.argmin(d)])
    idxs = np.array(idxs)
    return X[idxs], y[idxs]


def nonneg_mse_weights(oof_preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    非负简单权重（权重和=1）最小化 MSE。
    使用 numpy MSE，避免旧版 sklearn 的 squared 参数。
    """
    m = oof_preds.shape[1]
    if not SCIPY_AVAILABLE:
        return np.ones(m) / m

    x0 = np.ones(m) / m

    def objective(w):
        return mse_np(y, oof_preds @ w)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * m

    res = minimize(objective, x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 200})
    if res.success and np.isfinite(res.fun):
        return res.x
    return np.ones(m) / m


def build_meta_learner(alphas):
    """
    构建元学习器：优先 RidgeCV（不传 store_cv_values），
    若旧版不可用则回退到 GridSearchCV(Ridge)。
    """
    try:
        return RidgeCV(alphas=alphas)
    except TypeError:
        return GridSearchCV(
            estimator=Ridge(),
            param_grid={"alpha": list(alphas)},
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )


# —— 主流程 ——


def train_and_evaluate(cfg: TrainConfig):
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    df = pd.read_csv(cfg.data_path)
    base_features = [f"feature_{i}" for i in range(128)]
    extra_features = ["a", "b", "c", "void_fraction", "surface_area", "lcd", "pld",
                      "zeta_stable_s2", "zeta_stable_s3", "zeta_stable_s4", "ASA"]
    feature_cols = existing_columns(df, base_features + extra_features)
    if not feature_cols:
        raise ValueError("No specified feature columns were found. Please check the column names.")

    target_col = guess_target(df, feature_cols, cfg.target)

    X_all = df[feature_cols]
    y_all = df[target_col].astype(float).values

    cat_indices = detect_categorical_indices(df, feature_cols)

    # LightGBM
    if LGBMRegressor is None:
        raise ImportError("lightgbm not installed，use command：pip install lightgbm")

    lgbm_params = dict(DEFAULT_LGBM)
    if cfg.lgbm_params:
        lgbm_params.update(cfg.lgbm_params)
    if cfg.use_gpu:
        # 旧版用 device='gpu'；如失败用户可去掉该开关
        lgbm_params.update(dict(device="gpu"))

    # CatBoost
    if CatBoostRegressor is None:
        raise ImportError("catboost not installed，use command：pip install catboost")

    cat_params = dict(DEFAULT_CAT)
    if cfg.cat_params:
        cat_params.update(cfg.cat_params)
    if cfg.use_gpu:
        cat_params.update(dict(task_type="GPU"))

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    oof_cat = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))
    oof_gpr = np.zeros(len(df))

    fold_metrics: List[Dict[str, float]] = []

    models_cat, models_lgb, models_gpr = [], [], []

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
        X_tr, X_va = X_all.iloc[trn_idx], X_all.iloc[val_idx]
        y_tr, y_va = y_all[trn_idx], y_all[val_idx]

        # LightGBM：用 callbacks 控制日志/早停
        lgb = LGBMRegressor(**lgbm_params)
        cbks = []
        if early_stopping is not None:
            cbks.append(early_stopping(stopping_rounds=200))
        if log_evaluation is not None:
            cbks.append(log_evaluation(period=0))
        lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=cbks
        )
        pred_lgb = lgb.predict(X_va)
        oof_lgb[val_idx] = pred_lgb
        models_lgb.append(lgb)

        # CatBoost
        if cat_indices:
            train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            valid_pool = Pool(X_va, y_va, cat_features=cat_indices)
            cat = CatBoostRegressor(**cat_params)
            cat.fit(train_pool, eval_set=valid_pool, verbose=False)
            pred_cat = cat.predict(valid_pool)
        else:
            cat = CatBoostRegressor(**cat_params)
            cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            pred_cat = cat.predict(X_va)
        oof_cat[val_idx] = pred_cat
        models_cat.append(cat)

        # GPR（子样本 + 标准化）
        scaler = StandardScaler()
        kernel = RBF(length_scale=cfg.gpr_kernel_length_scale) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=cfg.random_state)
        X_sub, y_sub = kmeans_subset(
            X_tr.to_numpy(dtype=float), y_tr.astype(float),
            k=min(cfg.max_gpr_samples_per_fold, len(X_tr)), random_state=cfg.random_state
        )
        gpr_pipe = Pipeline([("scaler", scaler), ("gpr", gpr)])
        gpr_pipe.fit(X_sub, y_sub)
        pred_gpr = gpr_pipe.predict(X_va.to_numpy(dtype=float))
        oof_gpr[val_idx] = pred_gpr
        models_gpr.append(gpr_pipe)

        m = {
            "fold": fold,
            "RMSE_lgb": rmse(y_va, pred_lgb),
            "RMSE_cat": rmse(y_va, pred_cat),
            "RMSE_gpr": rmse(y_va, pred_gpr),
            "RMSE_best_base": min(rmse(y_va, pred_lgb), rmse(y_va, pred_cat), rmse(y_va, pred_gpr)),
        }
        fold_metrics.append(m)
        if cfg.verbose:
            print(f"[Fold {fold}] RMSE LGB={m['RMSE_lgb']:.5f} | CAT={m['RMSE_cat']:.5f} | GPR={m['RMSE_gpr']:.5f}")

    # OOF 总结
    metrics = {
        "RMSE_lgb": rmse(y_all, oof_lgb),
        "MAE_lgb": mean_absolute_error(y_all, oof_lgb),
        "R2_lgb": r2_score(y_all, oof_lgb),

        "RMSE_cat": rmse(y_all, oof_cat),
        "MAE_cat": mean_absolute_error(y_all, oof_cat),
        "R2_cat": r2_score(y_all, oof_cat),

        "RMSE_gpr": rmse(y_all, oof_gpr),
        "MAE_gpr": mean_absolute_error(y_all, oof_gpr),
        "R2_gpr": r2_score(y_all, oof_gpr),
    }

    # 元学习：RidgeCV / GridSearchCV(Ridge)
    Z = np.vstack([oof_lgb, oof_cat, oof_gpr]).T
    ridge = build_meta_learner(cfg.ridge_alphas)
    ridge.fit(Z, y_all)
    oof_stack = ridge.predict(Z)

    # 非负权重
    w_nonneg = nonneg_mse_weights(Z, y_all)
    oof_blend = Z @ w_nonneg

    metrics.update({
        "RMSE_stack_ridge": rmse(y_all, oof_stack),
        "MAE_stack_ridge": mean_absolute_error(y_all, oof_stack),
        "R2_stack_ridge": r2_score(y_all, oof_stack),

        "RMSE_blend_nonneg": rmse(y_all, oof_blend),
        "MAE_blend_nonneg": mean_absolute_error(y_all, oof_blend),
        "R2_blend_nonneg": r2_score(y_all, oof_blend),
    })

    # 保存 OOF 与指标
    oof_df = pd.DataFrame({
        "y": y_all,
        "oof_lgb": oof_lgb,
        "oof_cat": oof_cat,
        "oof_gpr": oof_gpr,
        "oof_stack_ridge": oof_stack,
        "oof_blend_nonneg": oof_blend,
    })
    oof_path = os.path.join(cfg.artifacts_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)

    metrics_path = os.path.join(cfg.artifacts_dir, "cv_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"folds": fold_metrics, "summary": metrics}, f, ensure_ascii=False, indent=2)

    if cfg.verbose:
        print("\n[CV Summary]")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
        print(f"Saved OOF -> {oof_path}")
        print(f"Saved metrics -> {metrics_path}")

    # 全量再训（不做早停以避免旧版接口差异）
    lgb_full = LGBMRegressor(**lgbm_params)
    lgb_full.fit(X_all, y_all)

    if cat_indices:
        pool_full = Pool(X_all, y_all, cat_features=cat_indices)
        cat_full = CatBoostRegressor(**cat_params)
        cat_full.fit(pool_full, verbose=False)
    else:
        cat_full = CatBoostRegressor(**cat_params)
        cat_full.fit(X_all, y_all, verbose=False)

    scaler_full = StandardScaler()
    kernel_full = RBF(length_scale=cfg.gpr_kernel_length_scale) + WhiteKernel(noise_level=1.0)
    gpr_full = GaussianProcessRegressor(kernel=kernel_full, alpha=1e-6, normalize_y=True, random_state=cfg.random_state)
    X_np, y_np = X_all.to_numpy(dtype=float), y_all.astype(float)
    X_sub_full, y_sub_full = kmeans_subset(X_np, y_np, k=min(cfg.max_gpr_samples_per_fold, len(X_np)), random_state=cfg.random_state)
    gpr_pipe_full = Pipeline([("scaler", scaler_full), ("gpr", gpr_full)])
    gpr_pipe_full.fit(X_sub_full, y_sub_full)

    # 特征重要性（可能旧版缺字段，失败跳过）
    try:
        fi_lgb = pd.DataFrame({"feature": feature_cols, "importance": lgb_full.feature_importances_})
        fi_lgb.sort_values("importance", ascending=False).to_csv(os.path.join(cfg.artifacts_dir, "feature_importance_lgbm.csv"), index=False)
    except Exception:
        pass
    try:
        fi_cat = pd.DataFrame({"feature": feature_cols, "importance": cat_full.get_feature_importance()})
        fi_cat.sort_values("importance", ascending=False).to_csv(os.path.join(cfg.artifacts_dir, "feature_importance_catboost.csv"), index=False)
    except Exception:
        pass

    bundle = dict(
        config=asdict(cfg),
        feature_cols=feature_cols,
        target_col=target_col,
        models=dict(lgb=lgb_full, cat=cat_full, gpr=gpr_pipe_full),
        meta=dict(ridge=ridge, nonneg_weights=w_nonneg.tolist(), strategy="ridge_and_nonneg"),
        metrics=metrics,
    )
    model_path = os.path.join(cfg.artifacts_dir, "ensemble_model.pkl")
    dump(bundle, model_path)
    if cfg.verbose:
        print(f"Saved model bundle -> {model_path}")

    return {
        "oof_path": oof_path,
        "metrics_path": metrics_path,
        "model_path": model_path,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }


def predict_csv(model_path: str, input_csv: str, output_csv: str, mode: str = "ridge"):
    bundle = load(model_path)
    feature_cols = bundle["feature_cols"]
    models = bundle["models"]
    df = pd.read_csv(input_csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing feature columns：{missing[:10]}{'...' if len(missing)>10 else ''}")
    X = df[feature_cols]
    pred_lgb = models["lgb"].predict(X)
    pred_cat = models["cat"].predict(X)
    pred_gpr = models["gpr"].predict(X.to_numpy(dtype=float))
    Z = np.vstack([pred_lgb, pred_cat, pred_gpr]).T
    if mode == "ridge":
        preds = bundle["meta"]["ridge"].predict(Z)
    elif mode == "nonneg":
        preds = Z @ np.array(bundle["meta"]["nonneg_weights"])
    else:
        preds = Z.mean(axis=1)
    out = df.copy()
    out["prediction"] = preds
    out.to_csv(output_csv, index=False)
    return output_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/mnt/data/features_with_ids_new_with_zeta.csv")
    p.add_argument("--target", type=str, default=None, help="目标列名；若不提供则自动推测")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--gpr_max_samples", type=int, default=3000)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="/mnt/data", help="产物输出目录")
    p.add_argument("--predict_in", type=str, default=None, help="推理输入CSV路径（可选）")
    p.add_argument("--predict_out", type=str, default="/mnt/data/predictions.csv", help="推理输出CSV路径（可选）")
    p.add_argument("--predict_mode", type=str, default="ridge", choices=["ridge", "nonneg", "avg"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        data_path=args.data,
        target=args.target,
        n_splits=args.splits,
        random_state=args.random_state,
        max_gpr_samples_per_fold=args.gpr_max_samples,
        use_gpu=args.use_gpu,
        verbose=args.verbose,
        artifacts_dir=args.out_dir,
    )
    result = train_and_evaluate(cfg)
    if args.predict_in:
        out_path = predict_csv(
            model_path=result["model_path"],
            input_csv=args.predict_in,
            output_csv=args.predict_out,
            mode=args.predict_mode
        )
        print(f"Predictions saved -> {out_path}")


if __name__ == "__main__":
    main()
