"""Small-sample RankNet variants and genuine LightGBM ranking models.

All feature/target statistics use training rows only. One row is one landing
site. `base` is a caller-supplied, leak-free site prediction. There is no test-y
argument. LightGBM native scores are not chemical abundance units.

The neural architecture is one shared tanh scoring network evaluated on every
site; its pairwise logistic likelihood follows the RankNet idea. It is not a
claim of reproducing the 2005 paper's exact software or architecture. The old
RankMLP was already a neural pair-logistic model; `legacy_rankmlp` preserves that
configuration as an explicit control.
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys
import time
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

SEEDS = (42, 43, 44, 45, 46)
ROOT = Path(__file__).resolve().parent


def _lightgbm():
    vendor = str(ROOT / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return importlib.import_module("lightgbm")


def core_grid() -> list[dict[str, Any]]:
    """Frozen modest search grid; controls are returned separately."""
    out = []
    for width in (4, 8, 16):
        for lam in (0.0, 0.1, 1.0, 3.0):
            out.append(dict(id=f"RN_W{width:02d}_JOINT_L{lam:g}",
                            family="ranknet", width=width, lambda_rank=lam,
                            rank_only=False, weight_decay=0.01,
                            correction_penalty=0.1, maxiter=200,
                            abundance_loss="smooth_L1", smooth_l1_epsilon=0.1))
        out.append(dict(id=f"RN_W{width:02d}_RANK", family="ranknet",
                        width=width, lambda_rank=1.0, rank_only=True,
                        weight_decay=0.01, correction_penalty=0.1,
                        maxiter=200))
    recipes = [
        dict(preset="S", max_depth=1, num_leaves=2, n_estimators=10,
             reg_lambda=10.0, min_child_samples=1),
        dict(preset="M", max_depth=2, num_leaves=3, n_estimators=30,
             reg_lambda=10.0, min_child_samples=1),
        dict(preset="L", max_depth=2, num_leaves=4, n_estimators=60,
             reg_lambda=1.0, min_child_samples=2),
    ]
    for objective in ("lambdarank", "rank_xendcg"):
        for rec in recipes:
            out.append(dict(id=f"LGB_{objective}_{rec['preset']}",
                            family="lightgbm", objective=objective,
                            learning_rate=0.05, calibration_rho=0.1, **rec))
    return out


def control_grid() -> list[dict[str, Any]]:
    out = [dict(id=f"RIDGE_A{a:g}", family="ridge", alpha=a)
           for a in (1.0, 10.0, 100.0)]
    for strength in (1.0, 10.0):
        out.append(dict(id=f"LINEAR_RANK_S{strength:g}",
                        family="linear_ranknet", rank_only=True,
                        lambda_rank=1.0, weight_decay=0.1 * strength,
                        correction_penalty=0.1 * strength, maxiter=300))
        out.append(dict(id=f"LEGACY_RANKMLP_S{strength:g}",
                        family="legacy_rankmlp", strength=strength))
    return out


def _inputs(X, y, base, Xt, bt):
    X, Xt = np.asarray(X, float), np.asarray(Xt, float)
    y, base, bt = [np.asarray(v, float).reshape(-1) for v in (y, base, bt)]
    if X.ndim != 2 or Xt.ndim != 2 or X.shape[1] != Xt.shape[1]:
        raise ValueError("Training/test X must be matrices with equal column counts")
    if len(X) != len(y) or len(y) != len(base) or len(Xt) != len(bt):
        raise ValueError("Input lengths differ")
    if len(y) < 2 or X.shape[1] < 1:
        raise ValueError("At least two training sites and one feature are needed")
    if not all(np.isfinite(a).all() for a in (X, Xt, y, base, bt)):
        raise ValueError("Nonfinite training/test input: no implicit imputation")
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-10] = 1.0
    scale = max(float(y.std()), 0.1)
    return X, y, base, Xt, bt, mu, sd, scale


def _pairs(y):
    i, j = np.triu_indices(len(y), 1)
    s = np.sign(y[i] - y[j])
    good = s != 0.0
    return i[good], j[good], s[good]


def _forward(z, w, width):
    if width == 0:
        return z @ w, None
    d = z.shape[1]
    A = w[:d * width].reshape(d, width)
    h_bias = w[d * width:d * width + width]
    v = w[d * width + width:d * width + 2 * width]
    c = w[-1]
    h = np.tanh(z @ A + h_bias)
    return h @ v + c, (h, v)


def _loss_grad(w, z, residual, base_scaled, pair, width, config):
    p, cache = _forward(z, w, width)
    n = len(z)
    loss, dp = 0.0, np.zeros(n)
    i, j, s = pair
    if not config.get("rank_only", False):
        err = p - residual
        eps = float(config.get("smooth_l1_epsilon", 0.1))
        root = np.sqrt(err * err + eps * eps)
        loss += float(np.mean(root - eps))
        dp += err / root / n
    lam = float(config.get("lambda_rank", 1.0))
    if len(i) and lam:
        margin = s * (base_scaled[i] + p[i] - base_scaled[j] - p[j])
        loss += lam * float(np.mean(np.logaddexp(0.0, -margin)))
        g = -lam * s * expit(-margin) / len(i)
        np.add.at(dp, i, g)
        np.add.at(dp, j, -g)
    mag = float(config.get("correction_penalty", 0.1))
    loss += 0.5 * mag * float(np.mean(p * p))
    dp += mag * p / n
    if width == 0:
        grad = z.T @ dp
    else:
        h, v = cache
        dh = (dp[:, None] * v) * (1.0 - h * h)
        grad = np.r_[(z.T @ dh).ravel(), dh.sum(0), h.T @ dp, dp.sum()]
    wd = float(config.get("weight_decay", 0.01))
    loss += 0.5 * wd * float(w @ w)
    grad += wd * w
    return loss, grad


def _initial_weights(d, width, seed, legacy=False):
    rng = np.random.default_rng(seed)
    if width == 0:
        return np.zeros(d)
    if legacy:
        return rng.normal(0.0, 0.1, d * width + 2 * width + 1)
    A = rng.normal(0.0, 1.0 / np.sqrt(d), (d, width))
    b = np.zeros(width)
    v = rng.normal(0.0, 1.0 / np.sqrt(width), width)
    return np.r_[A.ravel(), b, v, 0.0]


def _neural_fit(z, y, base, zt, bt, scale, config, seeds):
    cfg = copy.deepcopy(config)
    family = cfg["family"]
    legacy = family == "legacy_rankmlp"
    width = 0 if family == "linear_ranknet" else int(cfg.get("width", 4))
    if legacy:
        strength = float(cfg.get("strength", 1.0))
        width = 4
        cfg.update(rank_only=True, lambda_rank=1.0,
                   weight_decay=0.1 * strength, correction_penalty=0.1 * strength,
                   maxiter=100)
    effective_seeds = [42] if width == 0 else list(seeds)
    residual = (y - base) / scale
    bscaled = base / scale
    pair = _pairs(y)
    pp, tp, weights, logs = [], [], [], []
    for seed in effective_seeds:
        t0 = time.perf_counter()
        w0 = _initial_weights(z.shape[1], width, seed, legacy)
        with warnings.catch_warnings(record=True) as ww:
            warnings.simplefilter("always")
            fit = minimize(_loss_grad, w0,
                           args=(z, residual, bscaled, pair, width, cfg),
                           jac=True, method="L-BFGS-B",
                           options=dict(maxiter=int(cfg.get("maxiter", 200)),
                                        ftol=1e-10, gtol=1e-6, maxls=30))
        rawp, _ = _forward(zt, fit.x, width)
        rawtrain, _ = _forward(z, fit.x, width)
        pp.append(bt + rawp * scale)
        tp.append(base + rawtrain * scale)
        weights.append(fit.x.tolist())
        logs.append(dict(seed=int(seed), success=bool(fit.success),
                         message=str(fit.message), iterations=int(fit.nit),
                         loss=float(fit.fun), gradient_norm=float(np.linalg.norm(fit.jac)),
                         warning_messages=[str(w.message) for w in ww],
                         seconds=time.perf_counter() - t0))
    pp, tp = np.asarray(pp).T, np.asarray(tp).T
    pred, trainpred = pp.mean(1), tp.mean(1)
    state = dict(kind="neural_pair", width=width, weights=weights,
                 config=cfg, seeds=effective_seeds, y_scale=scale,
                 pair_count=len(pair[0]), rank_only=cfg.get("rank_only", False))
    return dict(prediction=pred, native_score=pred.copy(),
                train_native_score=trainpred.copy(), train_prediction=trainpred,
                seed_predictions=pp, seed_native_scores=pp.copy(),
                delta=pred - bt, state=state, logs=logs,
                chemical_units=True,
                calibration_kind="none: neural additive correction in wt%p")


def _ridge_fit(z, y, base, zt, bt, scale, config):
    alpha = float(config.get("alpha", 10.0))
    r = (y - base) / scale
    rmean = float(r.mean())
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (r-rmean))
    pred = bt + (rmean + zt @ coef) * scale
    trainpred = base + (rmean + z @ coef) * scale
    return dict(prediction=pred, native_score=pred.copy(),
                train_prediction=trainpred, train_native_score=trainpred.copy(),
                delta=pred-bt, seed_predictions=pred[:, None],
                seed_native_scores=pred[:, None], chemical_units=True,
                calibration_kind="none: ordinary residual regression in wt%p",
                state=dict(kind="ridge", coef=coef.tolist(), residual_mean=rmean,
                           y_scale=scale, config=config, seeds=[]),
                logs=[dict(success=True, message="closed_form_ridge", seed=None)])


def _positive_affine_calibration(train_scores, y, test_scores, rho):
    """Training-only in-sample calibration; assess the entire model held-out.

    This is an explicit component of the fitted algorithm. Raw scores from
    independently fitted inner models must not be pooled as if on one scale.
    """
    m = float(np.mean(train_scores))
    sd = float(np.std(train_scores))
    constant_score = sd < 1e-10
    raw_sd = sd
    if constant_score:
        sd = 1.0
    tr, te = (train_scores - m) / sd, (test_scores - m) / sd
    ym = float(y.mean())
    alpha = float(rho) * len(y)
    slope = max(0.0, float(tr @ (y-ym)) / (float(tr @ tr) + alpha))
    return ym + slope * te, ym + slope * tr, dict(
        kind="training_in_sample_positive_affine_ridge",
        score_mean=m, score_sd=sd, intercept=ym, slope=slope,
        mean_loss_rho=float(rho), sum_loss_alpha=alpha,
        constant_training_score=bool(constant_score),
        raw_training_score_sd=raw_sd), tr, te


def _tree_fit(z, y, base, zt, bt, config, seeds):
    lgb = _lightgbm()
    objective = config.get("objective", "lambdarank")
    if objective not in ("lambdarank", "rank_xendcg"):
        raise ValueError("Only true ranking objectives allowed in lightgbm family")
    labels = rankdata(y, method="dense").astype(int) - 1
    effective_seeds = list(config.get("seeds", [42] if objective == "lambdarank" else seeds))
    test_scores, train_scores, models, logs = [], [], [], []
    for seed in effective_seeds:
        t0 = time.perf_counter()
        params = dict(objective=objective, n_jobs=1, random_state=int(seed),
                      objective_seed=int(seed), data_random_seed=int(seed),
                      deterministic=True, force_col_wise=True,
                      verbosity=-1, feature_pre_filter=False,
                      learning_rate=float(config.get("learning_rate", 0.05)),
                      n_estimators=int(config.get("n_estimators", 30)),
                      max_depth=int(config.get("max_depth", 2)),
                      num_leaves=int(config.get("num_leaves", 3)),
                      min_child_samples=int(config.get("min_child_samples", 1)),
                      min_child_weight=1e-6, min_data_in_bin=1,
                      reg_lambda=float(config.get("reg_lambda", 10.0)),
                      reg_alpha=0.0, colsample_bytree=1.0,
                      subsample=1.0, subsample_freq=0)
        if objective == "lambdarank":
            params.update(label_gain=list(range(int(labels.max()) + 1)),
                          lambdarank_truncation_level=max(1, len(y)),
                          lambdarank_norm=True)
        with warnings.catch_warnings(record=True) as ww:
            warnings.simplefilter("always")
            est = lgb.LGBMRanker(**params)
            est.fit(z, labels, group=[len(y)])
            p = est.booster_.predict(zt, num_threads=1)
            ptrain = est.booster_.predict(z, num_threads=1)
        model = est.booster_.model_to_string()
        dump = est.booster_.dump_model()
        leaves = [int(t["num_leaves"]) for t in dump["tree_info"]]
        test_scores.append(p)
        train_scores.append(ptrain)
        models.append(model)
        logs.append(dict(seed=int(seed), success=True, message="fit_completed",
                         warning_messages=[str(w.message) for w in ww],
                         actual_trees=est.booster_.num_trees(), leaves=leaves,
                         informative_splits=int(sum(max(0, n-1) for n in leaves)),
                         train_score_std=float(np.std(ptrain)),
                         seconds=time.perf_counter()-t0))
    ss, st = np.asarray(test_scores).T, np.asarray(train_scores).T
    score, train_score = ss.mean(1), st.mean(1)
    pred, trainpred, cal, trstd, teststd = _positive_affine_calibration(
        train_score, y, score, config.get("calibration_rho", 0.1))
    seed_cal = cal["intercept"] + cal["slope"] * (ss-cal["score_mean"]) / cal["score_sd"]
    return dict(prediction=pred, native_score=score,
                train_prediction=trainpred, train_native_score=train_score,
                standardized_native_score=teststd,
                train_standardized_native_score=trstd,
                delta=pred-bt, seed_predictions=seed_cal,
                seed_native_scores=ss, chemical_units=False,
                calibration_kind=cal["kind"],
                state=dict(kind="lightgbm", model_strings=models,
                           config=config, seeds=effective_seeds,
                           calibration=cal, train_integer_labels=labels.tolist(),
                           query_group=[len(y)], lightgbm_version=lgb.__version__),
                logs=logs)


def fit_predict(X_train, y_site, base_train, X_test, base_test, config, seeds=SEEDS):
    """Fit on training sites only; return native scores plus explicit unit labels.

Keys: prediction, native_score, train_prediction, train_native_score,
    delta, seed_predictions, seed_native_scores, state, logs, chemical_units,
    calibration_kind. For LightGBM, `prediction` is the algorithm's training-only
    IN-SAMPLE-calibrated output, not the native score; calibration never sees test y.
`chemical_units` refers to native_score (False for LightGBM).
"""
    X, y, base, Xt, bt, mu, sd, scale = _inputs(
        X_train, y_site, base_train, X_test, base_test)
    z, zt = (X-mu)/sd, (Xt-mu)/sd
    cfg = copy.deepcopy(config)
    family = cfg.get("family")
    t0 = time.perf_counter()
    if family in ("ranknet", "linear_ranknet", "legacy_rankmlp"):
        res = _neural_fit(z, y, base, zt, bt, scale, cfg, seeds)
    elif family == "ridge":
        res = _ridge_fit(z, y, base, zt, bt, scale, cfg)
    elif family == "lightgbm":
        res = _tree_fit(z, y, base, zt, bt, cfg, seeds)
    else:
        raise ValueError(f"Unknown family: {family}")
    for key in ("prediction", "native_score", "train_prediction", "seed_predictions"):
        if not np.isfinite(res[key]).all():
            raise FloatingPointError(f"Nonfinite learner output: {key}")
    res["state"].update(x_mean=mu.tolist(), x_sd=sd.tolist(),
                        feature_count=X.shape[1], train_site_count=len(y),
                        schema_version=1)
    res["prediction_has_chemical_units"] = True
    res["native_score_has_chemical_units"] = bool(res["chemical_units"])
    res["seconds"] = time.perf_counter()-t0
    return res


def predict_from_state(state, X_test, base_test):
    """Replay portable saved state, including native LightGBM ranking scores."""
    X = np.asarray(X_test, float)
    base = np.asarray(base_test, float).reshape(-1)
    if X.ndim != 2 or len(X) != len(base) or X.shape[1] != state["feature_count"]:
        raise ValueError("Replay dimensions differ from trained model")
    z = (X-np.asarray(state["x_mean"])) / np.asarray(state["x_sd"])
    kind = state["kind"]
    if kind == "neural_pair":
        pp = [base + _forward(z, np.asarray(w), state["width"])[0] * state["y_scale"]
              for w in state["weights"]]
        seedpred = np.asarray(pp).T
        pred = seedpred.mean(1)
        return dict(prediction=pred, native_score=pred.copy(),
                    seed_predictions=seedpred, seed_native_scores=seedpred.copy(),
                    delta=pred-base)
    if kind == "ridge":
        pred = base + (state["residual_mean"] + z @ np.asarray(state["coef"])) * state["y_scale"]
        return dict(prediction=pred, native_score=pred.copy(),
                    seed_predictions=pred[:, None], seed_native_scores=pred[:, None],
                    delta=pred-base)
    if kind == "lightgbm":
        lgb = _lightgbm()
        ss = np.asarray([lgb.Booster(model_str=s).predict(z, num_threads=1)
                         for s in state["model_strings"]]).T
        score = ss.mean(1)
        cal = state["calibration"]
        standard = (score-cal["score_mean"])/cal["score_sd"]
        pred = cal["intercept"]+cal["slope"]*standard
        seedpred = cal["intercept"]+cal["slope"]*(ss-cal["score_mean"])/cal["score_sd"]
        return dict(prediction=pred, native_score=score, standardized_native_score=standard,
                    seed_predictions=seedpred, seed_native_scores=ss, delta=pred-base)
    raise ValueError(f"Unknown saved model kind: {kind}")


def environment():
    import scipy
    import sklearn
    lgb = _lightgbm()
    return dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, lightgbm=lgb.__version__,
                torch="not used", lightgbm_n_jobs=1)
