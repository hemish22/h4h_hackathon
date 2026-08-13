"""Part 3.1 — LightGBM tabular baseline (the safety submission + SHAP story).

Classification + 3 regressors on the 18 tabular features. Uses CSV splits.
Run: python -m src.baseline
"""
from __future__ import annotations
import os
import numpy as np
import joblib

from . import config as C
from . import data_utils as D
from . import splits as S
from . import evaluate as E


def _xy():
    csv = D.load_csv()
    idx = {sp: [int(i) for i in S.load_split(f"csv_{sp}")] for sp in ["train", "val", "test"]}
    X = {sp: csv.loc[idx[sp], C.FEATURE_COLS].values for sp in idx}
    y_cls = {sp: csv.loc[idx[sp], C.TARGET_CAT].map(C.CLASS_TO_ORD).values for sp in idx}
    y_reg = {sp: csv.loc[idx[sp], C.TARGET_REG].values for sp in idx}
    return X, y_cls, y_reg


def run():
    import lightgbm as lgb
    X, y_cls, y_reg = _xy()

    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             num_leaves=31, class_weight="balanced",
                             random_state=C.SEED, verbose=-1)
    clf.fit(X["train"], y_cls["train"])
    proba = clf.predict_proba(X["test"])
    pred = proba.argmax(1)
    cls_m = E.classification_metrics(y_cls["test"], pred, proba)

    reg_pred = np.zeros_like(y_reg["test"], dtype=float)
    regs = []
    for i, name in enumerate(C.TARGET_REG):
        r = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=31, random_state=C.SEED, verbose=-1)
        r.fit(X["train"], y_reg["train"][:, i])
        reg_pred[:, i] = r.predict(X["test"])
        regs.append(r)
    reg_m = E.regression_metrics(y_reg["test"], reg_pred)

    joblib.dump({"clf": clf, "regs": regs}, os.path.join(C.ARTIFACTS_DIR, "lgbm.joblib"))
    E.save_results("lgbm_tabular", cls_m, reg_m)
    print("Tabular LightGBM baseline:")
    print(f"  acc={cls_m['accuracy']:.3f} macroF1={cls_m['macro_f1']:.3f} "
          f"qwk={cls_m['qwk_EXTRA']:.3f}")
    for n in ["Depression", "Anxiety", "Stress"]:
        print(f"  {n} MAE={reg_m[n]['MAE']:.2f} R2={reg_m[n]['R2']:.3f}")
    print("\nAblation row:")
    print(E.ABLATION_HEADER)
    print(E.markdown_row("Tabular only (LightGBM)", cls_m, reg_m, "n/a"))
    return cls_m, reg_m


if __name__ == "__main__":
    run()
