from __future__ import annotations

import argparse
import itertools
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .features import ID_COL, TARGET_COL, build_domain_dataset
from .io import ensure_dir, save_dataframe, save_json, save_pickle
from .metrics import roc_auc_score_manual
from .split import make_or_load_holdout_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Home Credit domain models.")
    parser.add_argument("--config", default="configs/debug.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--domain",
        required=True,
        choices=[
            "application",
            "bureau",
            "previous_application",
            "installments",
            "pos_cash",
            "credit_card",
            "all",
        ],
        help="Domain name to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    domains = [args.domain]
    if args.domain == "all":
        domains = [
            "application",
            "bureau",
            "previous_application",
            "installments",
            "pos_cash",
            "credit_card",
        ]

    for domain_name in domains:
        train_single_domain(config_path=Path(args.config), domain_name=domain_name)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def train_single_domain(config_path: Path, domain_name: str) -> None:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'lightgbm'. Install project dependencies with "
            "'python3 -m pip install -r requirements.txt' and rerun training."
        ) from exc

    log(f"[{domain_name}] Loading config from {config_path}")
    config = load_config(config_path)
    log(f"[{domain_name}] Data directory: {config.data_dir}")
    log(f"[{domain_name}] Artifacts directory: {config.artifacts_dir}")

    log(f"[{domain_name}] Reading application target table")
    application_target = pd.read_csv(config.data_dir / "application_train.csv", usecols=[ID_COL, TARGET_COL])
    log(f"[{domain_name}] Application target shape: {application_target.shape}")

    log(
        f"[{domain_name}] Building or loading shared split with holdout_size={config.holdout_size:.3f}"
    )
    split_df = make_or_load_holdout_split(
        application_train=application_target,
        split_dir=config.artifacts_dir / "splits",
        holdout_size=config.holdout_size,
        random_seed=config.random_seed,
    )
    split_counts = split_df["split"].value_counts().to_dict()
    log(f"[{domain_name}] Split counts: {split_counts}")

    log(f"[{domain_name}] Building domain dataset")
    domain_dataset = build_domain_dataset(config.data_dir, domain_name)
    log(
        f"[{domain_name}] Domain dataset shapes: train={domain_dataset.train.shape}, "
        f"test={domain_dataset.test.shape}"
    )

    log(f"[{domain_name}] Merging domain features with target and split")
    train_frame = application_target.merge(domain_dataset.train, on=ID_COL, how="left")
    train_frame = train_frame.merge(split_df[[ID_COL, "split"]], on=ID_COL, how="left")
    test_frame = domain_dataset.test.copy()
    log(f"[{domain_name}] Merged train shape: {train_frame.shape}")
    log(f"[{domain_name}] Test shape: {test_frame.shape}")

    dev_frame = train_frame.loc[train_frame["split"] == "development"].reset_index(drop=True)
    holdout_frame = train_frame.loc[train_frame["split"] == "holdout"].reset_index(drop=True)
    log(
        f"[{domain_name}] Development shape: {dev_frame.shape}, "
        f"holdout shape: {holdout_frame.shape}"
    )

    feature_cols = [col for col in train_frame.columns if col not in {ID_COL, TARGET_COL, "split"}]
    log(f"[{domain_name}] Feature count before preprocessing: {len(feature_cols)}")
    dev_x, holdout_x, test_x, categorical_cols = prepare_features(
        dev_frame[feature_cols],
        holdout_frame[feature_cols],
        test_frame[feature_cols],
    )
    dev_x, holdout_x, test_x, feature_name_map = sanitize_feature_names(dev_x, holdout_x, test_x)
    dev_y = dev_frame[TARGET_COL].to_numpy()
    holdout_y = holdout_frame[TARGET_COL].to_numpy()
    log(
        f"[{domain_name}] Prepared matrices: dev={dev_x.shape}, holdout={holdout_x.shape}, "
        f"test={test_x.shape}, categorical_features={len(categorical_cols)}"
    )
    renamed_count = sum(1 for original, sanitized in feature_name_map.items() if original != sanitized)
    if renamed_count:
        log(f"[{domain_name}] Sanitized {renamed_count} feature names for LightGBM compatibility")

    training_cfg = config.training
    base_params = dict(training_cfg["params"])
    base_params["seed"] = config.random_seed
    cv_folds = int(training_cfg["cv_folds"])
    param_grid = training_cfg.get("param_grid", {})

    if cv_folds <= 1:
        log(f"[{domain_name}] Debug mode detected: skipping CV and hyperparameter search")
        cv_results = [
            {
                "params": dict(base_params),
                "mean_auc": None,
                "std_auc": None,
                "mean_best_iteration": None,
            }
        ]
        best_result = cv_results[0]
        best_params = dict(base_params)
    else:
        param_grid_size = len(expand_param_grid(param_grid))
        log(
            f"[{domain_name}] Starting CV hyperparameter search: folds={cv_folds}, "
            f"candidates={param_grid_size}"
        )

        cv_results = run_cv_search(
            domain_name=domain_name,
            dev_x=dev_x,
            dev_y=dev_y,
            categorical_cols=categorical_cols,
            base_params=base_params,
            training_cfg=training_cfg,
            random_seed=config.random_seed,
        )
        best_result = max(cv_results, key=lambda item: item["mean_auc"])
        best_params = dict(best_result["params"])
        log(
            f"[{domain_name}] Best CV result: mean_auc={best_result['mean_auc']:.6f}, "
            f"std_auc={best_result['std_auc']:.6f}, mean_best_iteration={best_result['mean_best_iteration']}"
        )
        log(f"[{domain_name}] Best params: {best_params}")

    train_dataset = lgb.Dataset(dev_x, label=dev_y, categorical_feature=categorical_cols, free_raw_data=False)
    valid_dataset = lgb.Dataset(holdout_x, label=holdout_y, categorical_feature=categorical_cols, free_raw_data=False)
    log(f"[{domain_name}] Starting final training on full development set")

    booster = lgb.train(
        params=best_params,
        train_set=train_dataset,
        num_boost_round=int(training_cfg["num_boost_round"]),
        valid_sets=[train_dataset, valid_dataset],
        valid_names=["development", "holdout"],
        callbacks=[lgb.early_stopping(int(training_cfg["early_stopping_rounds"]), verbose=True)],
    )

    best_iteration = booster.best_iteration or int(training_cfg["num_boost_round"])
    log(f"[{domain_name}] Final training finished with best_iteration={best_iteration}")
    dev_pred = booster.predict(dev_x, num_iteration=best_iteration)
    holdout_pred = booster.predict(holdout_x, num_iteration=best_iteration)
    test_pred = booster.predict(test_x, num_iteration=best_iteration)

    metrics = {
        "domain": domain_name,
        "development_rows": int(len(dev_frame)),
        "holdout_rows": int(len(holdout_frame)),
        "num_features": int(len(feature_cols)),
        "best_iteration": int(best_iteration),
        "cv_mean_auc": None if best_result["mean_auc"] is None else float(best_result["mean_auc"]),
        "cv_std_auc": None if best_result["std_auc"] is None else float(best_result["std_auc"]),
        "development_auc": roc_auc_score_manual(dev_y, dev_pred),
        "holdout_auc": roc_auc_score_manual(holdout_y, holdout_pred),
    }
    log(
        f"[{domain_name}] Metrics: development_auc={metrics['development_auc']:.6f}, "
        f"holdout_auc={metrics['holdout_auc']:.6f}"
    )

    output_dir = ensure_dir(config.artifacts_dir / "domain_models" / domain_name)
    log(f"[{domain_name}] Saving artifacts to {output_dir}")
    save_pickle(output_dir / "model.pkl", booster)
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "metadata.json",
        {
            "domain": domain_name,
            "feature_columns": list(dev_x.columns),
            "categorical_columns": categorical_cols,
            "config_path": str(config_path),
            "best_params": best_params,
            "feature_name_map": feature_name_map,
        },
    )
    cv_results_df = pd.DataFrame(cv_results)
    if "mean_auc" in cv_results_df.columns and cv_results_df["mean_auc"].notna().any():
        cv_results_df = cv_results_df.sort_values("mean_auc", ascending=False)
    save_dataframe(output_dir / "cv_results.csv", cv_results_df)

    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    save_dataframe(output_dir / "feature_importance.csv", importance)

    save_dataframe(
        output_dir / "train_predictions.csv",
        pd.DataFrame(
            {
                ID_COL: dev_frame[ID_COL],
                "split": "development",
                "target": dev_y,
                f"{domain_name}_score": dev_pred,
            }
        ),
    )
    log(f"[{domain_name}] Domain training completed successfully")
    save_dataframe(
        output_dir / "holdout_predictions.csv",
        pd.DataFrame(
            {
                ID_COL: holdout_frame[ID_COL],
                "split": "holdout",
                "target": holdout_y,
                f"{domain_name}_score": holdout_pred,
            }
        ),
    )
    save_dataframe(
        output_dir / "test_predictions.csv",
        pd.DataFrame(
            {
                ID_COL: test_frame[ID_COL],
                f"{domain_name}_score": test_pred,
            }
        ),
    )


def prepare_features(
    dev_x: pd.DataFrame,
    holdout_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    combined = pd.concat([dev_x, holdout_x, test_x], axis=0, ignore_index=True)

    categorical_cols = []
    for col in combined.columns:
        if combined[col].dtype == "object":
            combined[col] = combined[col].fillna("__MISSING__").astype("category")
            categorical_cols.append(col)

    total_dev = len(dev_x)
    total_holdout = len(holdout_x)
    dev_prepared = combined.iloc[:total_dev].reset_index(drop=True)
    holdout_prepared = combined.iloc[total_dev:total_dev + total_holdout].reset_index(drop=True)
    test_prepared = combined.iloc[total_dev + total_holdout:].reset_index(drop=True)

    return dev_prepared, holdout_prepared, test_prepared, categorical_cols


def sanitize_feature_names(
    dev_x: pd.DataFrame,
    holdout_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    original_columns = list(dev_x.columns)
    sanitized_columns = make_safe_feature_names(original_columns)
    rename_map = dict(zip(original_columns, sanitized_columns))

    dev_x = dev_x.rename(columns=rename_map)
    holdout_x = holdout_x.rename(columns=rename_map)
    test_x = test_x.rename(columns=rename_map)

    return dev_x, holdout_x, test_x, rename_map


def make_safe_feature_names(columns: list[str]) -> list[str]:
    safe_names: list[str] = []
    seen: dict[str, int] = {}

    for idx, column in enumerate(columns):
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", column)
        safe = re.sub(r"_+", "_", safe).strip("_")
        if not safe:
            safe = f"feature_{idx}"

        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0

        safe_names.append(safe)

    return safe_names


def run_cv_search(
    domain_name: str,
    dev_x: pd.DataFrame,
    dev_y: np.ndarray,
    categorical_cols: list[str],
    base_params: dict,
    training_cfg: dict,
    random_seed: int,
) -> list[dict]:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'lightgbm'. Install project dependencies with "
            "'python3 -m pip install -r requirements.txt' and rerun training."
        ) from exc

    folds = make_stratified_folds(
        y=dev_y,
        n_folds=int(training_cfg["cv_folds"]),
        random_seed=random_seed,
    )
    param_grid = expand_param_grid(training_cfg.get("param_grid", {}))
    results: list[dict] = []

    for candidate_idx, param_overrides in enumerate(param_grid, start=1):
        params = dict(base_params)
        params.update(param_overrides)
        log(f"[{domain_name}] CV candidate {candidate_idx}/{len(param_grid)}: {param_overrides}")

        fold_scores = []
        fold_iterations = []

        for fold_idx, valid_index in enumerate(folds):
            train_mask = np.ones(len(dev_x), dtype=bool)
            train_mask[valid_index] = False
            train_index = np.where(train_mask)[0]

            fold_train_x = dev_x.iloc[train_index].reset_index(drop=True)
            fold_valid_x = dev_x.iloc[valid_index].reset_index(drop=True)
            fold_train_y = dev_y[train_index]
            fold_valid_y = dev_y[valid_index]

            train_dataset = lgb.Dataset(
                fold_train_x,
                label=fold_train_y,
                categorical_feature=categorical_cols,
                free_raw_data=False,
            )
            valid_dataset = lgb.Dataset(
                fold_valid_x,
                label=fold_valid_y,
                categorical_feature=categorical_cols,
                free_raw_data=False,
            )

            booster = lgb.train(
                params=params,
                train_set=train_dataset,
                num_boost_round=int(training_cfg["cv_num_boost_round"]),
                valid_sets=[valid_dataset],
                valid_names=[f"fold_{fold_idx}"],
                callbacks=[lgb.early_stopping(int(training_cfg["cv_early_stopping_rounds"]), verbose=False)],
            )

            best_iteration = booster.best_iteration or int(training_cfg["cv_num_boost_round"])
            fold_pred = booster.predict(fold_valid_x, num_iteration=best_iteration)
            fold_auc = roc_auc_score_manual(fold_valid_y, fold_pred)

            fold_scores.append(fold_auc)
            fold_iterations.append(best_iteration)
            log(
                f"[{domain_name}] Candidate {candidate_idx}/{len(param_grid)}, "
                f"fold {fold_idx + 1}/{len(folds)}: auc={fold_auc:.6f}, "
                f"best_iteration={best_iteration}, train_rows={len(train_index)}, "
                f"valid_rows={len(valid_index)}"
            )

        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        mean_best_iteration = int(round(float(np.mean(fold_iterations))))
        log(
            f"[{domain_name}] Candidate {candidate_idx}/{len(param_grid)} summary: "
            f"mean_auc={mean_auc:.6f}, std_auc={std_auc:.6f}, "
            f"mean_best_iteration={mean_best_iteration}"
        )
        results.append(
            {
                "params": params,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_best_iteration": mean_best_iteration,
            }
        )

    return results


def expand_param_grid(param_grid: dict) -> list[dict]:
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    combinations = []
    for combo in itertools.product(*values):
        combinations.append({key: value for key, value in zip(keys, combo)})
    return combinations


def make_stratified_folds(y: np.ndarray, n_folds: int, random_seed: int) -> list[np.ndarray]:
    if n_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    y = np.asarray(y, dtype=np.int64)
    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]

    if len(positive_idx) < n_folds or len(negative_idx) < n_folds:
        raise ValueError("Not enough examples in each class for the requested number of folds.")

    rng = np.random.default_rng(random_seed)
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)

    positive_folds = np.array_split(positive_idx, n_folds)
    negative_folds = np.array_split(negative_idx, n_folds)

    folds = []
    for fold_id in range(n_folds):
        fold_index = np.concatenate([positive_folds[fold_id], negative_folds[fold_id]])
        folds.append(np.sort(fold_index))

    return folds


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
