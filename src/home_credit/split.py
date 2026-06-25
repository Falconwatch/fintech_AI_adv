from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import ensure_dir


def make_or_load_holdout_split(
    application_train: pd.DataFrame,
    split_dir: Path,
    holdout_size: float,
    random_seed: int,
) -> pd.DataFrame:
    split_path = split_dir / "train_holdout_split.csv"
    if split_path.exists():
        return pd.read_csv(split_path)

    if not 0.0 < holdout_size < 1.0:
        raise ValueError("holdout_size must be in (0, 1).")

    rng = np.random.default_rng(random_seed)
    positives = application_train.loc[application_train["TARGET"] == 1, "SK_ID_CURR"].to_numpy()
    negatives = application_train.loc[application_train["TARGET"] == 0, "SK_ID_CURR"].to_numpy()

    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_holdout = max(1, int(round(len(positives) * holdout_size)))
    neg_holdout = max(1, int(round(len(negatives) * holdout_size)))

    holdout_ids = set(positives[:pos_holdout].tolist() + negatives[:neg_holdout].tolist())

    split_df = application_train[["SK_ID_CURR", "TARGET"]].copy()
    split_df["split"] = np.where(
        split_df["SK_ID_CURR"].isin(holdout_ids),
        "holdout",
        "development",
    )

    ensure_dir(split_dir)
    split_df.to_csv(split_path, index=False)
    return split_df
