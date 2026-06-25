from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


@dataclass(frozen=True)
class DomainDataset:
    train: pd.DataFrame
    test: pd.DataFrame


def build_domain_dataset(data_dir, domain_name: str) -> DomainDataset:
    if domain_name == "application":
        return build_application_domain(data_dir)
    if domain_name == "bureau":
        return build_bureau_domain(data_dir)
    if domain_name == "previous_application":
        return build_previous_application_domain(data_dir)
    if domain_name == "installments":
        return build_installments_domain(data_dir)
    if domain_name == "pos_cash":
        return build_pos_cash_domain(data_dir)
    if domain_name == "credit_card":
        return build_credit_card_domain(data_dir)
    raise ValueError(f"Unknown domain: {domain_name}")


def build_application_domain(data_dir) -> DomainDataset:
    train = pd.read_csv(data_dir / "application_train.csv")
    test = pd.read_csv(data_dir / "application_test.csv")

    train = train.drop(columns=[TARGET_COL]).copy()
    return DomainDataset(train=train, test=test)


def build_bureau_domain(data_dir) -> DomainDataset:
    bureau = pd.read_csv(data_dir / "bureau.csv")
    bureau_balance = pd.read_csv(data_dir / "bureau_balance.csv")

    balance_features = aggregate_table(
        bureau_balance,
        group_key="SK_ID_BUREAU",
        prefix="bureau_balance",
    )
    bureau = bureau.merge(balance_features, on="SK_ID_BUREAU", how="left")

    train_ids, test_ids = load_base_ids(data_dir)
    domain_features = aggregate_table(
        bureau,
        group_key=ID_COL,
        prefix="bureau",
        drop_cols=["SK_ID_BUREAU"],
    )
    return split_train_test(domain_features, train_ids, test_ids)


def build_previous_application_domain(data_dir) -> DomainDataset:
    previous = pd.read_csv(data_dir / "previous_application.csv")
    train_ids, test_ids = load_base_ids(data_dir)
    domain_features = aggregate_table(
        previous,
        group_key=ID_COL,
        prefix="previous_application",
        drop_cols=["SK_ID_PREV"],
    )
    return split_train_test(domain_features, train_ids, test_ids)


def build_installments_domain(data_dir) -> DomainDataset:
    installments = pd.read_csv(data_dir / "installments_payments.csv")
    installments["payment_diff"] = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
    installments["days_diff"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]

    train_ids, test_ids = load_base_ids(data_dir)
    domain_features = aggregate_table(
        installments,
        group_key=ID_COL,
        prefix="installments",
        drop_cols=["SK_ID_PREV"],
    )
    return split_train_test(domain_features, train_ids, test_ids)


def build_pos_cash_domain(data_dir) -> DomainDataset:
    pos_cash = pd.read_csv(data_dir / "POS_CASH_balance.csv")
    train_ids, test_ids = load_base_ids(data_dir)
    domain_features = aggregate_table(
        pos_cash,
        group_key=ID_COL,
        prefix="pos_cash",
        drop_cols=["SK_ID_PREV"],
    )
    return split_train_test(domain_features, train_ids, test_ids)


def build_credit_card_domain(data_dir) -> DomainDataset:
    credit_card = pd.read_csv(data_dir / "credit_card_balance.csv")
    train_ids, test_ids = load_base_ids(data_dir)
    domain_features = aggregate_table(
        credit_card,
        group_key=ID_COL,
        prefix="credit_card",
        drop_cols=["SK_ID_PREV"],
    )
    return split_train_test(domain_features, train_ids, test_ids)


def load_base_ids(data_dir) -> tuple[pd.Series, pd.Series]:
    application_train = pd.read_csv(data_dir / "application_train.csv", usecols=[ID_COL])
    application_test = pd.read_csv(data_dir / "application_test.csv", usecols=[ID_COL])
    return application_train[ID_COL], application_test[ID_COL]


def split_train_test(
    domain_features: pd.DataFrame,
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> DomainDataset:
    train = pd.DataFrame({ID_COL: train_ids}).merge(domain_features, on=ID_COL, how="left")
    test = pd.DataFrame({ID_COL: test_ids}).merge(domain_features, on=ID_COL, how="left")
    return DomainDataset(train=train, test=test)


def aggregate_table(
    frame: pd.DataFrame,
    group_key: str,
    prefix: str,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    drop_cols = drop_cols or []

    for col in drop_cols:
        if col in frame.columns:
            frame = frame.drop(columns=[col])

    categorical_cols = [
        col for col in frame.columns
        if col != group_key and frame[col].dtype == "object"
    ]
    numeric_cols = [
        col for col in frame.columns
        if col != group_key and col not in categorical_cols
    ]

    pieces: list[pd.DataFrame] = []

    if numeric_cols:
        numeric_agg = frame.groupby(group_key)[numeric_cols].agg(["count", "mean", "min", "max", "sum", "std"])
        numeric_agg.columns = [
            f"{prefix}__{col}__{stat}"
            for col, stat in numeric_agg.columns.to_flat_index()
        ]
        pieces.append(numeric_agg)

    if categorical_cols:
        dummies = pd.get_dummies(frame[[group_key] + categorical_cols], columns=categorical_cols, dummy_na=True)
        categorical_agg = dummies.groupby(group_key).agg(["mean", "sum"])
        categorical_agg.columns = [
            f"{prefix}__{col}__{stat}"
            for col, stat in categorical_agg.columns.to_flat_index()
            if col != group_key
        ]
        pieces.append(categorical_agg)

    if not pieces:
        return frame[[group_key]].drop_duplicates()

    aggregated = pd.concat(pieces, axis=1).reset_index()
    return aggregated
