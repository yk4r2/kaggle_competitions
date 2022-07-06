from itertools import chain
import numpy as np
import pandas as pd
import re
from itertools import combinations

camel_to_snake = re.compile(r"(?<!^)(?=[A-Z])")

unused_cols = [
    "EffectiveDate",
    "TradeDate",
    "Close",
    "IssuedShares",
    "MarketCapitalization",
    "Name",
    "33SectorCode",
    "17SectorCode",
    "NewIndexSeriesSizeCode",
    "Universe0",
]
stock_list = pd.read_csv("../data/stock_list.csv").drop(unused_cols, axis=1)
groupers = [
    "NewIndexSeriesSize",
    "NewMarketSegment",
    "Section/Products",
    "33SectorName",
    "17SectorName",
    "Date",
]


def cpcv_split(
    folds: int = 5,
    test_folds: int = 3,
    start_embargo: bool = True,
) -> dict:
    if test_folds > folds:
        raise ValueError("Test folds count should be smaller than overall folds.")
    test_variants = list(combinations(range(folds), test_folds))

    splits = []
    for test_parts in test_variants:
        train_parts = list(set(range(folds)).difference(test_parts))
        parts = {k: "test" for k in test_parts}

        if start_embargo:
            parts[-1] = "embargo"
        parts.update({k: "train" for k in train_parts})

        # embargo adding
        parts.update({k - 0.5: "embargo" for k in test_parts})
        parts.update({k + 0.5: "embargo" for k in test_parts})

        parts = dict(sorted(parts.items(), key=lambda item: item[0]))
        # reindex
        parts = {k: v for k, v in zip(range(len(parts)), parts.values())}

        purged_elements = []
        if parts[0] == "embargo" and parts[1] == "embargo":
            purged_elements = [0, 1] if parts[2] == "embargo" else [0]

        left = 0
        while left < len(parts) - 2:
            if (
                parts[left] == "test"
                and parts[left + 1] == "embargo"
                and parts[left + 2] == "test"
            ):
                purged_elements.append(left + 1)
                left += 1
            left += 1

        if parts[len(parts) - 1] == "embargo":
            purged_elements.append(len(parts) - 1)

        parts = {k: parts[k] for k in range(len(parts)) if k not in purged_elements}
        # reindex
        parts = {k: v for k, v in zip(range(len(parts)), parts.values())}
        splits.append(parts)
    return splits


def cpcv(
    timeline: pd.DatetimeIndex,
    embargo_size: pd.Timedelta = pd.Timedelta(days=30),
    folds: int = 5,
    test_folds: int = 3,
    start_embargo: bool = True,
):
    split_dicts = cpcv_split(folds, test_folds, start_embargo)
    print(f"splits count: {len(split_dicts)}\n")
    splits = []
    for idx, split in enumerate(split_dicts):
        print(f"split number {idx}")
        embargo_delta = (
            len(list(filter(lambda x: x == "embargo", split.values()))) * embargo_size
        )
        timeline_delta = np.timedelta64(max(timeline) - min(timeline), "D")

        if embargo_delta > timeline_delta:
            raise ValueError(
                "Embargo is greater than the whole timeline!\n",
                f"emb: {embargo_delta}, timeline delta: {timeline_delta}.",
            )
        fold_size = np.timedelta64((timeline_delta - embargo_delta) // folds, "D")
        print(f"fold size: {fold_size}")

        if fold_size < pd.Timedelta(days=30):
            raise ValueError(
                "Fold size is less than 30 days!\n", f"fold size: {fold_size}"
            )

        timedeltas = []
        prev_timedelta = pd.Timedelta(days=0)
        for count, value in split.items():
            delta = fold_size if value in {"test", "train"} else embargo_size
            timedeltas.append(delta + prev_timedelta)
            prev_timedelta += delta

        splits.append((timedeltas, list(split.values())))
    return splits


def get_cpcv_timelines(
    timeline: pd.DatetimeIndex,
    embargo_size: pd.Timedelta = pd.Timedelta(days=30),
    folds: int = 5,
    test_folds: int = 3,
    start_embargo: bool = True,
):
    min_date = timeline.min()
    splits = cpcv(timeline, embargo_size, folds, test_folds, start_embargo)
    #     print("splits", splits)

    timelines = []
    for deltas, names in splits:
        deltas_cp = deltas[:]
        deltas_cp.insert(0, pd.Timedelta(days=0))
        left, right = 0, 1
        pairs = []
        while right < len(deltas_cp):
            pairs.append(
                (
                    min_date + deltas_cp[left],
                    min_date + deltas_cp[right] - pd.Timedelta(days=1),
                )
            )
            left += 1
            right += 1
        timelines.append((pairs, names))
    return timelines


def get_train_test_cpcv(timeline: tuple, indices: pd.DatetimeIndex):
    train_idx = pd.DatetimeIndex([])
    test_idx = pd.DatetimeIndex([])
    for starts_stops, name in zip(*timeline):
        if name == "train":
            train_idx = train_idx.append(
                indices.intersection(pd.bdate_range(*starts_stops))
            )
        elif name == "test":
            test_idx = test_idx.append(
                indices.intersection(pd.bdate_range(*starts_stops))
            )
    return train_idx, test_idx


def preprocess_options(options: pd.DataFrame, index_data: pd.DataFrame):
    options = options.drop("Dividend", axis=1)
    options = options.loc[options["TradingVolume"] != 0]
    options["Date"] = pd.to_datetime(options["Date"].astype(str), format="%Y-%m-%d")
    options["exercise_start"] = pd.to_datetime(
        options["ContractMonth"].astype(str), format="%Y%m"
    )
    options["time_delta"] = options["exercise_start"] - options["Date"]
    options["settlement_to_theoretical"] = (
        options["SettlementPrice"] / options["TheoreticalPrice"]
    )

    columns_needed = [
        "Date",
        "Putcall",
        "WholeDayClose",
        "TradingValue",
        "TradingVolume",
        "WholeDayVolume",
        "ImpliedVolatility",
        "SettlementPrice",
        "settlement_to_theoretical",
        "StrikePrice",
        "time_delta",
    ]
    options = options.loc[:, columns_needed]

    buckets = ["-30D", "0D", "30D", "60D", "150D", "9999D"]
    buckets = list(map(pd.Timedelta, buckets))
    options.loc[:, "bucket"] = pd.cut(
        options["time_delta"], buckets, labels=buckets[:-1]
    )
    options.loc[:, "volume_weight"] = options.groupby(["Date", "Putcall", "bucket"])[
        "WholeDayVolume"
    ].apply(lambda x: x / x.sum())

    columns_divided = [
        "WholeDayClose",
        "TradingValue",
        "ImpliedVolatility",
        "StrikePrice",
        "SettlementPrice",
    ]
    options.loc[:, columns_divided] = options.loc[:, columns_divided].mul(
        options["volume_weight"], axis=0
    )
    options = options.drop(columns="volume_weight")

    opts_active = options.loc[options["bucket"] == pd.Timedelta("-30D")]

    puts_active = opts_active.loc[opts_active["Putcall"] == 1]
    puts_active = puts_active.groupby("Date").sum().resample("B").ffill()
    puts_active["Putcall"] = 1

    calls_active = opts_active.loc[opts_active["Putcall"] == 2]
    calls_active = calls_active.groupby("Date").sum().resample("B").ffill()
    calls_active["Putcall"] = 2

    opts_active = pd.concat([puts_active, calls_active]).reset_index()
    opts_active["bucket"] = pd.Timedelta("-29D")

    opts_active["bucket"] = pd.cut(opts_active["bucket"], buckets, labels=buckets[:-1])

    opts_inactive = options.loc[options["bucket"] != pd.Timedelta("-30D")]
    opts_inactive = opts_inactive.groupby(["Date", "Putcall", "bucket"]).sum()
    opts_inactive = opts_inactive.reset_index().dropna()
    opts_aggregated = pd.concat([opts_inactive, opts_active])

    # index
    index_data.loc[:, "Date"] = pd.to_datetime(
        index_data.loc[:, "Date"].astype(str), format="%Y-%m-%d"
    )
    merged_options = index_data.merge(opts_aggregated, on="Date")

    merged_options["estimated_price"] = (
        merged_options["SettlementPrice"]
        + merged_options["Close"]
        + merged_options["WholeDayClose"]
    )
    merged_options.loc[:, "price_to_opt"] = (
        merged_options["estimated_price"] / merged_options["Close"]
    )
    merged_options = merged_options.groupby(["Date", "Putcall", "bucket"]).mean()
    merged_options = merged_options.reset_index()

    merged_options["volume_share"] = merged_options.groupby(["Date", "bucket"])[
        "TradingVolume"
    ].apply(lambda x: x / x.sum())

    merged_options = merged_options.rename(
        columns=lambda x: camel_to_snake.sub("_", x).lower()
    )
    merged_options = merged_options.set_index("date")
    merged_options = merged_options[merged_options["bucket"] < pd.Timedelta(days=60)]
    merged_options = merged_options.sort_values(by=["date", "bucket"])
    numerical_cols = merged_options.select_dtypes(include=np.number).columns.difference(
        ["putcall"]
    )

    puts = merged_options[merged_options["putcall"] == 1].drop("putcall", axis=1)
    puts[numerical_cols] = puts[numerical_cols].div(
        puts[numerical_cols].groupby("date").shift(1)
    )
    calls = merged_options[merged_options["putcall"] == 2].drop("putcall", axis=1)
    calls[numerical_cols] = calls[numerical_cols].div(
        calls[numerical_cols].groupby("date").shift(1)
    )

    puts_pivoted = puts.pivot(columns="bucket")
    puts_pivoted.columns = [
        "{}_{}d".format(stat, bucket.days) for stat, bucket in puts_pivoted.columns
    ]
    calls_pivoted = calls.pivot(columns="bucket")
    calls_pivoted.columns = [
        "{}_{}d".format(stat, bucket.days) for stat, bucket in calls_pivoted.columns
    ]

    options = pd.concat(
        [calls_pivoted.add_prefix("calls_"), puts_pivoted.add_prefix("puts_")], axis=1
    )
    options = options.dropna(axis=1, how="all")
    return options.drop(
        columns=[
            "calls_close_0d",
            "calls_close_30d",
            "calls_volume_0d",
            "calls_volume_30d",
        ]
    )


def preprocess_financials(financials: pd.DataFrame):
    bad_cols = [
        "DateCode",
        "DisclosedDate",
        "DisclosedTime",
        "DisclosedUnixTime",
        "CurrentFiscalYearStartDate",
        "CurrentFiscalYearEndDate",
        "ApplyingOfSpecificAccountingOfTheQuarterlyFinancialStatements",
        "ForecastDividendPerShareFiscalYearEnd",
        "ForecastNetSales",
        "ForecastOperatingProfit",
        "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        "OperatingProfit",
        "OrdinaryProfit",
    ]
    financials = financials.drop(bad_cols, axis=1)
    financials = financials.replace("－", None)
    financials.sample(5)

    financials["is_nonconsolidated"] = (
        financials["TypeOfDocument"].str.contains("NonConsolidated").astype(bool)
    )
    financials["is_consolidated"] = (
        financials["TypeOfDocument"].str.contains("Consolidated").astype(bool)
        & ~financials["is_nonconsolidated"]
    )
    financials["agency"] = financials["TypeOfDocument"].str.split("_").str[-1]
    financials["is_correction"] = (
        financials["TypeOfDocument"].str.contains("NumericalCorrection").astype(bool)
    )
    financials["is_forecast"] = (
        financials["TypeOfDocument"].str.contains("ForecastRevision").astype(bool)
    )

    financials = financials.drop(["TypeOfDocument"], axis=1)

    for col in financials.columns.difference(
        ["Date", "CurrentPeriodEndDate", "TypeOfCurrentPeriod", "agency"]
    ):
        try:
            financials[col] = pd.to_numeric(financials[col], errors="coerce")
        except Exception as e:
            print(f"exc {e}, col {col}")

    financials_full = financials.merge(stock_list, on="SecuritiesCode")

    financials_avg = (
        financials_full.groupby(groupers).mean().drop(columns="SecuritiesCode")
    )

    divided = financials_full.columns.difference(
        [
            "SecuritiesCode",
            "TypeOfCurrentPeriod",
            "CurrentPeriodEndDate",
            "agency",
            "is_consolidated",
            "is_correction",
            "is_forecast",
            "is_nonconsolidated",
        ]
        + groupers
    )
    divisors = [divisor + "_cluster_avg" for divisor in divided]

    financials_merged = financials_full.merge(
        financials_avg,
        on=groupers,
        how="left",
        suffixes=("", "_cluster_avg"),
    )
    financials_merged = financials_merged.drop(columns=groupers[:-1])
    financials = financials.sort_values(by="Date").groupby("SecuritiesCode").ffill()

    financials_merged[divisors] = financials_merged[divided].div(
        financials_merged[divisors].rename(
            columns=lambda x: x.replace("_cluster_avg", "")
        )
    )
    financials_merged = financials_merged.rename(
        columns=lambda x: camel_to_snake.sub("_", x).lower()
    )
    financials_merged.drop(
        columns=["current_period_end_date", "type_of_current_period"]
    )
    return financials_merged


def preprocess_trades(trades: pd.DataFrame):
    trades = trades.rename(
        {"CityBKsRegionalBKsEtcPurchase": "CityBKsRegionalBKsEtcPurchases"}, axis=1
    )
    trades = trades.dropna()
    trades = trades.drop(["StartDate", "EndDate", "Section"], axis=1)
    trades = trades.groupby("Date").sum()

    sales_columns = trades.columns[trades.columns.str.endswith("Sales")]
    purch_columns = trades.columns[trades.columns.str.endswith("Purchases")]
    balance_columns = trades.columns[trades.columns.str.endswith("Balance")]
    total_columns = trades.columns[trades.columns.str.endswith("Total")]

    saved_columns = []
    for sales, purch, balance, total in zip(
        sales_columns, purch_columns, balance_columns, total_columns
    ):
        colname = camel_to_snake.sub("_", sales[:-5]).lower()
        purch_to_total = colname + "_purchases_to_total"
        purch_to_balance = colname + "_purchases_to_balance"
        sales_to_total = colname + "_sales_to_total"
        sales_to_balance = colname + "_sales_to_balance"
        balance_to_total = colname + "_balance_to_total"

        trades[purch_to_total] = trades[purch].div(trades[total])
        trades[purch_to_balance] = trades[purch].div(trades[balance])
        trades[sales_to_total] = trades[sales].div(trades[total])
        trades[sales_to_balance] = trades[sales].div(trades[balance])
        trades[balance_to_total] = trades[balance].div(trades[total])

        saved_columns += [
            purch_to_total,
            purch_to_balance,
            sales_to_total,
            sales_to_balance,
            balance_to_total,
        ]

    trades = trades.drop(
        chain(sales_columns, purch_columns, balance_columns, total_columns), axis=1
    )
    return trades.rename_axis("date")


def preprocess_prices(stock_prices: pd.DataFrame, secondary_prices: pd.DataFrame):
    secondary_prices = secondary_prices.drop(
        ["AdjustmentFactor", "SupervisionFlag", "ExpectedDividend", "RowId"], axis=1
    )
    secondary_prices_full = secondary_prices.merge(stock_list, on="SecuritiesCode")

    secondary_prices_avg = secondary_prices_full.sort_values(by="Date").drop(
        columns=["Target"]
    )
    secondary_prices_avg = (
        secondary_prices_avg.set_index("Date")
        .groupby(groupers[:-1])
        .rolling("7D")
        .mean()
        .reset_index()
    )
    secondary_prices_avg = secondary_prices_avg.drop("SecuritiesCode", axis=1).dropna(
        subset=["Close"]
    )
    secondary_prices_avg = (
        secondary_prices_avg.set_index("Date")
        .groupby(groupers[:-1])
        .resample("B")
        .last()
        .drop(groupers[:-1], axis=1)
        .ffill()
    )

    stock_prices = stock_prices.drop(
        ["AdjustmentFactor", "SupervisionFlag", "ExpectedDividend"], axis=1
    )
    stock_prices_full = stock_prices.merge(stock_list, on="SecuritiesCode")

    stock_prices_avg = stock_prices_full.sort_values(by="Date").drop(columns="Target")
    stock_prices_avg = stock_prices_avg.groupby(groupers).mean()
    stock_prices_avg = stock_prices_avg.drop_duplicates()
    stock_prices_avg = stock_prices_avg.drop("SecuritiesCode", axis=1)
    stock_prices_avg = stock_prices_avg.dropna(subset=["Close"])

    stock_prices_merged = stock_prices_full.merge(
        stock_prices_avg,
        on=groupers,
        how="left",
        suffixes=("", "_avg_cluster"),
    )
    stock_prices_merged = stock_prices_merged.merge(
        secondary_prices_avg,
        on=groupers,
        how="left",
        suffixes=("", "_avg_2nd_cluster"),
    )

    stock_prices_merged = stock_prices_merged.set_index(
        ["Date", "SecuritiesCode"]
    ).drop(columns="RowId")

    numeric_cols = stock_prices_merged.select_dtypes(include=np.number).columns
    divided_cols = numeric_cols[~numeric_cols.str.contains("target")]
    stock_prices_merged[divided_cols] = stock_prices_merged[divided_cols].div(
        stock_prices_merged.groupby(groupers[:-1])[divided_cols].shift(1).bfill()
    )

    stock_prices_merged = (
        stock_prices_merged.reset_index()
        .rename(columns=lambda x: camel_to_snake.sub("_", x).lower())
        .set_index(["date", "securities_code"])
    )

    column_names = ["open", "high", "low", "close", "volume"]
    for name in column_names:
        cols_with_name = stock_prices_merged.columns.str.startswith(name)

        suffixed_cols = stock_prices_merged.columns.str.contains("_")
        divisor_cols = stock_prices_merged.columns[cols_with_name & suffixed_cols]
        print("divisor_cols", divisor_cols)

        for divisor in divisor_cols:
            stock_prices_merged.loc[:, divisor] = stock_prices_merged[name].div(
                stock_prices_merged[divisor], axis=0
            )
    return stock_prices_merged
