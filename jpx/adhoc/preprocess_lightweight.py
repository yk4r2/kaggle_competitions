import pandas as pd
import numpy as np
from itertools import chain
import re
from pathlib import Path


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", 1000)

camel_to_snake = re.compile(r"(?<!^)(?=[A-Z])")

OPTIONS_COLUMNS = [
    "puts_implied_volatility_0d",
    "puts_trading_value_0d",
    "puts_settlement_price_0d",
    "puts_settlement_to_theoretical_30d",
    "puts_price_to_opt_0d",
    "puts_estimated_price_0d",
    "puts_whole_day_close_0d",
    "puts_estimated_price_30d",
    "puts_estimated_price_0d_prev_day_ratio",
    "puts_price_to_opt_30d",
    "puts_settlement_to_theoretical_0d",
    "calls_settlement_to_theoretical_30d",
    "calls_implied_volatility_0d",
    "calls_estimated_price_0d",
    "calls_price_to_opt_0d",
    "calls_trading_value_30d",
    "calls_whole_day_close_30d",
    "calls_estimated_price_0d_prev_day_ratio",
    "calls_implied_volatility_0d_prev_day_ratio",
    "calls_trading_volume_0d_prev_day_ratio",
    "calls_trading_volume_30d_prev_day_ratio",
    "calls_whole_day_volume_0d",
    "calls_strike_price_0d",
    "calls_settlement_to_theoretical_0d",
]

WINDOWS = ("2d", "7d", "15d", "30d")


prices_cols = ["Close", "Volume", "SecuritiesCode", "Date", "Target"]
groupers = [
    "NewIndexSeriesSize",
    "NewMarketSegment",
    "Section/Products",
    "33SectorName",
    "17SectorName",
]


def preprocess_prices(
    stock_prices: pd.DataFrame,
    secondary_prices: pd.DataFrame,
    stock_list: pd.DataFrame,
    index_data: pd.DataFrame,
) -> pd.DataFrame:
    stock_prices = stock_prices.sort_values(by="Date")
    stock_prices["high_low_close"] = (stock_prices["High"] - stock_prices["Low"]).div(
        stock_prices["Close"]
    )
    secondary_prices = secondary_prices.sort_values(by="Date")
    secondary_prices["high_low_close"] = (
        secondary_prices["High"] - secondary_prices["Low"]
    ).div(secondary_prices["Close"])

    secondary_prices = secondary_prices[prices_cols]
    secondary_prices_full = secondary_prices.merge(
        stock_list, on="SecuritiesCode", how="left"
    )

    stock_prices = stock_prices[prices_cols]
    stock_prices["target_shift"] = stock_prices["Target"].shift(2)

    stock_prices_full = stock_prices.merge(stock_list, on="SecuritiesCode", how="left")

    all_stock_prices = pd.concat([stock_prices_full, secondary_prices_full], axis=0)

    stock_prices_full = stock_prices_full.set_index("Date")
    all_stock_prices = all_stock_prices.set_index("Date").drop(columns=["Target"])
    index_data = index_data.set_index("Date")

    stock_prices_full["close_pct"] = stock_prices_full.groupby("SecuritiesCode")[
        "Close"
    ].pct_change()
    stock_prices_full["volume_pct"] = stock_prices_full.groupby("SecuritiesCode")[
        "Volume"
    ].pct_change()

    for win in WINDOWS:
        print(f"{win} started")
        new_index_data = index_data.rolling(win).mean()

        stock_prices_full = stock_prices_full.merge(
            new_index_data,
            how="left",
            on="Date",
            suffixes=("", f"_nikkei_{win}"),
        )

        daily_cluster_mean = (
            all_stock_prices.groupby(groupers + ["Date"])
            .mean()
            .reset_index()
            .drop(columns="SecuritiesCode")
        )
        new_stock_prices = (
            daily_cluster_mean.set_index("Date").groupby(groupers).rolling(win).mean()
        )

        stock_prices_full = stock_prices_full.merge(
            new_stock_prices,
            how="left",
            on=groupers + ["Date"],
            suffixes=("", f"_cluster_{win}"),
        )

        daily_stock_mean = (
            all_stock_prices.groupby(["SecuritiesCode", "Date"]).mean().reset_index()
        )
        daily_stock_prices = (
            daily_stock_mean.set_index("Date")
            .groupby("SecuritiesCode")
            .rolling(win)
            .mean()
        )

        stock_prices_full = stock_prices_full.merge(
            daily_stock_prices,
            how="left",
            on=["SecuritiesCode", "Date"],
            suffixes=("", f"_mean_{win}"),
        )
        print(f"{win} done")

    stock_prices_full = stock_prices_full.set_index(["SecuritiesCode"], append=True)
    stock_prices_full = stock_prices_full.groupby(
        level=["Date", "SecuritiesCode"]
    ).ffill()
    for win in WINDOWS:
        vol_nikkei_str = "Volume_nikkei_" + win
        vol_cluster_str = "Volume_cluster_" + win
        vol_mean_str = "Volume_mean_" + win

        close_nikkei_str = "Close_nikkei_" + win
        close_cluster_str = "Close_cluster_" + win
        close_mean_str = "Close_mean_" + win

        target_mean_str = "target_shift_mean_" + win
        target_cluster_str = "target_shift_cluster_" + win

        stock_prices_full[vol_nikkei_str] = stock_prices_full["Volume"].div(
            stock_prices_full[vol_nikkei_str]
        )
        stock_prices_full[vol_cluster_str] = stock_prices_full["Volume"].div(
            stock_prices_full[vol_cluster_str]
        )
        stock_prices_full[vol_mean_str] = stock_prices_full["Volume"].div(
            stock_prices_full[vol_mean_str]
        )

        stock_prices_full[close_nikkei_str] = stock_prices_full["Volume"].div(
            stock_prices_full[close_nikkei_str]
        )
        stock_prices_full[close_cluster_str] = stock_prices_full["Volume"].div(
            stock_prices_full[close_cluster_str]
        )
        stock_prices_full[close_mean_str] = stock_prices_full["Volume"].div(
            stock_prices_full[close_mean_str]
        )

        stock_prices_full[target_mean_str + "_div"] = stock_prices_full["Volume"].div(
            stock_prices_full[target_mean_str]
        )
        stock_prices_full[target_cluster_str + "_div"] = stock_prices_full[
            "Volume"
        ].div(stock_prices_full[target_cluster_str])

    target_cols = stock_prices_full.columns[
        stock_prices_full.columns.str.contains("target")
    ]
    rank_cols = "rank" + target_cols.str.lstrip("target")
    stock_prices_full[rank_cols] = stock_prices_full.groupby(level=["Date"])[
        target_cols
    ].rank()
    stock_prices_full["weekday"] = stock_prices_full.index.get_level_values(0).day

    stock_prices_full.drop(columns=groupers).drop(columns=target_cols)
    stock_prices_full = stock_prices_full.rename(
        columns=lambda x: camel_to_snake.sub("_", x).lower()
    )
    return stock_prices_full.rename_axis(["date", "securities_code"]).dropna(
        subset=["target"]
    )


def preprocess_trades(trades: pd.DataFrame) -> pd.DataFrame:
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
    start_columns = list(
        chain(sales_columns, purch_columns, balance_columns, total_columns)
    )

    for sales, purch, balance, total in zip(
        sales_columns, purch_columns, balance_columns, total_columns
    ):
        colname = camel_to_snake.sub("_", sales[:-5]).lower()
        purch_to_total = colname + "_purchases_to_total"
        purch_to_balance = colname + "_purchases_to_balance"
        sales_to_total = colname + "_sales_to_total"
        balance_to_total = colname + "_balance_to_total"

        trades[purch_to_total] = trades[purch].div(trades[total])
        trades[purch_to_balance] = trades[purch].div(trades[balance])
        trades[sales_to_total] = trades[sales].div(trades[total])
        trades[balance_to_total] = trades[balance].div(trades[total])

    for win in WINDOWS:
        appendix = (
            trades[start_columns]
            .div(trades[start_columns].rolling(win).mean())
            .add_suffix("_" + win)
        )
        trades = pd.concat([trades, appendix], axis=1)

    trades = trades.drop(
        chain(sales_columns, purch_columns, balance_columns, total_columns), axis=1
    )
    trades = trades.rename(columns=lambda x: camel_to_snake.sub("_", x).lower())
    return trades.rename_axis("date")


def preprocess_options(
    options: pd.DataFrame,
    index_data: pd.DataFrame,
) -> pd.DataFrame:
    options = options.drop("Dividend", axis=1)
    options = options.loc[options["TradingVolume"] != 0]
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
    options = options[options["bucket"].isin([pd.Timedelta("0D"), pd.Timedelta("30D")])]
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

    options = options.groupby(["Date", "Putcall", "bucket"]).sum()
    options = options.reset_index().dropna()

    # index
    merged_options = index_data.merge(options, on="Date")

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
    return options[options.columns.intersection(OPTIONS_COLUMNS)]


def get_features(
    stock_prices: pd.DataFrame,
    secondary_prices: pd.DataFrame,
    stock_list: pd.DataFrame,
    index_data: pd.DataFrame,
    options: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    prices = preprocess_prices(stock_prices, secondary_prices, stock_list, index_data)
    prices = prices[pd.Timestamp("2017-01-05") :]
    options = preprocess_options(options, index_data)
    trades = preprocess_trades(trades)
    prices = (
        prices.reset_index()
        .merge(trades, how="left", on="date")
        .merge(options, how="left", on="date")
        .set_index(["date", "securities_code"])
    )
    prices = prices.groupby(level=[0, 1]).ffill()
    prices["rank"] = prices.groupby("date")["target"].rank().astype(int)
    prices = prices.replace([np.inf, -np.inf], np.nan)
    return prices


def get_features_from_path(data_root: Path) -> pd.DataFrame:
    options = pd.read_csv(data_root / "options.csv", parse_dates=["Date"])
    trades = pd.read_csv(
        data_root / "trades.csv", parse_dates=["Date", "StartDate", "EndDate"]
    )
    secondary_prices = pd.read_csv(
        data_root / "secondary_stock_prices.csv", parse_dates=["Date"]
    )
    stock_prices = pd.read_csv(data_root / "stock_prices.csv", parse_dates=["Date"])

    options = options.sort_values(by="Date")
    trades = trades.sort_values(by="Date")
    secondary_prices = secondary_prices.sort_values(by="Date")
    stock_prices = stock_prices.sort_values(by="Date")

    index_selected = 1321
    index_cols = ["Date", "Close", "Volume"]
    index_data = secondary_prices.loc[
        secondary_prices["SecuritiesCode"] == index_selected
    ]
    index_data = index_data.loc[:, index_cols]

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
    stock_list = (
        pd.read_csv(data_root / "../stock_list.csv")
        .drop(unused_cols, axis=1)
        .replace("－", None)
    )
    return get_features(
        stock_prices, secondary_prices, stock_list, index_data, options, trades
    )
