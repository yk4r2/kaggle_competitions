from functools import reduce
from typing import List
from typing import Tuple

from tqdm.notebook import tqdm
from catboost import CatBoostRanker, Pool
from pytorch_tabnet.tab_model import TabNetRegressor
import pandas as pd


cb_good_columns = [
 'rank_shift_cluster_2d',
 'rank_shift',
 'target_shift_cluster_2d',
 'target_shift_cluster_2d_div',
 'close_pct',
 'rank_shift_cluster_2d_div',
 'target_shift_cluster_15d',
 'target_shift',
 'volume_nikkei_2d',
 'individuals_purchases_to_balance',
 'rank_shift_cluster_7d',
 'rank_shift_cluster_7d_div',
 'volume_nikkei_7d',
 'volume_nikkei_30d',
 'close_nikkei_15d',
 'close_nikkei_7d',
 'city_b_ks_regional_b_ks_etc_balance_30d',
 'proprietary_balance_15d',
 'rank_shift_mean_2d_div',
 'volume',
 'close_nikkei_2d',
 'close_nikkei_30d',
 'business_cos_purchases_15d',
 'individuals_balance_to_total',
 'target_shift_cluster_7d_div',
 'business_cos_balance_15d',
 'securities_cos_sales_to_total',
 '17_sector_name',
 'rank_shift_cluster_15d_div',
 'foreigners_balance_30d',
 'securities_cos_balance_15d',
 'rank_shift_mean_30d',
 'volume_nikkei_15d',
 'close_cluster_2d',
 'target_shift_mean_30d_div',
 'trust_banks_sales_to_total',
 'individuals_balance_30d',
 'securities_cos_purchases_to_total',
 'close_cluster_30d',
 'volume_cluster_15d',
 'close_cluster_7d',
 'city_b_ks_regional_b_ks_etc_purchases_30d',
 'investment_trusts_balance_7d',
 'target_shift_mean_7d_div',
 'city_b_ks_regional_b_ks_etc_sales_30d',
 'volume_cluster_2d',
 'foreigners_balance_7d',
 'investment_trusts_sales_30d',
 'individuals_sales_30d',
 'securities_cos_balance_7d',
 'foreigners_total_15d',
 'foreigners_sales_to_total',
 'other_financial_institutions_purchases_30d',
 'trust_banks_sales_15d',
]


tn_good_columns = [
 'proprietary_total_15d',
 'rank_shift_mean_15d',
 'other_institutions_total_7d',
 'other_financial_institutions_sales_to_total',
 'other_financial_institutions_purchases_to_balance',
 'other_financial_institutions_sales_15d',
 'other_institutions_sales_2d',
 'securities_cos_sales_7d',
 'target_shift_mean_15d',
 'city_b_ks_regional_b_ks_etc_purchases_to_balance',
 'brokerage_sales_30d',
 'calls_whole_day_close_30d',
]


all_good_columns = sorted(cb_good_columns + tn_good_columns)

target_columns = ['target', 'rank']


def train_models(
    features: pd.DataFrame,
    indices: List[pd.Timestamp],
    labels: pd.Series,
    epochs_tn: int = 6,
    depth: int = 6,
    iterations: int = 1000,
    random_seed: int = 42,
    loss_function: str = "YetiRank",
) -> Tuple[List[CatBoostRanker], List[TabNetRegressor]]:
    cb_models, tn_models = [], []

    for train_idx, test_idx in tqdm(indices):
        train_dataframe = features.loc(axis=0)[train_idx].swaplevel().sort_index()
        train_ts = train_dataframe.index.get_level_values("date").asi8
        train_target = labels.loc(axis=0)[train_idx]['target']
        train_rank = labels.loc(axis=0)[train_idx]['rank']

        eval_dataframe = features.loc(axis=0)[test_idx].swaplevel().sort_index()
        eval_ts = eval_dataframe.index.get_level_values("date").asi8
        eval_target = labels.loc(axis=0)[test_idx]['target']
        eval_rank = labels.loc(axis=0)[test_idx]['rank']


        tabnet_train = train_dataframe[tn_good_columns].fillna(-1).reset_index()
        tabnet_train['date'] = train_ts
        tabnet_eval = eval_dataframe[tn_good_columns].fillna(-1).reset_index()
        tabnet_eval['date'] = eval_ts


        tabnet = TabNetRegressor(
            cat_idxs = [0],
            cat_dims = [10000],
            seed = random_seed,
        )

        tabnet.fit(
            X_train=tabnet_train.values,
            y_train=train_target.values.reshape(-1, 1),
            eval_set=[(tabnet_eval.values, eval_target.values.reshape(-1, 1))],
            eval_name=['train'],
            eval_metric=['mae', 'rmse', 'mse'],
            max_epochs=epochs_tn,
            pin_memory=True,
        )
        tn_models.append(tabnet)

        train_dataframe['tabnet_preds'] = tabnet.predict(tabnet_train.values)
        eval_dataframe['tabnet_preds'] = tabnet.predict(tabnet_eval.values)

        categorical_cb = set(features.select_dtypes("object").columns)
        categorical_cb = list(categorical_cb.intersection(cb_good_columns))

        train_pool = Pool(
            train_dataframe[cb_good_columns + ['tabnet_preds']],
            label=train_rank,
            cat_features=categorical_cb,
            timestamp=train_ts,
            group_id=train_dataframe.index.get_level_values("securities_code").values,
        )

        eval_pool = Pool(
            eval_dataframe[cb_good_columns + ['tabnet_preds']],
            label=eval_rank,
            cat_features=categorical_cb,
            timestamp=eval_ts,
            group_id=eval_dataframe.index.get_level_values("securities_code").values,
        )

        cb_model = CatBoostRanker(
            depth=depth,
            iterations=iterations,
            random_seed=random_seed,
            task_type="GPU",
            loss_function=loss_function,
        )

        cb_model.fit(train_pool, eval_set=eval_pool, verbose=50)
        cb_models.append(cb_model)
    return cb_models, tn_models


def predict_models(
    cb_models: List[CatBoostRanker],
    tn_models: List[TabNetRegressor],
    dataset: pd.DataFrame,
):
    predictions = []
    for cb, tn in zip(tn_models, cb_models):
        tn_data = dataset[tn_good_columns].fillna(-1).reset_index()
        tn_data['date'] = tn_data['date'].asi8
        dataset['tabnet_preds'] = tn.predict(tn_data.values)
        
        categorical_cb = set(dataset.select_dtypes("object").columns)
        categorical_cb = list(categorical_cb.intersection(cb_good_columns))

        cb_pool = Pool(
            dataset[cb_good_columns + ['tabnet_preds']],
            cat_features=categorical_cb,
            timestamp=train_dataframe.index.get_level_values("date").asi8,
            group_id=eval_dataframe.index.get_level_values("securities_code").values,
        )
        predictions.append(cb_model.predict(cb_pool))
    return predictions
