from itertools import combinations
import pandas as pd
import numpy as np
from itertools import chain


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
