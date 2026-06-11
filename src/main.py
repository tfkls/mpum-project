import hashlib
import itertools
import pickle
from pathlib import Path

import matplotlib

# Do not use GUI, just write images
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import MODELS

ROOT = Path(__file__).resolve().parents[1]

BIG_SUBJECTS = 35
SWEEP_SIZES = [10, 20, 40, 60, 80, 120, 160, 260, 360]
VALID_SIZES = 40
RANDOM_SEEDS = [42, 24601, 8675309]
GRADE_SAMPLES = 40
TRAIN_VALID_RATIO = 0.2

assert SWEEP_SIZES[-1] + GRADE_SAMPLES == 400


def cache_path(label, *key):
    """File in cache/ named by the label and a hash of everything in `key`."""
    digest = hashlib.md5(str(key).encode()).hexdigest()[:10]
    return ROOT / "cache" / f"{label}_{digest}.pkl"


def load_data():
    df = pd.read_csv(ROOT / "data" / "DSL-StrongPasswordData.csv")
    df["subject"] = df["subject"].str.removeprefix("s").astype(int)
    return df[["subject"] + [c for c in df.columns if c.startswith(("H.", "UD."))]]


def split_data(df, small_reps, seed):
    """Returns: training (later split into valid) set, grade/valid set and which subjects are the "small sample" group"""
    rng = np.random.default_rng(seed)
    subjects = rng.permutation(df["subject"].unique())
    parts = [
        (
            rng.permutation(df.index[df["subject"] == s]),
            None if i < BIG_SUBJECTS else small_reps,
        )
        for i, s in enumerate(subjects)
    ]
    train = np.concatenate([rows[GRADE_SAMPLES:][:reps] for rows, reps in parts])
    test = np.concatenate([rows[:GRADE_SAMPLES] for rows, _ in parts])
    return df.loc[train], df.loc[test], subjects[BIG_SUBJECTS:]


def evaluate(model, df, small_subjects):
    """Returns a dict of: mean error over all subjects, mean error over big sample subjects, mean error over small sample subjects and their ratio"""
    y = df["subject"].to_numpy()
    y_pred = np.array(
        [model.predict(row) for row in df.drop(columns="subject").to_numpy()]
    )
    wrong = y != y_pred
    small = np.isin(y, small_subjects)
    big_err, small_err = wrong[~small].mean(), wrong[small].mean()
    return {
        "all": wrong.mean(),
        "big": big_err,
        "small": small_err,
        "ratio": small_err / big_err if big_err else float("nan"),
    }


def grid_search(cls, df_train, small_subjects):
    """Returns: records (for use by plotting) and the best models found for each error criterion"""
    path = cache_path(
        f"search_{cls.__name__}",
        cls.param_grid,
        len(df_train),
        sorted(small_subjects),
        TRAIN_VALID_RATIO,
    )
    if path.exists():
        print(f"\tCache hit ({path.name})")
        records, best = pickle.loads(path.read_bytes())
    else:
        print(f"\tCache miss ({path.name})")
        valid = df_train.groupby("subject").sample(
            frac=TRAIN_VALID_RATIO, random_state=0
        )
        train = df_train.drop(valid.index)
        records = []
        for values in itertools.product(*cls.param_grid.values()):
            params = dict(zip(cls.param_grid, values))
            errors = evaluate(cls(**params).train(train), valid, small_subjects)
            records.append((params, errors))
            print(
                f"\t{params}: {'{'} all: {errors['all']:.4f}, big: {errors['big']:.4f}, small: {errors['small']:.4f}, ratio: {errors['ratio']:.2f} {'}'}"
            )
        best = {
            obj: min(records, key=lambda r: r[1][obj])[0]
            for obj in ["all", "big", "small"]
        }
        path.write_bytes(pickle.dumps((records, best)))
    for obj, params in best.items():
        print(f"\tbest for {obj}: {params}")
    return records, best


def size_sweep(cls, params, data):
    """Returns curve to use by plotting (error on selecting soo many small samples)"""
    path = cache_path(
        f"sweep_{cls.__name__}",
        params,
        SWEEP_SIZES,
        RANDOM_SEEDS,
        BIG_SUBJECTS,
        GRADE_SAMPLES,
    )
    if path.exists():
        print(f"\tCache hit ({path.name})")
        return pickle.loads(path.read_bytes())
    print(f"\tCache miss ({path.name})")
    curve = {"all": [], "big": [], "small": []}
    for size in SWEEP_SIZES:
        print(f"\tSweeping for {size} small samples")
        runs = [
            evaluate(cls(**params).train(train), test, small)
            for train, test, small in (
                split_data(data, size, seed) for seed in RANDOM_SEEDS
            )
        ]
        for group in curve:
            curve[group].append(np.mean([r[group] for r in runs]))
    path.write_bytes(pickle.dumps(curve))
    return curve


def plot_grid(name, records):
    y = np.arange(len(records))
    plt.figure(figsize=(9, 0.4 * len(records) + 2))
    plt.barh(y - 0.25, [val["all"] for _, val in records], height=0.25, label="all")
    plt.barh(y, [val["big"] for _, val in records], height=0.25, label="big")
    plt.barh(y + 0.25, [val["small"] for _, val in records], height=0.25, label="small")
    plt.yticks(y, [str(key) for key, _ in records], fontsize=7)
    plt.xlabel("Classification error")
    plt.title(f"Hyperparameter search for {name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "results" / f"{name}_search.png", dpi=150)
    plt.close()


def plot_curves(lines, title, filename):
    plt.figure()
    for label, values in lines.items():
        plt.plot(SWEEP_SIZES, values, "o-", label=label)
    plt.xscale("log")
    plt.xticks(SWEEP_SIZES, [str(x) for x in SWEEP_SIZES])
    plt.xlabel("Small sample set training size")
    plt.ylabel("Classification error")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "results" / filename, dpi=150)
    plt.close()


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "cache").mkdir(exist_ok=True)
    data = load_data()
    summary, comparison = [], {"all": {}, "small": {}}

    for cls in MODELS:
        name = cls.__name__
        print(f"Hyperparameter search for {name}...")
        df_train, df_grade, small = split_data(data, VALID_SIZES, seed=3238462)
        records, best = grid_search(cls, df_train, small)
        plot_grid(name, records)

        print(f"Sweep for {name}...")
        sweeps = {}
        for objective in ["all", "small"]:
            result = evaluate(cls(**best[objective]).train(df_train), df_grade, small)
            summary.append(
                {
                    "model": name,
                    "for": objective,
                    "params": str(best[objective]),
                    **{k: round(v, 4) for k, v in result.items()},
                }
            )
            curve = size_sweep(cls, best[objective], data)
            for group in curve:
                sweeps[f"{group} for {objective}"] = curve[group]
            comparison[objective][name] = curve["small"]
        plot_curves(sweeps, f"{name} error by small sample count", f"{name}_sweep.png")

    plot_curves(comparison["all"], "Model comparison (for all)", "comparison_all.png")
    plot_curves(
        comparison["small"], "Model comparison (for small)", "comparison_small.png"
    )
    summary = pd.DataFrame(summary)
    print()
    print(summary.to_string(index=False))
    summary.to_csv(ROOT / "results" / "summary.csv", index=False)
    print("Saved results")


if __name__ == "__main__":
    main()
