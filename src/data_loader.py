import random
import pandas as pd


def load_data(path='../data/DSL-StrongPasswordData.csv', include_rep=True,
              include_h=True, include_dd=True, include_ud=True):
    df = pd.read_csv(path)
    columns_to_drop = []
    if include_rep:
        new_column = (df['sessionIndex'] - 1) * 50 + df['rep']
        df.insert(1, 'rep2', new_column)
    columns_to_drop.extend(['sessionIndex', 'rep'])
    df["subject"] = df["subject"].str.replace("^s", "", regex=True).astype(int)
    if not include_h:
        columns_to_drop.extend(df.filter(like="H.").columns)
    if not include_dd:
        columns_to_drop.extend(df.filter(like="DD.").columns)
    if not include_ud:
        columns_to_drop.extend(df.filter(like="UD.").columns)
    df = df.drop(columns=columns_to_drop)
    return df

def split_data(df, sample: list[tuple[int, int]] = None, test_bound = 400, randomized=False):
    """
    Splits the DataFrame into training and testing sets based on per-subject sample rules.

    Args:
        df: Input DataFrame containing a 'subject' column.
        sample: List of tuples (num_subjects, train_rows_per_subject) defining the split strategy.
        test_bound: Maximum number of rows per subject to allocate for the test set.
        randomized: If True, uses non-deterministic seeding.

    Returns:
        A tuple (df_train, df_test) containing the shuffled split DataFrames.
    """
    if sample is None:
        sample = [(51, 360)]
    assert sum(s[0] for s in sample) == 51
    for x, y in sample:
        assert x >= 1 and 0 <= y <= 400
    if randomized:
        local_random = random.Random()
    else:
        local_random = random.Random(42)
    subjects = df['subject'].unique().tolist()
    local_random.shuffle(subjects)
    i = 0
    training_indices = []
    test_indices = []
    for x, y in sample:
        for _ in range(x):
            subject = subjects[i]
            ds = df[df['subject'] == subject]
            indices = local_random.sample(list(ds.index), k=y)
            training_indices.extend(indices)
            test_indices.extend(local_random.sample(list(set(ds.index) - set(indices)), k = min(400 - y, test_bound)))
            i += 1
    df_train = df.loc[training_indices]
    df_test = df.loc[test_indices]
    if randomized:
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_test = df_test.sample(frac=1).reset_index(drop=True)
    else:
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_train, df_test