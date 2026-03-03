"""
KuaiRec data preprocessing pipeline.

This script covers the full preprocessing workflow from raw KuaiRec files to
the .npy feature arrays consumed by the GR model trainer.

Pipeline overview:
  Step 1 — Load pre-normalised interaction matrices (big / small).
  Step 2 — Parse item category features (feat0–feat3).
  Step 3 — Compute per-item mean video duration.
  Step 4 — Label-encode user features (ordinal + one-hot).
  Step 5 — Join all features into a single wide table per split.
  Step 6 — Serialise each wide table to a .npy feature array.

Expected directory layout under data/:
  data_raw/
    big_matrix_processed.csv    # pre-normalised big-matrix interactions
    small_matrix_processed.csv  # pre-normalised small-matrix interactions
    item_categories.csv         # item category tags
    user_features.csv           # raw user features

Outputs written to data_processed/:
  train_kauiRec.npy             # training split
  test_kauiRec.npy              # test / validation split

Each row in the .npy array is packed as (52 columns total):
  [  0: 19]  user features  — [user_id, onehot_feat0 … onehot_feat17]
  [ 19: 24]  item features  — [item_id, feat0, feat1, feat2, feat3]
  [ 24: 43]  user mask      — 1.0 for valid fields, 0.0 for padding
  [ 43: 48]  item mask      — 1.0 for valid fields, 0.0 for padding
  [     48]  watch_ratio_normed   (used as the watch-time label)
  [     49]  item_duration_normed (video duration, used for bucketing)
  [     50]  usr_len        — always 19
  [     51]  item_len       — number of valid item feature entries (1–5)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
ROOTPATH    = os.path.dirname(__file__)
DATAPATH    = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")
os.makedirs(PRODATAPATH, exist_ok=True)

BIG_PROCESSED   = os.path.join(DATAPATH, "big_matrix_processed.csv")
SMALL_PROCESSED = os.path.join(DATAPATH, "small_matrix_processed.csv")

# User feature column lists
ORDINAL_COLS = [
    "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
]
ONEHOT_COLS = [f"onehot_feat{x}" for x in range(18)]


# ---------------------------------------------------------------------------
# Step 1: Load pre-processed interaction matrices
# ---------------------------------------------------------------------------
def load_processed_matrix():
    """Load big_matrix_processed.csv and small_matrix_processed.csv.

    Both files are expected to contain normalised watch-ratio and duration
    columns produced by an upstream normalisation step.

    Returns:
        df_big   (DataFrame): Interactions from the big matrix (training).
        df_small (DataFrame): Interactions from the small matrix (test).
    """
    assert os.path.exists(BIG_PROCESSED) and os.path.exists(SMALL_PROCESSED), (
        "Processed matrix files not found under data_raw/. "
        "Please run the matrix normalisation step first."
    )
    print("Loading processed matrix files...")
    cols = ["user_id", "item_id", "timestamp", "watch_ratio_normed", "duration_normed"]
    df_big   = pd.read_csv(BIG_PROCESSED,   usecols=cols)
    df_small = pd.read_csv(SMALL_PROCESSED, usecols=cols)
    return df_big, df_small


# ---------------------------------------------------------------------------
# Step 2: Load item category features (feat0–feat3)
# ---------------------------------------------------------------------------
def load_item_category_feat():
    """Parse item_categories.csv into per-item integer category columns.

    Each item has up to 4 category tags stored as a Python-literal list string
    in the 'feat' column.  Missing tags are filled with -1 before a +1 shift
    so that the final encoding reserves 0 for unknown / absent categories.

    Returns:
        df_feat (DataFrame): Index = item_id, columns feat0/feat1/feat2/feat3.
    """
    filepath = os.path.join(DATAPATH, "item_categories.csv")
    df_raw = pd.read_csv(filepath, header=0)
    df_raw["feat"] = df_raw["feat"].map(eval)   # string literal → Python list

    df_feat = pd.DataFrame(df_raw["feat"].tolist(),
                           columns=["feat0", "feat1", "feat2", "feat3"])
    df_feat.index.name = "item_id"
    df_feat[df_feat.isna()] = -1
    df_feat = (df_feat + 1).astype(int)         # shift: NaN/unknown (-1) → 0
    return df_feat


# ---------------------------------------------------------------------------
# Step 3: Load per-item mean video duration
# ---------------------------------------------------------------------------
def load_item_duration():
    """Compute (or load from cache) the per-item mean normalised duration.

    If video_duration_normed.csv already exists under data_raw/ it is loaded
    directly; otherwise it is computed from both matrix files and cached.

    Returns:
        video_mean_duration (Series): Index = item_id, values = duration_normed.
    """
    duration_path = os.path.join(DATAPATH, "video_duration_normed.csv")
    if os.path.exists(duration_path):
        video_mean_duration = pd.read_csv(duration_path, header=0)["duration_normed"]
    else:
        cols = ["item_id", "duration_normed"]
        df_big_dur   = pd.read_csv(BIG_PROCESSED,   usecols=cols)
        df_small_dur = pd.read_csv(SMALL_PROCESSED, usecols=cols)
        combined = pd.concat([df_big_dur, df_small_dur], axis=0)
        video_mean_duration = combined.groupby("item_id")["duration_normed"].mean()
        video_mean_duration.to_csv(duration_path, index=False)
    video_mean_duration.index.name = "item_id"
    return video_mean_duration


# ---------------------------------------------------------------------------
# Step 4: Load and label-encode user features
# ---------------------------------------------------------------------------
def _label_encode_col(series: pd.Series) -> pd.Series:
    """Apply LabelEncoder to a single categorical column.

    'UNKNOWN' strings are mapped to chr(0) so they sort to position 0.
    If the resulting encoding already contains a zero-indexed unknown
    placeholder (chr(0) or -124), no +1 offset is applied; otherwise all
    labels are shifted by +1 to reserve index 0 for padding / unknown.

    Args:
        series: Raw categorical column.

    Returns:
        Integer-encoded Series with the same index.
    """
    series = series.map(lambda x: chr(0) if x == "UNKNOWN" else x)
    lbe = LabelEncoder()
    encoded = lbe.fit_transform(series)
    unknown_in_classes = (chr(0) in lbe.classes_.tolist()
                          or -124 in lbe.classes_.tolist())
    if not unknown_in_classes:
        encoded = encoded + 1
    return pd.Series(encoded, index=series.index)


def load_user_feat(user_filter=None):
    """Load user_features.csv and label-encode all feature columns.

    Args:
        user_filter (array-like, optional): Subset of user IDs to retain.
            When provided, only rows whose index is in this set are returned.

    Returns:
        df_user (DataFrame): Index = user_id, columns = encoded features.
    """
    filepath = os.path.join(DATAPATH, "user_features.csv")
    df_user = pd.read_csv(filepath,
                          usecols=["user_id"] + ORDINAL_COLS + ONEHOT_COLS)

    for col in ORDINAL_COLS:
        df_user[col] = _label_encode_col(df_user[col])

    for col in ONEHOT_COLS:
        df_user[col] = df_user[col].fillna(-124)
        df_user[col] = _label_encode_col(df_user[col])

    df_user = df_user.set_index("user_id")

    if user_filter is not None:
        df_user = df_user.loc[user_filter]
    return df_user


# ---------------------------------------------------------------------------
# Step 5: Join all features into a wide table
# ---------------------------------------------------------------------------
def build_dataset(df_interact: pd.DataFrame,
                  df_feat: pd.DataFrame,
                  df_user: pd.DataFrame,
                  df_item: pd.DataFrame) -> pd.DataFrame:
    """Merge interaction data with item and user features into a wide table.

    Args:
        df_interact: User–item interaction records.
        df_feat:     Item category features (feat0–feat3).
        df_user:     Encoded user features.
        df_item:     Item features including mean duration.

    Returns:
        Wide-table DataFrame with all features joined on user_id / item_id.
    """
    # Join item category features
    df = df_interact.join(df_feat[["feat0", "feat1", "feat2", "feat3"]],
                          on="item_id", how="left")

    # Join per-item mean video duration
    df = df.join(
        df_item[["duration_normed"]].rename(
            columns={"duration_normed": "item_duration_normed"}),
        on="item_id", how="left")

    # Join user features
    df = df.join(df_user, on="user_id", how="left")

    return df


# ---------------------------------------------------------------------------
# Step 6: Serialise wide-table DataFrames to .npy feature arrays
# ---------------------------------------------------------------------------
def _build_npy(df: pd.DataFrame) -> np.ndarray:
    """Convert a wide-table DataFrame to a packed NumPy feature array.

    See module docstring for the exact column layout of the output array.

    Args:
        df: Wide-table DataFrame produced by build_dataset().

    Returns:
        Float64 NumPy array of shape (N, 52).
    """
    data_sets = []
    for index, data in df.iterrows():
        if index % 10000 == 0:
            print(f"  Processing row {index} ...")

        # --- User features: [user_id, onehot_feat0 … onehot_feat17] (19 dims) ---
        usr_list = data[["user_id"] + ONEHOT_COLS].values.tolist()
        usr_list = [0 if i == 12345 else int(i) + 1 for i in usr_list]
        usr_list[0] = usr_list[0] - 1   # restore user_id to its original encoding
        usr_len = len(usr_list)
        assert usr_len == 19

        # --- Item features: [item_id, feat0, feat1, feat2, feat3] (5 dims) ---
        item_list = [int(data["item_id"]),
                     int(data["feat0"]), int(data["feat1"]),
                     int(data["feat2"]), int(data["feat3"])]
        item_list = item_list[:5]
        # item_len counts item_id (always valid) plus any non-zero category tags
        item_len  = 1 + sum(1 for v in item_list[1:] if v != 0)
        item_list = item_list + [0] * (5 - len(item_list))

        # --- Masks ---
        usr_mask  = [0.0 if v == 0 and idx != 0 else 1.0
                     for idx, v in enumerate(usr_list)]
        item_mask = [1.0] * item_len + [0.0] * (5 - item_len)

        # --- Scalar labels / auxiliary fields ---
        watch_ratio = float(data["watch_ratio_normed"])
        duration    = float(data["item_duration_normed"])

        row = (usr_list
               + item_list
               + usr_mask
               + item_mask
               + [watch_ratio, duration, float(usr_len), float(item_len)])
        data_sets.append(row)

    return np.array(data_sets, dtype=float)


def save_npy_arrays(df_train: pd.DataFrame,
                    df_val: pd.DataFrame,
                    out_dir: str = PRODATAPATH) -> None:
    """Serialise training and test DataFrames to .npy feature arrays.

    Writes:
        <out_dir>/train_kauiRec.npy
        <out_dir>/test_kauiRec.npy

    Args:
        df_train: Training-split wide table.
        df_val:   Test/validation-split wide table.
        out_dir:  Output directory (defaults to data_processed/).
    """
    print("\nConverting training split to NumPy array...")
    train_arr  = _build_npy(df_train)
    train_path = os.path.join(out_dir, "train_kauiRec.npy")
    np.save(train_path, train_arr)
    print(f"  Saved  → {train_path}   shape={train_arr.shape}")
    print(f"  Sample row: {train_arr[0]}")

    print("\nConverting test split to NumPy array...")
    test_arr  = _build_npy(df_val)
    test_path = os.path.join(out_dir, "test_kauiRec.npy")
    np.save(test_path, test_arr)
    print(f"  Saved  → {test_path}    shape={test_arr.shape}")
    print(f"  Sample row: {test_arr[0]}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    """Run the full preprocessing pipeline end-to-end."""

    # Step 1 — Load interaction matrices
    df_big, df_small = load_processed_matrix()
    print(f"big_matrix:   {len(df_big):,} interactions  "
          f"| #users={df_big['user_id'].nunique()}  "
          f"#items={df_big['item_id'].nunique()}")
    print(f"small_matrix: {len(df_small):,} interactions  "
          f"| #users={df_small['user_id'].nunique()}  "
          f"#items={df_small['item_id'].nunique()}")

    # Step 2 — Item category features
    print("\nLoading item category features...")
    df_feat = load_item_category_feat()

    # Step 3 — Item duration
    print("Loading item duration features...")
    video_mean_duration = load_item_duration()
    df_item = df_feat.join(video_mean_duration, on="item_id", how="left")

    # Step 4 — User features
    print("Loading user features (training / all users)...")
    df_user_train = load_user_feat()

    val_users = df_small["user_id"].unique()
    print("Loading user features (test / small-matrix users)...")
    df_user_val = load_user_feat(user_filter=val_users)

    # Step 5 — Build wide tables
    print("\nBuilding wide table for training split...")
    df_train = build_dataset(df_big, df_feat, df_user_train, df_item)
    print(f"  df_train shape: {df_train.shape}")

    print("Building wide table for test split...")
    df_val = build_dataset(df_small, df_feat, df_user_val, df_item)
    print(f"  df_val shape:   {df_val.shape}")

    # Step 6 — Serialise to .npy
    save_npy_arrays(df_train, df_val)

    return df_train, df_val


if __name__ == "__main__":
    main()