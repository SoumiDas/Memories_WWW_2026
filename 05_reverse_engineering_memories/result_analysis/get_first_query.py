'''
This file is used to extract the first query from each conversation, which has the ground 
truth memory and rephrased message. It is used for evaluating the model's performance on the first query.
'''
import os
import json
import pandas as pd
import random
from typing import List
from tqdm import tqdm

RAW_DATA_DIR = '.../results/raw'

def get_first_all_en(
    output_file: str,
    user_id_list: List[int]
) -> pd.DataFrame:
    """
    For each user_id in user_id_list, read
    RAW_DATA_DIR/updated_memories_user_{user_id}_merge.csv,
    select the first message per conversation_id (by earliest Create Time),
    keep all columns, filter out rows where:
      - Updated Memory == "No memory"
      - rephrased_message is NA or blank
    Merge all users' results into a single DataFrame and write to output_file.
    Returns the merged DataFrame.
    """
    frames = []

    required_cols = {"conversation_id", "Create Time", "Updated Memory", "rephrased_message"}

    for user_id in tqdm(user_id_list):
        user_file = os.path.join(RAW_DATA_DIR, f"updated_memories_user_{user_id}_merge.csv")
        if not os.path.exists(user_file):
            print(f"[warn] Missing file for user {user_id}: {user_file}")
            continue

        df = pd.read_csv(user_file)

        missing = required_cols - set(df.columns)
        if missing:
            print(f"[warn] Skipping user {user_id}; missing required columns: {sorted(missing)}")
            continue

        # Parse Create Time; rows with unparsable timestamps will drop out when sorted/grouped
        df["Create Time"] = pd.to_datetime(df["Create Time"], errors="coerce", utc=True)

        # Sort by time so groupby().head(1) keeps the earliest row per conversation
        df_sorted = df.sort_values("Create Time", kind="mergesort")

        # Keep the first row per conversation_id (earliest Create Time)
        firsts = df_sorted.groupby("conversation_id", as_index=False, sort=False).head(1)

        # Apply filters
        mask = (
            firsts["Updated Memory"].fillna("").ne("No memory") &
            firsts["rephrased_message"].fillna("").str.strip().ne("")
        )
        firsts = firsts.loc[mask].copy()

        # Preserve user_id (add if not present)
        if "user_id" not in firsts.columns:
            firsts["user_id"] = user_id

        frames.append(firsts)

    if not frames:
        print("[info] No data collected; writing empty CSV.")
        pd.DataFrame().to_csv(output_file, index=False)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Optional: stable sort by user_id then Create Time for readability (keeps all cols)
    sort_cols = [c for c in ["user_id", "Create Time"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    out.to_csv(output_file, index=False)
    print(f"[info] Wrote {len(out)} rows to {output_file}")
    return out
        

if __name__ == "__main__":
    output_file = '.../results/first_queries_all_en.csv'
    # english user's ids
    user_ids = [12, 13, 16, 27, 28, 29, 31, 32, 34, 35, 38, 39, 42, 45, 46, 51, 54, 55, 56, 59, 63, 64, 65, 68, 70, 71, 72, 76, 77, 78]

    get_first_all_en(output_file=output_file,
                     user_id_list=user_ids)