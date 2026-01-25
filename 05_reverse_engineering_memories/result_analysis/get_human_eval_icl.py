''''
This file is used to sample 100 random entries for the human evaluation.
'''

import pandas as pd
import os

model_name = 'gemma-3-27b-it'
BASE_DIR = f'.../result/icl_csv/{model_name}'
# for FT results:
# BASE_DIR = f'.../result/ft_csv/{model_name}'
memory_all, rephrased_all = [], []

user_id_list = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 59, 60, 61, 63, 64, 65, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78
]

for user_id in user_id_list:
    MEMORY_CSV_NAME = f"results_user-user_{user_id}_all-user-full_{model_name}_memory.csv"
    REPHRASED_CSV_NAME = f"results_user-user_{user_id}_all-user-full_{model_name}_rephrased_message.csv"
    memory_csv_path = os.path.join(BASE_DIR, MEMORY_CSV_NAME)
    rephrased_csv_path = os.path.join(BASE_DIR, REPHRASED_CSV_NAME)

    # first contact all csv files
    memory_df = pd.read_csv(memory_csv_path)
    rephrased_df = pd.read_csv(rephrased_csv_path)
    memory_all.append(memory_df)
    rephrased_all.append(rephrased_df)

memory_all_df = pd.concat(memory_all, ignore_index=True)
rephrased_all_df = pd.concat(rephrased_all, ignore_index=True)

# sample 100 random entries, only if 'response' is not empty and not 'No Memory' or 'No rephrased message.'
# only keep those needed columns
# sample the same random seed for both dataframes
useful_memory_columns = [
    "user_id",
    "conversation_id",
    "message_id",
    "query",
    "response", # Response of the imitated memory from the targeted model
    "ground_truth",
]

userful_rephrased_columns = [
    "user_id",
    "conversation_id",
    "message_id",
    "query",
    "response", # Response of the imitated rephrased query from the target model
    "ground_truth"
]

memory_all_df = memory_all_df[useful_memory_columns]
rephrased_all_df = rephrased_all_df[userful_rephrased_columns]

memory_filtered_df = memory_all_df[
    (memory_all_df["response"].notnull()) &
    (memory_all_df["response"] != "No Memory.") &
    (memory_all_df["ground_truth"] != "No memory") &
    (memory_all_df["response"] != "")
]   

rephrased_filtered_df = rephrased_all_df[
    rephrased_all_df["response"].notnull() &
    (rephrased_all_df["response"].str.strip() != "") &
    (~rephrased_all_df["response"].str.contains(
        "no rephrased message",
        case=False,
        na=False
    ))
]

# now for each data in the memory_filtered_df and rephrased_filtered_df,
# get and add the context based on the user_id, conversation_id, message_id
reference_data_df = pd.read_csv(".../dummy_data/test.csv")
join_cols = ["user_id", "conversation_id", "message_id"]

memory_filtered_df = memory_filtered_df.merge(
    reference_data_df[join_cols + ["context"]],
    on=join_cols,
    how="left"
)

rephrased_filtered_df = rephrased_filtered_df.merge(
    reference_data_df[join_cols + ["context"]],
    on=join_cols,
    how="left"
)

# save to csv for all entries
SAVE_DIR = ".../result/"
FILTERED_MEMORY_CSV_NAME = f"all_filtered_memory_{model_name}.csv"
FILTERED_REPHRASED_CSV_NAME = f"all_filtered_rephrased_{model_name}.csv"

memory_filtered_df.to_csv(os.path.join(SAVE_DIR, FILTERED_MEMORY_CSV_NAME), index=False)
rephrased_filtered_df.to_csv(os.path.join(SAVE_DIR, FILTERED_REPHRASED_CSV_NAME), index=False)

# 100 samples for human annotation

sampled_memory_df = memory_filtered_df.sample(n=100, random_state=42).reset_index(drop=True)
sampled_rephrased_df = rephrased_filtered_df.sample(n=100, random_state=42).reset_index(drop=True)

# save to csv
SAVE_DIR = ".../result/human_eval/"
SAMPLED_MEMORY_CSV_NAME = f"human_eval_memory_{model_name}.csv"
SAMPLED_REPHRASED_CSV_NAME = f"human_eval_rephrased_{model_name}.csv"

sampled_memory_df.to_csv(os.path.join(SAVE_DIR, SAMPLED_MEMORY_CSV_NAME), index=False)
sampled_rephrased_df.to_csv(os.path.join(SAVE_DIR, SAMPLED_REPHRASED_CSV_NAME), index=False)
