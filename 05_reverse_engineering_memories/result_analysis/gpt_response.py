'''
This is used to get the GPT responses for different query generation methods (original, ft, icl),
and compute the BERTScore between GPT responses and original assistant messages.
'''

import pandas as pd 
import numpy as np 
from openai import OpenAI

from .separate_eval import safe_bertscore


try:
    client = OpenAI(api_key='YOUR_OPENAI_API_KEY')
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure you have set your OPENAI_API_KEY environment variable or replaced 'YOUR_API_KEY_HERE' in the script.")
    exit()

SYSTEM_PROMPT = """You are ChatGPT, a large language model trained by OpenAI. Engage warmly yet honestly with the user. Be direct; avoid ungrounded or sycophantic flattery. Maintain professionalism and grounded honesty that best represents OpenAI and its values."""


def get_llm_response(prompt_text, model="gpt-4o"):
    """A helper function to make a single API call and return the content."""
    print(f"\n--- Sending API Request (Model: {model}) ---")
    try:
        response = client.chat.completions.create(
            model=model,
            # response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An API error occurred: {e}")
        return None

def get_response(
    original_path: str,
    query_mode: str, #original or ft or icl
    first_query_path = '.../data/first_queries_all_en.csv'
):
    # get the user_id list from first_query_path
    first_query_df = pd.read_csv(first_query_path)
    user_id_list = first_query_df['user_id'].unique().tolist()
    original_df = pd.read_csv(original_path)
    if query_mode == 'original':
        query_path = '.../data/first_queries_all_en.csv'
        query_df = pd.read_csv(query_path)
        query_lable = 'User Message'
    elif query_mode == 'ft':
        query_df = pd.DataFrame()
        for user_id in user_id_list:
            query_path_ft = f'.../result/ft_csv/qwen2.5-32b-instruct_query-all3-full-conversation_train-sft_ratio-0.6_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-40/results_user-{user_id}_first-queries_qwen2.5-32b-instruct_query-all3-full-conversation_train-sft_ratio-0.6_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-40_rephrased_message.csv'
            query_df = pd.concat([query_df, pd.read_csv(query_path_ft)], axis=0)
        query_lable = 'response'
    elif query_mode == 'icl':
        query_df = pd.DataFrame()
        for user_id in user_id_list:
            query_path_icl = f'.../result/icl_csv/qwen2.5-32b-instruct/results_user-{user_id}_first-queries_qwen2.5-32b-instruct_rephrased_message.csv'
            query_df = pd.concat([query_df, pd.read_csv(query_path_icl)], axis=0)
        query_lable = 'response'
    else:
        raise NotImplementedError
    
    query_list = query_df[query_lable].tolist()
    # get the model label list from original_df, first to get the same samples in user_id, conversation_id, message_id
    # the model label is in the column 'Model'
    model_list = []
    result_df = pd.DataFrame()
    for index, row in query_df.iterrows():
        user_id = row['user_id']
        conversation_id = row['conversation_id']
        message_id = row['message_id']
        model_line = original_df[(original_df['user_id'] == user_id) & (original_df['conversation_id'] == conversation_id) & (original_df['message_id'] == message_id)]
        result_df = pd.concat([result_df, model_line], axis=0)

        model_message = model_line['Model'].values
        
        if len(model_message) == 0:
            model_list.append("")
        else:
            model_list.append(model_message[0])
    
    # now get the corresponding model message for each query message
    response_list = []
    for i, query in enumerate(query_list):
        print(f"Processing query {i+1}/{len(query_list)}")
        model = model_list[i]
        # remove [ and ] and " to avoid errors
        model = model.replace("[", "").replace("]", "").replace('"', '').strip()
        if model == "":
            raise ValueError(f"No corresponding model message found for query index {i}")
        prompt_text = query
        response = get_llm_response(prompt_text, model=model)
        if response is None:
            response = "API Error"
        response_list.append(response)
        print(f"Query: {query}\nResponse: {response}\n")

    # now compute the bertscore between response_list and original response
    original_response_list = original_df['Assistant Message'].tolist()
    # save each response's bertscore
    bertscore_list = []
    for i in range(len(response_list)):
        score = safe_bertscore(response_list[i], original_response_list[i])
        bertscore_list.append(score)

    # save all the results to a csv file, keep all the columns in original_df, and add a column for response and bertscore

    result_df[f'Query_{query_mode}'] = query_list
    result_df[f'Response_{query_mode}'] = response_list
    result_df[f'BertScore_{query_mode}'] = bertscore_list
    result_df.to_csv(f'.../result/response/gpt_response_{query_mode}.csv', index=False)
    print(f"Results saved to .../result/response/gpt_response_{query_mode}.csv")
    return result_df

def aggregate_results(df_path, query_mode):
    df = pd.read_csv(df_path)
    # compute the average bertscore
    # skip the rows with 'qeury_{query_mode}' is NaN or contains the information of  'No rephrased message'
    valid_rows = df[~df[f'Query_{query_mode}'].isna() & ~df[f'Query_{query_mode}'].str.contains('No rephrased message', na=False)]
    print(f"Number of valid rows for {query_mode}: {len(valid_rows)}")
    avg_bertscore_list = valid_rows[f'BertScore_{query_mode}'].tolist()
    precision, recall, f1 = [], [], []
    # "{'precision': 0.7840324640274048, 'recall': 0.8429630994796753, 'f1': 0.8124305009841919}"
    for score in avg_bertscore_list:
        try:
            score_dict = eval(score)
            precision.append(score_dict['precision'])
            recall.append(score_dict['recall'])
            f1.append(score_dict['f1'])
        except:
            print(f"Error parsing score: {score}")
    avg_precision = np.mean(precision) if precision else 0
    avg_recall = np.mean(recall) if recall else 0
    avg_f1 = np.mean(f1) if f1 else 0
    avg_bertscore = {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}

    print(f"Average BertScore for {query_mode}: {avg_bertscore}")
    # save the average bertscore to a text file
    with open(f'.../result/response/gpt_response_{query_mode}_avg.txt', 'w') as f:
        f.write(f'Average BertScore for {query_mode}: {avg_bertscore}\n Total valid samples: {len(valid_rows)}\n')
    return avg_bertscore

def aggregate_ft_icl(
    ft_path: str,
    icl_path: str,
    original_path: str
):
    ft_df = pd.read_csv(ft_path)
    icl_df = pd.read_csv(icl_path)
    original_df = pd.read_csv(original_path)
    # keep only the rows with the same user_id, conversation_id, message_id in both df
    merged_df = pd.merge(ft_df, icl_df, on=['user_id', 'conversation_id', 'message_id'], suffixes=('_ft', '_icl'))
    # drop the rows with 'qeury_ft' or 'qeury_icl' is NaN or contains the information of  'No rephrased message'
    merged_df = merged_df[~merged_df['Query_ft'].isna() & ~merged_df['Query_ft'].str.contains('No rephrased message', na=False)]
    merged_df = merged_df[~merged_df['Query_icl'].isna() & ~merged_df['Query_icl'].str.contains('No rephrased message', na=False)]
   
   # also merge with original_df to make sure the samples are in original_df
    merged_df = pd.merge(merged_df, original_df, on=['user_id', 'conversation_id', 'message_id'])

    print(f"Number of valid rows for ft and icl: {len(merged_df)}")
    avg_bertscore_list_ft = merged_df['BertScore_ft'].tolist()
    avg_bertscore_list_icl = merged_df['BertScore_icl'].tolist()
    avg_bertscore_list_original = merged_df['BertScore_original'].tolist() 
    precision_ft, recall_ft, f1_ft = [], [], []
    precision_icl, recall_icl, f1_icl = [], [], []
    precision_original, recall_original, f1_original = [], [], []

    for score in avg_bertscore_list_ft:
        try:
            score_dict = eval(score)
            precision_ft.append(score_dict['precision'])
            recall_ft.append(score_dict['recall'])
            f1_ft.append(score_dict['f1'])
        except:
            print(f"Error parsing score: {score}")
    for score in avg_bertscore_list_icl:
        try:
            score_dict = eval(score)
            precision_icl.append(score_dict['precision'])
            recall_icl.append(score_dict['recall'])
            f1_icl.append(score_dict['f1'])
        except:
            print(f"Error parsing score: {score}")

    for score in avg_bertscore_list_original:
        try:
            score_dict = eval(score)
            precision_original.append(score_dict['precision'])
            recall_original.append(score_dict['recall'])
            f1_original.append(score_dict['f1'])
        except:
            print(f"Error parsing score: {score}")

    avg_precision_ft = np.mean(precision_ft) if precision_ft else 0
    avg_recall_ft = np.mean(recall_ft) if recall_ft else 0
    avg_f1_ft = np.mean(f1_ft) if f1_ft else 0
    avg_bertscore_ft = {'precision': avg_precision_ft, 'recall': avg_recall_ft, 'f1': avg_f1_ft}
    avg_precision_icl = np.mean(precision_icl) if precision_icl else 0
    avg_recall_icl = np.mean(recall_icl) if recall_icl else 0
    avg_f1_icl = np.mean(f1_icl) if f1_icl else 0
    avg_bertscore_icl = {'precision': avg_precision_icl, 'recall': avg_recall_icl, 'f1': avg_f1_icl}
    avg_precision_original = np.mean(precision_original) if precision_original else 0
    avg_recall_original = np.mean(recall_original) if recall_original else 0
    avg_f1_original = np.mean(f1_original) if f1_original else 0
    avg_bertscore_original = {'precision': avg_precision_original, 'recall': avg_recall_original, 'f1': avg_f1_original}
    print(f"Average BertScore for ft: {avg_bertscore_ft}")
    print(f"Average BertScore for icl: {avg_bertscore_icl}")
    print(f"Average BertScore for original: {avg_bertscore_original}")
    # save the average bertscore to a text file
    with open(f'.../result/response/gpt_response_ft_icl_avg.txt', 'w') as f:
        f.write(f'Average BertScore for ft: {avg_bertscore_ft}\n')
        f.write(f'Average BertScore for icl: {avg_bertscore_icl}\n')
        f.write(f'Average BertScore for original: {avg_bertscore_original}\n')
        f.write(f'Total valid samples: {len(merged_df)}\n')

    return avg_bertscore_ft, avg_bertscore_icl

if __name__ == "__main__":
    # original_path = '.../result/response/chatgpt_processed_resp-conversations_cleaned.csv'
    # # get_response(original_path, query_mode='original')
    # aggregate_results(
    #     df_path = '.../result/response/gpt_response_original.csv',
    #     query_mode='original'
    #     )

    # # get_response(original_path, query_mode='ft')
    # aggregate_results(
    #     df_path = '.../result/response/gpt_response_ft.csv',
    #     query_mode='ft'
    #     )

    # # get_response(original_path, query_mode='icl')
    # aggregate_results(
    #     df_path = '.../result/response/gpt_response_icl.csv',
    #     query_mode='icl'
    #     )

    aggregate_ft_icl(
        ft_path = '.../result/response/gpt_response_ft.csv',
        icl_path = '.../result/response/gpt_response_icl.csv',
        original_path = '.../result/response/gpt_response_original.csv'
    )

