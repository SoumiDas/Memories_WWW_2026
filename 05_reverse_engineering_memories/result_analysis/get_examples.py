'''
This file is used to extract the first query from each conversation, which has the ground 
truth memory and rephrased message. It is used for evaluating the model's performance on the first query.
'''
import pandas as pd
from typing import List

RAW_DATA_DIR = '.../dummy_data/first_queries_all_en.csv'


def get_examples_en(
    output_file: str,
    user_id_list: List[int]
) -> pd.DataFrame:
    '''
    Extract examples which has both ground truth memory, rephrased message,
    and also the extracted memory, rephrased message in the 3 model's ICL and FT results.
    '''
    test_data = pd.read_csv(RAW_DATA_DIR)
    user_id_list = set(test_data['user_id'].apply(lambda x: int(x.split('_')[1]))).intersection(set(user_id_list))
    print(f"Total {len(test_data)} examples in the raw test data.")
    # print(f"Head of the raw test data:\n{test_data.head()}")
    # get test_data which has ground truth memory (!= 'No memory') and rephrased message
    test_data = test_data[(test_data['Updated Memory'] != 'No memory')]
    test_data = test_data[test_data['rephrased_message'].notnull()]
    user_id_list_str = [f'user_{user_id}' for user_id in user_id_list]
    test_data = test_data[test_data['user_id'].isin(user_id_list_str)]
    print(f"Total {len(test_data)} examples with ground truth memory and rephrased message.")
    example_df = test_data.copy()
    # add columns for the 6 model results
    example_df['qwen_ft_memory'] = ''
    example_df['qwen_ft_rephrased_message'] = ''
    example_df['qwen_icl_memory'] = ''
    example_df['qwen_icl_rephrased_message'] = ''

    for item in ['memory', 'rephrased_message']:
        for user_id in user_id_list:
            qwen_ft_result = pd.read_csv(f'.../memory/result/csv/qwen2.5-32b-instruct_query-all3-full-conversation_train-sft_ratio-0.6_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-40/results_user-user_{user_id}_first-queries_qwen2.5-32b-instruct_query-all3-full-conversation_train-sft_ratio-0.6_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-40_{item}.csv')
            qwen_icl_result = pd.read_csv(f'.../memory/result/open_csv/qwen2.5-32b-instruct/results_user-user_{user_id}_first-queries_qwen2.5-32b-instruct_{item}.csv')
            qwen_ft_result = qwen_ft_result[(qwen_ft_result['response'] != 'No response') & (qwen_ft_result['response'] != 'error') & (qwen_ft_result['response'] != 'no memory') & (qwen_ft_result['response'] != 'No rephrased message.') & (qwen_ft_result['response'].notnull())]
            qwen_ft_result = qwen_ft_result[qwen_ft_result['conversation_id'].isin(test_data['conversation_id'])]
            qwen_icl_result = qwen_icl_result[(qwen_icl_result['response'] != 'No response') & (qwen_icl_result['response'] != 'error') & (qwen_icl_result['response'] != 'no memory') & (qwen_icl_result['response'] != 'No rephrased message.') & (qwen_icl_result['response'].notnull())]
            qwen_icl_result = qwen_icl_result[qwen_icl_result['conversation_id'].isin(test_data['conversation_id'])]
            
            filtered_test_data = test_data[test_data['conversation_id'].isin(qwen_ft_result['conversation_id']) & test_data['conversation_id'].isin(qwen_icl_result['conversation_id'])]
            filtered_test_data = filtered_test_data[filtered_test_data['message_id'].isin(qwen_ft_result['message_id']) & filtered_test_data['message_id'].isin(qwen_icl_result['message_id'])]
            # print(f"User {user_id}: {len(filtered_test_data)} examples with valid responses from all 6 model results.")
            # for each user combine the firtered test data to corresponding example_df rows, based on conversation_id and message_id
            for idx, row in filtered_test_data.iterrows():
                conv_id = row['conversation_id']
                msg_id = row['message_id']
                example_idx = example_df[(example_df['conversation_id'] == conv_id) & (example_df['message_id'] == msg_id)].index
                if len(example_idx) == 1:
                    example_idx = example_idx[0]
                    if item == 'memory':
                        example_df.at[example_idx, 'qwen_ft_memory'] = qwen_ft_result[qwen_ft_result['conversation_id'] == conv_id][qwen_ft_result['message_id'] == msg_id]['response'].values[0]
                        example_df.at[example_idx, 'qwen_icl_memory'] = qwen_icl_result[qwen_icl_result['conversation_id'] == conv_id][qwen_icl_result['message_id'] == msg_id]['response'].values[0]
                    elif item == 'rephrased_message':
                        example_df.at[example_idx, 'qwen_ft_rephrased_message'] = qwen_ft_result[qwen_ft_result['conversation_id'] == conv_id][qwen_ft_result['message_id'] == msg_id]['response'].values[0]
                        example_df.at[example_idx, 'qwen_icl_rephrased_message'] = qwen_icl_result[qwen_icl_result['conversation_id'] == conv_id][qwen_icl_result['message_id'] == msg_id]['response'].values[0]
                else:
                    print(f"Warning: User {user_id}: Cannot find unique example for conversation_id {conv_id} and message_id {msg_id} in example_df.")

    # now filter out all the rows which has any of the 6 model results missing or contains the 'No rephrased message.' or 'no memory'
    filtered_data =pd.DataFrame()
    for idx, row in example_df.iterrows():
        if all([
            row['qwen_ft_memory'] not in ['', 'No rephrased message.', 'no memory'],
            row['qwen_icl_memory'] not in ['', 'No rephrased message.', 'no memory'],
        ]):
            filtered_data = pd.concat([filtered_data, pd.DataFrame([row])], ignore_index=True)
    example_df = filtered_data
    print(f"Total {len(example_df)} examples after combining all users.")

        # save output
    example_df.to_csv(output_file, index=False)
    print(f"Saved {len(example_df)} examples to {output_file}")
    return example_df


if __name__ == "__main__":
    output_file = '.../memory/data/examples_en.csv'
    user_ids = [0,1,4,8,9, 12, 13, 16, 27, 28, 29, 31, 32, 34, 35, 38, 39, 42, 45, 46, 51, 54, 55, 56, 59, 63, 64, 65, 68, 70, 71, 72, 76, 77, 78]

    get_examples_en(output_file, user_ids)
