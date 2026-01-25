'''
Construct the prefix for all the LKEs
'''

import pandas as pd
import yaml
import random
from transformers import AutoTokenizer
from ..util_public.get_model_path import ModelPath

class ConstructPrompt:
    def __init__(self, config_path):
        """
        Initialize the ConstructPrompt class by loading the configuration from a YAML file.
        """
        self.config = self.read_config(config_path)
        self.model_name = self.config.get('model_name')
        self.model_path = ModelPath(config_path).get_model_path()
        self.test_dataset_path = self.config.get('test_dataset_path')
        self.test_dataset_name = self.config.get('test_dataset_name')
        self.random_seed = self.config.get('random_seed')
        self.open_content = self.config.get('open_content')
        self.chat_template = self.config.get('chat_template')
    
    def read_config(self, path):
        """
        Read the YAML configuration file and return the config dictionary.
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_tokenizer(self):
        """
        Load the tokenizer from the tokenizer path.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    
    def apply_chat_templates(self, chat_template_path, user_prompt, system_prompt):
        """
        Apply chat templates for the model.
        """
        tokenizer = self.load_tokenizer()
        with open(chat_template_path, 'r') as f:
            tokenizer.chat_template = f.read()

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "user", "content": user_prompt}]
    
        full_prompt = tokenizer.apply_chat_template(messages, 
                                                    tokenize=False,
                                                    add_generation_prompt=True, 
                                                    )
        # print(f'Full prompt: {full_prompt}')
        return full_prompt
    
    def load_data(self):
        """
        Load the test dataset from the specified path.
        """
        #read test dataset
        # print(f'test_dataset_path: {dataset_path}')
        if self.test_dataset_name == 'test':
            test_df = pd.read_csv('.../dummy_data/test.csv')
        else:
            raise KeyError('No valid test_dataset_name!')
        
        return test_df

    def get_open_content(self):
        """
        Get the open content for the test dataset.
        The open content contains a number of examples from the training set.
        """
        df_train = '.../dummy_data/train.csv'

        query_list = df_train['User Message'].tolist()
        memory_list = df_train['Updated Memory'].tolist()
        context_list = df_train['context'].tolist()
        personal_data_list = df_train['personal_data'].tolist()
        rephrased_message_list = df_train['rephrased_message'].tolist()
        # only select the query which is shorter than 500 tokens
        tokenizer = self.load_tokenizer()
        filtered_query_list = []
        filtered_memory_list = []
        filtered_context_list = []
        filtered_personal_data_list = []
        filtered_rephrased_message_list = []

        # No filtering for now
        filtered_query_list = query_list
        filtered_memory_list = memory_list
        filtered_context_list = context_list
        filtered_personal_data_list = personal_data_list
        filtered_rephrased_message_list = rephrased_message_list

        # check how many examples we need in the open content
        num_example = self.config.get('open_content', 5)
        if num_example > len(filtered_query_list):
            num_example = len(filtered_query_list)
            print(f'Warning: The number of examples in the open content is larger than the number of available examples. Set to {num_example}.')
        selected_indices = random.sample(range(len(filtered_query_list)), num_example)
        selected_queries = [filtered_query_list[i] for i in selected_indices]
        selected_memories = [filtered_memory_list[i] for i in selected_indices]
        selected_contexts = [filtered_context_list[i] for i in selected_indices]
        selected_personal_data = [filtered_personal_data_list[i] for i in selected_indices]
        selected_rephrased_message = [filtered_rephrased_message_list[i] for i in selected_indices]

        open_content = ""
        for query, memory, context, personal_data, rephrased_message in zip(selected_queries, selected_memories, selected_contexts, selected_personal_data, selected_rephrased_message):
            open_content += f'User Query: {query}\nRelevant Memory: {memory}\nContext: {context}\nPersonal Data: {personal_data}\nRephrased Message: {rephrased_message}\n\n'

        return open_content

    def construct_prompt(self):
        """
        Construct the prompt string based on the configuration parameters.
        """

        df_test = self.load_data()

        if self.test_dataset_name == 'rephrased_icl' or self.test_dataset_name == 'rephrased_ft':
            queries = df_test['rephrased prediction'].tolist()
            memory = ["NA"] * len(df_test)
        else:
            queries = df_test['User Message'].tolist()
            memory = df_test['Updated Memory'].tolist()
        user_ids = df_test['user_id'].tolist()
        conversion_ids = df_test['conversation_id'].tolist()
        message_ids = df_test['message_id'].tolist()
        context = df_test['context'].tolist() 
        # personal_data = df_test['personal_data'].tolist() 
        # rephrased_message = df_test['rephrased_message'].tolist() 
 
        base_prompts = []

        for _, row in df_test.iterrows():
            query = row['User Message']
            if self.test_dataset_name == 'query-memory':
                if self.open_content:
                    open_content = self.get_open_content()
                    user_instruction = f"""'Here are some examples of user queries, the relevant memories, the context of the user query which is a list of previous 1 to 3 user queries,
                    the personal data in the form of (verbatim data from query, GDPR article with classification) extracted from the user queries, if there are any, 
                    and the rephrased queries if personal data is present: \n {open_content}. 
                    Your task is to predict *memory*, *personal data* and *rephrased query*. Do not attach the context while predicting memory.
                    The memory should primarily be extracted from the user query, and if needed, you can extract from the context for completeness of the memory. 
                    Remember to extract personal data from the user query only. Do not extract personal data from the context.
                    The rephrased query should be generic and seek the same core information as in user query, without revealing any personal details about the user. 
                    If personal data is "NA", rephrased query should also be "NA".
                    Follow the trend carefully in the in-context examples and do the following.
                    
                    Given the context and user query, your task is to identify the underlying pattern and predict *Memory*, *Personal data* and *Rephrased message*.\n
                    Context: {row["context"]}
                    User Query: {query}"""
                else:
                    user_instruction = f'Please extract the key information in user query, which should be put into your external memory. \n Query: {query}'
                
                # if the user instruction is too long, truncate it
                tokenizer = self.load_tokenizer()
                if len(tokenizer.encode(user_instruction)) > 20000:
                    user_instruction = ''
                    print('Warning: The user instruction is too long, truncate it to empty string.')
                base_prompts.append(user_instruction)
            elif self.test_dataset_name == 'query-all3' or self.test_dataset_name == 'query-all3-full-conversation' or self.test_dataset_name == 'all-user-limit' or self.test_dataset_name == 'all-user-full' or self.test_dataset_name == 'first-queries' or self.test_dataset_name == 'rephrased_icl' or self.test_dataset_name == 'rephrased_ft':
                if self.open_content:
                    open_content = self.get_open_content()
                    user_instruction = f"""'Here are some examples of user queries, the relevant memories, the context of the user query which is a list of previous 1 to 3 user queries,
                    the personal data in the form of (verbatim data from query, GDPR article with classification) extracted from the user queries, if there are any, 
                    and the rephrased queries if personal data is present: \n {open_content}. 
                    Your task is to predict *memory*, *personal data* and *rephrased query*. Do not attach the context while predicting memory.
                    The memory should primarily be extracted from the user query, and if needed, you can extract from the context for completeness of the memory. 
                    Remember to extract personal data from the user query only. Do not extract personal data from the context.
                    The rephrased query should be generic and seek the same core information as in user query, without revealing any personal details about the user. 
                    If personal data is "NA", rephrased query should also be "NA".
                    Follow the trend carefully in the in-context examples and do the following.
                    
                    Given the context and user query, your task is to identify the underlying pattern and predict *Memory*, *Personal data* and *Rephrased message*.\n
                    Context: {row["context"]}
                    User Query: {query}"""
                else:
                    user_instruction = f"""Given the context and user query, your task is to identify the underlying pattern and predict memory, personal data and rephrased query.\n
                    Context: {row["context"]}
                    User Query: {query}"""
                # filter 
                tokenizer = self.load_tokenizer()
                if len(tokenizer.encode(user_instruction)) > 10000:
                    user_instruction = ''
                    print('Warning: The user instruction is too long, truncate it to empty string.')
                base_prompts.append(user_instruction)
                
        print(f'Number of test examples: {len(base_prompts)}')
        if self.chat_template:
            system_prompt = """You are a highly precise data privacy analyst analysing conversations. Given the context and user query, provide the generated memory, personal data as per GDPR 4.1 and 9.1, and rephrased queries. 
                    Do not infer memories from other queries or context. \n Please answer the query as the same language as the user query. \n
                    Keep your attention window till the entire feed of the considered conversation meant for that user only. 
                    The rephrased query should be generic and seek the same core information as in user query, without revealing any personal details about the user.

                    FYI: GDPR Definitions with Classifications

                        **GDPR Article 4(1) - Personal Data: Classifications such as a name, an identification number, location data, an online identifier, or factors specific to the physical, physiological, genetic, mental, economic, cultural or social identity.**
                        **GDPR Article 9(1) - Special Category Personal Data: Data classified into racial or ethnic origin, political opinions, religious or philosophical beliefs, trade union membership, genetic data, biometric data, data concerning health, or data concerning a natural person’s sex life or sexual orientation.**
                        
                        If no personal data is present in the user_query, output "Personal Data": "NA".
                        If there is no rephrased query (because no personal data exists), output "Rephrased Query": "NA"."""
            base_prompts = [self.apply_chat_templates(self.chat_template, p, system_prompt) for p in base_prompts]

        return base_prompts, (queries, context, memory, ['no personal data here'], ['no rephrased message here']), user_ids, conversion_ids, message_ids


