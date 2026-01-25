import pandas as pd
import yaml
import os
import json
import warnings
import random
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from ..util_public.get_model_path import ModelPath

class ConstructDatasetCostume:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.config_path = config_path
        self.model_name = self.config.get('model_name')
        self.model_path_map = self.config.get('model_path_map')
        self.train_data_path = self.config.get('train_data_path')
        self.train_data_name = self.config.get('train_data_name')
        self.train_ratio = self.config.get('train_ratio')
        self.chat_template = self.config.get('chat_template')
        self.loss = self.config.get('loss', 'unsupervised').lower()  # <<< changed

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_tokenizer(self):
        ModelPath_obj = ModelPath(self.config_path)
        model_path = ModelPath_obj.get_model_path()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def apply_chat_templates(self, system_prompt, user_prompt, response):
        """
        Apply chat templates for the model; returns the rendered string (not tokenized).
        """
        tokenizer = self.load_tokenizer()
        with open(self.chat_template, 'r') as f:
            tokenizer.chat_template = f.read()

        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
    
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # <<< changed: full string with assistant included
        )
        return full_prompt
    
    def load_train_data(self):
        '''
        Load the train dataset from the specified path.
        Splits data into train/test for "with memory" and saves test + without-memory.
        '''
        all_files = ['.../dummy_data/train.csv'] # add more training data here

        df_train_parts = []
        df_test_parts = []
        df_wo_memory_parts = []

        for file in all_files:
            df = pd.read_csv(file)
            # Be robust to NaNs
            df['Updated Memory'] = df['Updated Memory'].fillna('No memory')

            # Split into with-memory and without-memory
            df_w_memory = df[df['Updated Memory'] != 'No memory']
            df_wo_memory = df[df['Updated Memory'] == 'No memory']
            
            if self.train_ratio == 1.0:
                # don't split, all in train
                df_train = df_w_memory
                df_test = df_w_memory.iloc[0:0]  # empty
            else:
                # Train/test split per file (guard against tiny splits)
                if len(df_w_memory) >= 2 and int(len(df_w_memory) * float(self.train_ratio)) >= 1:
                    df_train, df_test = train_test_split(
                        df_w_memory, train_size=float(self.train_ratio), random_state=42, shuffle=True
                    )
                else:
                    # If too small to split, put everything in train and leave test empty
                    df_train, df_test = df_w_memory, df_w_memory.iloc[0:0]

            print(f"File: {file}, Train size: {len(df_train)}, Test size: {len(df_test)}, Without memory size: {len(df_wo_memory)}")
            print(f"Sample data:\n{df_train.head(1)}")

            # Accumulate (tag with source file for traceability)
            df_train_parts.append(df_train.assign(_source=file))
            df_test_parts.append(df_test.assign(_source=file))
            df_wo_memory_parts.append(df_wo_memory.assign(_source=file))

            print(f"Accumulated sizes so far — train: {sum(len(d) for d in df_train_parts)}, test: {sum(len(d) for d in df_test_parts)}, without-memory: {sum(len(d) for d in df_wo_memory_parts)}")

        # --- Merge all files here ---
        df_train_all = pd.concat(df_train_parts, ignore_index=True) if df_train_parts else pd.DataFrame()
        df_test_all = pd.concat(df_test_parts, ignore_index=True) if df_test_parts else pd.DataFrame()
        df_wo_memory_all = pd.concat(df_wo_memory_parts, ignore_index=True) if df_wo_memory_parts else pd.DataFrame()

        print(f"TOTALS — train: {len(df_train_all)}, test: {len(df_test_all)}, without-memory: {len(df_wo_memory_all)}")

        # Save df_test
        save_dir = '.../dummy_data'
        os.makedirs(save_dir, exist_ok=True)
        df_train_all.to_csv(os.path.join(save_dir, f'train_{self.train_data_name}.csv'), index=False)
        df_test_all.to_csv(os.path.join(save_dir, f'test_{self.train_data_name}.csv'), index=False)
        df_wo_memory_all.to_csv(os.path.join(save_dir, f'without_memory_{self.train_data_name}.csv'), index=False)

        return df_train_all

    def construct_train_dataset(self):
        # 1) Load and sanity-check
        df_train = self.load_train_data()

        user_prompts, responses, system_prompts = [], [], []  

        required_cols = ['User Message', 'Updated Memory', 'personal_data', 'rephrased_message', 'context']
        missing = [c for c in required_cols if c not in df_train.columns]
        if missing:
            raise KeyError(f"Missing required columns in df_train: {missing}")

        for _, row in df_train.iterrows():
            personal_data = 'No personal data extracted.' if pd.isna(row['personal_data']) else row['personal_data']
            context = '' if pd.isna(row['context']) else row['context']
            rephrased = 'No rephrased message.' if pd.isna(row['rephrased_message']) else row['rephrased_message']

            user_prompt = (
                "Given the context and user query, your task is to identify the underlying pattern and "
                "predict memory, personal data and rephrased query.\n"
                f"Context: {context}\n"
                f"User Query: {row['User Message']}"
            )
            response = (
                "{\n"
                f'  "Memory": "{row["Updated Memory"]}",\n'
                f'  "Personal Data": "{personal_data}",\n'
                f'  "Rephrased Query": "{rephrased}"\n'
                "}"
            )
            user_prompts.append(user_prompt)
            responses.append(response)
            system_prompts.append("")  # keep optional system blank for now

        system_prompt_default =  (
            "You are a highly precise data privacy analyst analysing conversations. Given the context and user query, "
            "provide the generated memory, personal data as per GDPR 4.1 and 9.1, and rephrased queries. "
            "Do not infer memories from other queries or context. "
            "Keep your attention window till the entire feed of the considered conversation meant for that user only. "
            "The rephrased query should be generic and seek the same core information as in user query, "
            "without revealing any personal details about the user.\n\n"
            "FYI: GDPR Definitions with Classifications\n\n"
            "**GDPR Article 4(1) - Personal Data: Classifications such as a name, an identification number, "
            "location data, an online identifier, or factors specific to the physical, physiological, genetic, mental, "
            "economic, cultural or social identity.**\n"
            "**GDPR Article 9(1) - Special Category Personal Data: Data classified into racial or ethnic origin, "
            "political opinions, religious or philosophical beliefs, trade union membership, genetic data, biometric data, "
            "data concerning health, or data concerning a natural person’s sex life or sexual orientation.**\n\n"
            'If no personal data is present in the user_query, output "Personal Data": "NA".\n'
            'If there is no rephrased query (because no personal data exists), output "Rephrased Query": "NA".'
        )

        # 2) Prepare serialized text if using chat templates or plain text otherwise
        pre_dataset = []
        using_chat_template = bool(self.chat_template)

        if using_chat_template:
            # render full messages to text (assistant included)
            for sp, up, rp in zip(system_prompts, user_prompts, responses):
                sp_eff = sp if sp else system_prompt_default
                pre_dataset.append(self.apply_chat_templates(sp_eff, up, rp))
        else:
            # fall back to simple concatenation (assistant included)
            for up, rp in zip(user_prompts, responses):
                pre_dataset.append(f"{up}\n{rp}")

        # 3) Shuffle
        random.seed(42)  # deterministic
        random.shuffle(pre_dataset)
        # 5) Tokenize
        tokenizer = self.load_tokenizer()

        # filter out rows over than 20k tokens
        filtered_dataset = []
        for text in pre_dataset:
            enc = tokenizer(text, truncation=False)
            if len(enc['input_ids']) <= 20000:
                filtered_dataset.append(text)
            else:
                warnings.warn(f"Skipping a row with {len(enc['input_ids'])} tokens (exceeds 20k limit).")
        pre_dataset = filtered_dataset
        
        # 4) Create a DataFrame then a HF Dataset with a 'text' column
        text_df = pd.DataFrame({'text': pre_dataset})
        print(f"Example of the training data:\n{text_df.head()}")
        ds = Dataset.from_pandas(text_df, preserve_index=False)

        # --- Build a parallel, non-shuffled structure that preserves boundaries for SFT ---
        # We need boundaries to compute the prompt length reliably.
        boundary_rows = list(zip(system_prompts, user_prompts, responses))  # original order
        if using_chat_template and self.loss == 'sft':
            # Build a parallel Dataset that keeps message boundaries so we can mask accurately.
            df_bound = pd.DataFrame(
                {"system": [s or "" for s in system_prompts],
                 "user": user_prompts,
                 "assistant": responses}
            )
            # Keep same shuffle to align with text_df order
            random.seed(42)
            idx = list(range(len(df_bound)))
            random.shuffle(idx)
            df_bound = df_bound.iloc[idx].reset_index(drop=True)

        def _tokenize_unsup(batch):
            out = tokenizer(batch['text'], truncation=True)
            out["labels"] = out["input_ids"].copy()
            return out

        def _tokenize_sft(batch, batch_indices=None):
            # batch contains 'text' rows in shuffled order; we align with df_bound using indices
            input_ids_list = []
            attn_mask_list = []
            labels_list = []

            for i, text in enumerate(batch['text']):
                # Fetch the corresponding messages (system, user, assistant)
                if using_chat_template:
                    # alignment: batch_indices gives original row indices in current batch map call
                    row = df_bound.iloc[batch_indices[i]]
                    system_p = row['system']
                    user_p = row['user']
                    assistant_p = row['assistant']

                    # Load/refresh chat template into tokenizer
                    with open(self.chat_template, 'r') as f:
                        tokenizer.chat_template = f.read()

                    # Build message lists
                    msgs_prompt = [{"role": "user", "content": user_p}]
                    if system_p:
                        msgs_prompt = [{"role": "system", "content": system_p}] + msgs_prompt

                    msgs_full = msgs_prompt + [{"role": "assistant", "content": assistant_p}]

                    # Tokenize both
                    prompt_enc = tokenizer.apply_chat_template(
                        msgs_prompt, tokenize=True, add_generation_prompt=True, return_tensors=None
                    )
                    full_enc = tokenizer.apply_chat_template(
                        msgs_full, tokenize=True, add_generation_prompt=False, return_tensors=None
                    )

                    input_ids = full_enc
                    attention_mask = [1] * len(full_enc)

                    prompt_len = len(prompt_enc)
                    # Sanity: if mismatch, try to align by prefix match
                    if input_ids[:prompt_len] != prompt_enc:
                        # fallback: find the longest common prefix
                        pref = 0
                        for a, b in zip(input_ids, prompt_enc):
                            if a == b:
                                pref += 1
                            else:
                                break
                        prompt_len = pref

                    labels = [-100] * prompt_len + input_ids[prompt_len:]

                    input_ids_list.append(input_ids)
                    attn_mask_list.append(attention_mask)
                    labels_list.append(labels)
                else:
                    # No chat template: assume plain "user\nassistant" format and mask the "user\n" part.
                    # We’ll split at the last newline as a heuristic.
                    user_assist_split = text.rfind("\n")
                    if user_assist_split == -1:
                        enc = tokenizer(text, truncation=True)
                        input_ids_list.append(enc["input_ids"])
                        attn_mask_list.append(enc["attention_mask"])
                        labels_list.append([-100] * len(enc["input_ids"]))  # no assistant found
                    else:
                        user_part = text[:user_assist_split+1]  # include newline
                        enc_full = tokenizer(text, truncation=True)
                        enc_user = tokenizer(user_part, truncation=True)
                        input_ids = enc_full["input_ids"]
                        attention_mask = enc_full["attention_mask"]
                        prompt_len = len(enc_user["input_ids"])
                        labels = [-100] * prompt_len + input_ids[prompt_len:]
                        input_ids_list.append(input_ids)
                        attn_mask_list.append(attention_mask)
                        labels_list.append(labels)

            return {
                "input_ids": input_ids_list,
                "attention_mask": attn_mask_list,
                "labels": labels_list,
            }

        if self.loss == 'unsupervised':
            ds = ds.map(_tokenize_unsup, batched=True)
        elif self.loss == 'sft':
            if using_chat_template:
                # HuggingFace map gives indices via with_indices=True
                ds = ds.map(
                    _tokenize_sft,
                    batched=True,
                    with_indices=True,  # <<< needed so we can align to df_bound
                    fn_kwargs={},        # none
                )
            else:
                ds = ds.map(_tokenize_sft, batched=True)
        else:
            raise ValueError(f"Unknown loss type: {self.loss}. Use 'unsupervised' or 'sft'.")

        # 6) Compute total tokens

        # trunct the sequence to max length 8k
        def _truncate_to_maxlen(sample, max_len=8192):
            input_ids = sample['input_ids'][:max_len]
            attention_mask = sample['attention_mask'][:max_len]
            labels = sample['labels'][:max_len]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        
        ds = ds.map(_truncate_to_maxlen)
        total_tokens = int(sum(len(ids) for ids in ds['input_ids']))
        print(f"Total tokens in the dataset: {total_tokens}")

        return ds
