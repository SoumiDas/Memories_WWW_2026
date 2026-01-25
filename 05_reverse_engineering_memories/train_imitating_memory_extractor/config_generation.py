'''
This file is used to generate different configurations for the training process and evaluation process.
'''

import os
import json
import yaml

def get_tokenizer_path(model_name):
    model_mapping = '.../util_public/models.json'
    with open(model_mapping, 'r') as file:
        model_map = json.load(file)
        #get the model path by model name
    model_entry = model_map[model_name]
    base_path = model_entry[1]
    model_path = os.path.join(base_path, model_entry[0])
    return model_path

def generate_train_config(
        train_config_template,
        train_data_index_list,
):
    #load the template config file
    try:
        with open(train_config_template, 'r') as file:
            config = yaml.safe_load(file)
        print("Config loaded from file:")
        print(config)
        if config is None:
            print("The loaded config is None, which indicates an issue with the file content or format.")
    except Exception as e:
        print(f"Error reading config: {e}")

    config_list = []
    for train_data_index in train_data_index_list:
        new_config = config.copy()
        #update the model name
        new_config['train_data_index_list'] = train_data_index
        new_config['save_label'] = f'{train_data_index[0]}-{train_data_index[-1]}'   
        #save the new config to the save_path
        #get the value of config['model_name']
        model_name = config['model_name']
        save_dir = f'.../_temp/{model_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        else:
            print(f'{save_dir} already exists.')
        save_file = os.path.join(save_dir, f'training_config_{train_data_index[0]}.yaml')
        #rewrite
        with open(save_file, 'w') as file:
            yaml.dump(new_config, file)
        config_list.append(save_file)
        print(f'Config file saved at {save_file}')

    return config_list, model_name


