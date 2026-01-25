import json
import yaml
import os

class ModelPath:
    def __init__(self, config_path):
        self.config = self.read_config(config_path)
        self.model_name = self.config.get('model_name')
        self.model_path_map = self.config.get('model_path_map')
    
    def read_config(self, path):
        """
        Read the YAML configuration file and return the config dictionary.
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def parse_model_name_to_path(self):
        base_path = '.../self_trained'
        # split the model name by '_'
        parts = self.model_name.split('_')
        save_label = parts[-2]
        checkpoints = parts[-1]
        left_part = '_'.join(parts[:-2])
        full_model_path = os.path.join(base_path, left_part, save_label, checkpoints)
        return full_model_path
        
    def get_model_path(self):
        #model_path_map is a dictionary
        models_map_path = self.model_path_map

        with open(models_map_path, 'r') as file:
            model_path_map = json.load(file)

        if self.model_name not in model_path_map:
            return self.parse_model_name_to_path()
        
        else:
            #get the model path by model name
            model_entry = model_path_map[self.model_name]
            base_path = model_entry[1]
            model_path = os.path.join(base_path, model_entry[0])
        return model_path

    def get_tokenizer_path(self):
        return self.get_model_path() #use the same path as model pasth

