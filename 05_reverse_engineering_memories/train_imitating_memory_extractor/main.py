"""
This script continues training a model with new data.
You can use it to customize the loss computation on certain tokens.
"""

import os
import json
import yaml
import argparse
import time

from .construct_dataset_custome import ConstructDatasetCostume as ConstructDataset
from ..util_public.training.continue_training_custome import ContinueTrainingCustome as ContinueTraining
from .config_generation import generate_train_config  # if used elsewhere

# ----------------------------
# Utilities
# ----------------------------

def write_model(output_dir: str):
    """
    Record trained checkpoint(s) in a central models.json mapping.

    The mapping file format is:
        {
          "name": ["checkpoint_folder_name", "full/output/dir"]
        }
    """
    if not output_dir or not os.path.isdir(output_dir):
        raise ValueError(f"Invalid output_dir: {output_dir}")

    model_mapping = '.../util_public/models.json'
    # Ensure the mapping file exists
    os.makedirs(os.path.dirname(model_mapping), exist_ok=True)
    if not os.path.exists(model_mapping):
        with open(model_mapping, 'w') as f:
            json.dump({}, f, indent=4)

    with open(model_mapping, 'r') as file:
        model_map = json.load(file)

    # Read subfolders in output_dir and pick checkpoints
    try:
        model_folder = os.listdir(output_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to listdir({output_dir}): {e}")

    model_name_list = []
    for folder in model_folder:
        if folder.startswith('checkpoint'):
            # model_name = basemodel_relation_text_num_index_ckp
            # Compose something stable and informative
            # e.g. parentdir_currdir_checkpoint-xxx
            parts = output_dir.rstrip('/').split('/')
            if len(parts) >= 2:
                model_name = f"{parts[-2]}_{parts[-1]}_{folder}"
            else:
                model_name = f"{parts[-1]}_{folder}"

            model_map[model_name] = [f"{folder}", f"{output_dir}"]
            model_name_list.append(model_name)

    with open(model_mapping, 'w') as file:
        json.dump(model_map, file, indent=4)

    print(f"Model entries added to the model mapping file: {model_name_list if model_name_list else 'None found'}")
    return model_name_list


def new_training_config(
    base_model_name: str,
    data_name: str,
    train_ratio: float,
    train_template_config_path: str,
    epoch: int = 50,
    learning_rate: float = 2e-6,
    lr_scheduler_type: str = 'linear',
    chat_template: str = None,
    seed: int = 42,
    loss: str = 'unsupervised'
) -> str:
    """
    Load a training template config YAML, override a few fields, and write a new YAML to a temp path.
    Returns the path to the new YAML.
    """
    if not os.path.exists(train_template_config_path):
        raise FileNotFoundError(f"Training template config not found: {train_template_config_path}")

    with open(train_template_config_path, 'r') as file:
        train_config = yaml.safe_load(file)

    # Required keys that should exist in the template
    required_keys = ['per_device_train_batch_size']
    for k in required_keys:
        if k not in train_config:
            raise KeyError(f"Missing required key '{k}' in training template config: {train_template_config_path}")

    train_config['model_name'] = base_model_name
    train_config['train_data_name'] = data_name
    train_config['train_ratio'] = train_ratio
    train_config['epochs'] = epoch
    train_config['learning_rate'] = learning_rate
    train_config['lr_scheduler_type'] = lr_scheduler_type
    train_config['chat_template'] = chat_template
    train_config['seed'] = seed
    train_config['loss'] = loss

    batch_size = train_config['per_device_train_batch_size']

    # Shorten extremely long model names to keep path manageable
    base_for_path = base_model_name
    if len(base_for_path) > 200:
        all_tokens = base_for_path.split('-')
        checkpoint = all_tokens[-1] if all_tokens else 'ckpt'
        base_for_path = f'qwen2.5-shorten-{checkpoint}'

    # Compose output dir
    train_config['output_dir'] = (
        f"/NS/chatgpt/nobackup/qwu/self_trained/"
        f"{base_for_path}_{data_name}_train-{loss}_ratio-{train_ratio}_{batch_size}_{learning_rate}_{lr_scheduler_type}"
    )
    if chat_template:
        train_config['output_dir'] += "_chat-template"

    # WandB run name (shortened if too long)
    wandb_run_name = f"{base_for_path}_{data_name}_{train_ratio}_{batch_size}"
    if len(wandb_run_name) > 100:
        wandb_run_name = wandb_run_name[:100]
    train_config['wandb_run_name'] = wandb_run_name

    # Write new config
    os.makedirs(".../_temp", exist_ok=True)
    new_train_config_path = (
        f".../_temp/"
        f"training_config_{base_for_path}_{data_name}_{loss}_{batch_size}_{learning_rate}_{lr_scheduler_type}.yaml"
    )

    with open(new_train_config_path, 'w') as file:
        yaml.dump(train_config, file)

    # Give the filesystem a moment on shared systems
    time.sleep(5)
    return new_train_config_path


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Continue training the model with the new data.')
    parser.add_argument('--base_model_name', type=str, required=True,
                        help='The base model name to continue training.')
    parser.add_argument('--data_name', type=str, required=True,
                        help='The data name to continue training on.')
    parser.add_argument('--train_template_config_path', type=str,
                        default='.../train_imitating_memory_extractor/training_config.yaml',
                        help='Path to the training template config YAML.')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-6, help='Learning rate.')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='LR scheduler type.')
    parser.add_argument('--chat_template', type=str, default=None,
                        help='Chat template identifier (or leave as None).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--loss', type=str, default='unsupervised', help='Loss type: supervised or unsupervised.')
    args = parser.parse_args()

    print('Arguments received:')
    print(f'Base model name: {args.base_model_name}')
    print(f'Data name: {args.data_name}')
    print(f'Train template config path: {args.train_template_config_path}')
    print(f'Epochs: {args.epoch}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'LR scheduler type: {args.lr_scheduler_type}')
    print(f'Chat template: {args.chat_template}')
    print(f'Seed: {args.seed}')
    print(f'Training Ratio: {args.train_ratio}')
    print(f'Loss type: {args.loss}')

    # Create a new, concrete training config from the template
    new_train_template_config_path = new_training_config(
        base_model_name=args.base_model_name,
        data_name=args.data_name,
        train_ratio=args.train_ratio,
        train_template_config_path=args.train_template_config_path,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        chat_template=args.chat_template,
        seed=args.seed,
        loss=args.loss
    )

    print(f'Training config path: {new_train_template_config_path}')
    print('Constructing the dataset...')

    dataset_builder = ConstructDataset(new_train_template_config_path)
    ds = dataset_builder.construct_train_dataset()

    print('Dataset constructed.')

    trainer = ContinueTraining(new_train_template_config_path, ds)
    # print("HF args.deepspeed:", getattr(trainer.args, "deepspeed", None))
    # print("Trainer.deepspeed engine:", type(getattr(trainer, "deepspeed", None)))
    output_dir = trainer.main()

    if not output_dir:
        raise RuntimeError("Training finished but no output_dir was returned by ContinueTraining.main().")

    print('Training is done. Model saved at:', output_dir)

    # Write the trained checkpoints into the mapping file
    model_name_list = write_model(output_dir)
    if model_name_list:
        print('Registered model names:', model_name_list)
    else:
        print('No checkpoint folders found to register.')


if __name__ == "__main__":
    main()
