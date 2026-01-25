#!/bin/bash
# language="Japanese"
train_dataset_name='query-all3-full-conversation' #'all-user-limit' #'query-all3' #'query-all3-full-conversation' #'query-no-memory'   #'query-memory-train' # #'query-memory-test' #
test_dataset_name='rephrased_ft' #'query-all3' #'all-user-limit' #'all-user-full'
seed='42'

base_model="qwen2.5-32b-instruct_all-user-full_train-sft_ratio-1.0_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-380"
# base_model="gpt-oss-20b"
# base_model='gemma-3-27b-it'
# base_model='mistral-small-3.1-24b-instruction'
chat_template='.../util_public/chat_templates/qwen2.5.jinja'
# chat_template='.../util_public/chat_templates/gemma3.jinja'
# chat_template='.../util_public/chat_templates/mistral3.1.jinja'
# chat_template='.../util_public/chat_templates/gpt-oss.jinja'

open_content=10  

# lr='1e-06'
lrs=(
    # 0.1
    # 0.05
    # 0.01
    # 0.005
    # 0.001
    # 0.0005
    # 0.0001
    5e-05
    # 1e-05
    # 5e-07
    # 8e-06
    # 5e-06
    # 3e-06
    # 1e-06
)
lr_schedule='cosine'

max_new_tokens=500

for lr in "${lrs[@]}"; do
    model_names=(
        "${base_model}"
    )
    # for checkpoint in {40..40..2}; do
    #     # model_names+=("${base_model}_query-all3_train-unsupervised_ratio-0.6_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-${checkpoint}")
    #     model_names+=("${base_model}_${train_dataset_name}_train-sft_ratio-0.6_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-${checkpoint}")
    # done

    # Logging directory
    log_dir=".../_logging"
    mkdir -p "$log_dir"  # Ensure the logging directory exists
    for model_name in "${model_names[@]}"; do
        # Unique job name
        job_name="${model_name}_${relation_id}_eval_${lke_type}"
        # If the model name is too long, truncate it
        if [[ ${#job_name} -gt 200 ]]; then
            job_name="${job_name:0:200}"
        fi

        # Prepare a unique job script
        job_script="${log_dir}/${job_name}.sh"
        cat > "$job_script" << EOL
#!/bin/bash
#SBATCH --partition=a100,h100,h200
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01-00:00
#SBATCH --mem=1600GB
#SBATCH --exclude=sws-3a100grid-01
#SBATCH -o ${log_dir}/${job_name}_%j.out
#SBATCH -e ${log_dir}/${job_name}_%j.err
#SBATCH --job-name=${job_name}

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export GPUS_PER_NODE=$(scontrol show job $SLURM_JOB_ID | grep -oP "(?<=GRES=gpu:)\d+")

export TOKENIZERS_PARALLELISM=false

python -m inference_imitating_memory_extractor.main \
--config_path '.../evaluation/config.yaml' \
--model_name "$model_name" \
--test_dataset_name "$test_dataset_name" \
--max_new_tokens "$max_new_tokens" \
--chat_template "$chat_template" 


EOL
        # Ensure the script is executable
        chmod +x "$job_script"

        # Submit the job
        sbatch "$job_script"
        if [[ $? -eq 0 ]]; then
            echo "Job ${job_name} submitted successfully."
        else
            echo "Failed to submit job ${job_name}."
        fi
    done
done