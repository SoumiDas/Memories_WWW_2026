#!/bin/bash

# Parameters
lrs=(
    # 0.1
    # 0.05
    # 0.01
    # 0.005
    # 0.001
    # 0.0005
    # 0.0001
    # 0.00008
    0.00005 
    # 0.00001
    # 0.000008
    # 0.000005
    # 0.000003 
    # 0.000001
    # 0.0000005
)

seed=42

# Generate a range of 100 unique master ports
master_ports=($(seq 4910 5009))

# Logging directory
log_dir=".../_logging"
mkdir -p "$log_dir"  # Ensure the logging directory exists

# Validate master port range
if [[ ${#master_ports[@]} -lt 1 ]]; then
    echo "Error: No master ports available."
    exit 1
fi

for lr in "${lrs[@]}"; do
    master_port="${master_ports[RANDOM % ${#master_ports[@]}]}"

    # Unique job name
    job_name="FT_lr${lr}"

    # Prepare a unique job script
    job_script="${log_dir}/${job_name}.sh"
    cat > "$job_script" << EOL
#!/bin/bash
#SBATCH --partition=h200
#SBATCH --gres=gpu:8
#SBATCH -c 128
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01-00:00
#SBATCH --mem=1600GB

#SBATCH -o ${log_dir}/${job_name}_%j.out
#SBATCH -e ${log_dir}/${job_name}_%j.err
#SBATCH --job-name=${job_name}

export GPUS_PER_NODE=8
export MASTER_PORT=${master_port}

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR='.../util_public/training/.cache/triton'

# Set for A100, H100 and H200
export CUDA_HOME=/usr/lib/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

accelerate launch --config_file=".../util_public/training/config/sft_ds_z3.yaml" \
    --num_processes 8 \
    --main_process_port 3000 \
    -m train_imitating_memory_extractor.main \
    --train_template_config_path ".../training/training_config_offload.yaml" \
    --base_model_name "qwen2.5-32b-instruct" \
    --data_name 'query-all3-full-conversation' \
    --train_ratio 0.6 \
    --epoch 20 \
    --learning_rate $lr \
    --lr_scheduler_type 'cosine' \
    --seed $seed \
    --chat_template .../util_public/chat_templates/qwen2.5.jinja \
    --loss 'sft' 

EOL
    # Ensure the script is executable
    # _${train_relation_id}_sync_random_o_hgp-0_old-0_s-o_1_trex_MC-50_1e-05_cosine_0-99_checkpoint-1000
    chmod +x "$job_script"

    # Submit the job
    sbatch "$job_script"
    if [[ $? -eq 0 ]]; then
        echo "Job ${job_name} submitted successfully."
    else
        echo "Failed to submit job ${job_name}."
    fi

done
