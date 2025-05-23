#!/bin/bash

# Array of train sets
declare -A MODEL_NAMES=(
    ["qwen2.5"]="Qwen/Qwen2.5-7B-Instruct"
    ["llama3.1"]="meta-llama/Llama-3.1-8B-Instruct"
    ["saillama"]="GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
    ["aisgllama"]="aisingapore/Llama-SEA-LION-v3-8B-IT"
    ["gemma"]="google/gemma-2-9b-it"
)
train_sets=(
    # 'jvsu_llm_filtered'
    # 'jvsu_localize_gpt'
    'jvsu_llm'
    'jvsu_mt'
    )

for train_set in "${train_sets[@]}"; do
    for key in "${!MODEL_NAMES[@]}"
    do
        model_name="${MODEL_NAMES[$key]}"
        echo "Running generator train.py with train_set: $train_set for model: $model_name"
        # python train.py --model_name "$model_name" --train_set "$train_set"

        # Check if the script executed successfully
        if [ $? -ne 0 ]; then
            echo "Error: train.py failed for train_set: $train_set"
            exit 1
        fi

        echo "Completed train.py for train_set: $train_set, $model_name"
        echo "-------------------------------------------"
        echo "Running evaluation for train_set: $train_set, $model_name"
        python evaluation.py --model_name "$model_name" --train_set "$train_set"
    done
done
echo "All training have been processed successfully."