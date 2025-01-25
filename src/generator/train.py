from unsloth import FastLanguageModel
import argparse
import torch
import random
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

class Args:
    def __init__(self):
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument("--train_set", default="id", help="train set to choose")
        args_parser.add_argument('--model_name', default='sahabat', help='model name')
        args_parser.add_argument('--seed', type=int, default=3407, help='random seed')

        # Parse arguments
        self.args = args_parser.parse_args()

# Set arguments for training or evaluation
args = Args().args
set_seed(args)
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

if args.model_name == 'cendol':
    model_name = "indonlp/cendol-llama2-7b-chat"
elif args.model_name == 'komodo':
    model_name = "Yellow-AI-NLP/komodo-7b-base"
elif args.model_name == 'sahabat':
    model_name = 'GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct'
else:
    assert False, "Model name not found"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 128,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

import pandas as pd
train_set = args.train_set
data_path = f'../../dataset/train/{train_set}.csv'
def load_data(data_path):
    data = pd.read_csv(data_path)
    contexts = []
    endings = []
    EOS_TOKEN = tokenizer.eos_token
    for idx, row in data.iterrows():
        sents = []
        for i in [4,3,2,1]:
            current_sentence = row[f'sentence_{i}']
            current_sentence = current_sentence + '.' if current_sentence[-1] != '.' else current_sentence
            sents.insert(0, current_sentence)
        context = ' '.join(sents) 
        ending = row['correct_ending']
        # ending2 = row['incorrect_ending']
        contexts.append(context)
        endings.append(ending)

    # convert to huggingface dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({'context': contexts, 'ending': endings})
    def formatting_prompts_func(examples):
        contexts  = examples["context"]
        endings = examples["ending"]
        texts = []
        for context, ending in zip(contexts, endings):
            chat = [
                {"role": "user", "content": f"Generate the ending for the following given story context\nStory Context: {context}"},
                {"role": "assistant", "content": f"{ending}"},
            ]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    return dataset
dataset = load_data(data_path)

outputs_dir = f"outputs/{model_name.split('/')[-1]}/{train_set}"
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 8,
        num_train_epochs = 1,
        # max_steps=200,
        warmup_steps = 5,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = outputs_dir,
        report_to = "none", # Use this for WandB etc
    ),
)

train_stats = trainer.train()
