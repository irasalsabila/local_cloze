from unsloth import FastLanguageModel
import argparse
import torch


class Args:
    def __init__(self):
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument("--train_set", default="id", help="train set to choose")
        args_parser.add_argument('--test_language', default='jv', help='test language')
        args_parser.add_argument('--model_name', default='sahabat', help='model name')
        args_parser.add_argument('--seed', type=int, default=3407, help='random seed')

        # Parse arguments
        self.args = args_parser.parse_args()

# Set arguments for training or evaluation
args = Args().args

if args.model_name == 'cendol':
    model_name = "indonlp/cendol-llama2-7b-chat"
elif args.model_name == 'komodo':
    model_name = "Yellow-AI-NLP/komodo-7b-base"
elif args.model_name == 'sahabat':
    model_name = 'GoToCompany/gemma2-9b-cpt-sahabatai-v1-base'
else:
    assert False, "Model name not found"

from unsloth import FastLanguageModel
train_set = args.train_set
test_language = args.test_language
model_path = f"outputs/{model_name.split('/')[-1]}/{train_set}"
import os
import re
checkpoint = max(os.listdir(model_path), key=lambda x: int(re.search(r'\d+', x).group()))
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{model_path}/{checkpoint}",
    max_seq_length = 8192,
    load_in_4bit = True,
    # token = "hf_..."
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

import pandas as pd
data_path = f"../../dataset/test/test_{test_language}.csv"
def load_test_data(data_path):
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
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Generate a single sentence ending for the given story context.

    ### Input:
    {}

    ### Response:
    """
        contexts  = examples["context"]
        endings = examples["ending"]
        texts = []
        for context, ending in zip(contexts, endings):
            # chat = [
            #     {"role": "user", "content": f"Generate the ending for the following given story context\nStory Context: {context}"},
            # ]
            # text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            text = alpaca_prompt.format(context)
            texts.append(text)
        return { "text" : texts, }
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    return dataset
dataset = load_test_data(data_path)
contexts = dataset['text']
correct_endings = dataset['ending']

predicted_endings = []
from tqdm.auto import tqdm
for context in tqdm(contexts):
    inputs = tokenizer(context,return_tensors = "pt").to("cuda")
    res = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=128, use_cache=True, do_sample=False)
    predicted_ending = tokenizer.decode(res[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    predicted_endings.append(predicted_ending)

from evaluate import load
rouge = load('rouge')
rouge_score = rouge.compute(predictions=predicted_endings, references=correct_endings)

bertscore = load("bertscore")
results = bertscore.compute(predictions=predicted_endings, references=correct_endings, lang=test_language)
def get_mean(l):
    return sum(l) / len(l)
bert_score = {
    "precision": get_mean(results['precision']),
    "recall": get_mean(results['recall']),
    "f1": get_mean(results['f1'])
}

references = [[x] for x in correct_endings]
bleu = load("bleu")
bleu_score = bleu.compute(predictions=predicted_endings, references=references)

meteor = load('meteor')
meteor_score = meteor.compute(predictions=predicted_endings, references=correct_endings)

#write scores to an external file
with open(f"result_{test_language}/{model_name.split('/')[-1]}_{train_set}.txt", "w") as f:
    f.write(f"ROUGE: {rouge_score}\n")
    f.write(f"BERTScore: {bert_score}\n")
    f.write(f"BLEU: {bleu_score}\n")
    f.write(f"METEOR: {meteor_score}\n")