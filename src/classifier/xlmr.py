import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class XLMRData():
    def __init__(self, args):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_type, do_lower_case=True)
        self.sep_token = '</s>'
        self.cls_token = '<s>'
        self.pad_token = '<pad>'
        self.sep_vid = self.tokenizer.get_vocab()[self.sep_token]
        self.cls_vid = self.tokenizer.get_vocab()[self.cls_token]
        self.pad_vid = self.tokenizer.get_vocab()[self.pad_token]
        self.MAX_TOKEN_CHAT = args.max_token_chat
        self.MAX_TOKEN_RESP = args.max_token_resp

    def preprocess_one(self, chat, resp, label):
        chat_subtokens = [self.cls_token] + self.tokenizer.tokenize(chat) + [self.sep_token]        
        chat_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(chat_subtokens)
        if len(chat_subtoken_idxs) > self.MAX_TOKEN_CHAT:
            chat_subtoken_idxs = chat_subtoken_idxs[len(chat_subtoken_idxs)-self.MAX_TOKEN_CHAT:]
            chat_subtoken_idxs[0] = self.cls_vid

        resp_subtokens = self.tokenizer.tokenize(resp) + [self.sep_token]
        resp_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(resp_subtokens)
        if len(resp_subtoken_idxs) > self.MAX_TOKEN_RESP:
            resp_subtoken_idxs = resp_subtoken_idxs[:self.MAX_TOKEN_RESP]
            resp_subtoken_idxs[-1] = self.sep_vid

        src_subtoken_idxs = chat_subtoken_idxs + resp_subtoken_idxs
        segments_ids = [0] * len(chat_subtoken_idxs) + [1] * len(resp_subtoken_idxs)
        assert len(src_subtoken_idxs) == len(segments_ids)
        return src_subtoken_idxs, segments_ids, label
    
    def preprocess(self, chats, resps, labels):
        assert len(chats) == len(resps) == len(labels)
        output = []
        for idx in range(len(chats)):
            output.append(self.preprocess_one(chats[idx], resps[idx], labels[idx]))
        return output
    
class Batch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    # do padding here
    def __init__(self, data, idx, batch_size, device):
        PAD_ID=0
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        seg = torch.tensor(self._pad([x[1] for x in cur_batch], PAD_ID))
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src != PAD_ID)
        
        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src
    
class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.bert = XLMRobertaModel.from_pretrained(args.model_type)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction='none') 

    def forward(self, src, seg, mask_src):
        batch_size = src.shape[0]
        top_vec = self.bert(input_ids=src, attention_mask=mask_src)[0]  # Update for XLM-Roberta
        clss = top_vec[:,0,:]
        final_rep = self.dropout(clss)
        conclusion = self.linear(final_rep).squeeze()
        return self.sigmoid(conclusion)
    
    def get_loss(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        return self.loss(output, label.float())

    def predict(self, src, seg, mask_src, label):
        output = self.forward(src, seg, mask_src)
        batch_size = output.shape[0]
        assert batch_size%2 == 0
        output = output.view(int(batch_size/2), 2)
        prediction = torch.argmax(output, dim=-1).data.cpu().numpy().tolist()
        answer = label.view(int(batch_size/2), 2)
        answer = torch.argmax(answer, dim=-1).data.cpu().numpy().tolist()
        return answer, prediction
    
def prediction(dataset, model, args):
    preds = []
    golds = []
    model.eval()
    assert len(dataset)%2==0
    assert args.batch_size%2==0
    for j in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src = Batch(dataset, j, args.batch_size, args.device).get()
        answer, prediction = model.predict(src, seg, mask_src, label)
        golds += answer
        preds += prediction
    return accuracy_score(golds, preds), preds

def read_data(fname, num_sent=4, test_language=None):
    contexts = []
    endings = []
    labels = []

    # Read the data
    data = pd.read_csv(fname)

    # Apply filtering based on test_language
    if test_language and test_language.endswith("_mt"):
        data = data[data[["topic", "category"]].isnull().all(axis=1)]
    elif test_language and test_language.endswith("_syn"):
        data = data[data[["topic", "category"]].notnull().all(axis=1)]

    # Process each row
    for idx, row in data.iterrows():
        sents = []
        for i in range(num_sent, 0, -1):
            if len(sents) == num_sent:
                break
            sents.insert(0, row.get(f"sentence_{i}", ""))  # Safely get sentences
        
        # Combine sentences into a single context
        context = " ".join(sents)
        ending1 = row["correct_ending"]
        ending2 = row["incorrect_ending"]

        # Add correct and incorrect endings
        contexts.append(context)
        endings.append(ending1)
        labels.append(1)

        contexts.append(context)
        endings.append(ending2)
        labels.append(0)

    return contexts, endings, labels


def train(args, train_dataset, test_dataset, model):
    train_data = train_dataset
    _, dev_data = train_test_split(test_dataset, test_size=0.2, random_state=args.seed)
    if len(dev_data) %2 == 1:
        dev_data = dev_data[:-1]


    no_decay = ["bias", "LayerNorm.weight"]
    t_total = len(train_data) // args.batch_size * args.num_train_epochs
    args.warmup_steps = int(0.1 * t_total)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warming up = %d", args.warmup_steps)
    logger.info("  Patience  = %d", args.patience)

    set_seed(args)
    tr_loss = 0.0
    global_step = 1
    best_loss = float('inf')  # Use loss for early stopping
    best_acc_test = 0
    best_acc_dev = 0
    cur_patience = 0
    for i in range(int(args.num_train_epochs)):
        random.shuffle(train_data)
        epoch_loss = 0.0
        for j in range(0, len(train_data), args.batch_size):
            src, seg, label, mask_src = Batch(train_data, j, args.batch_size, args.device).get()
            model.train()
            loss = model.get_loss(src, seg, label, mask_src)
            loss = loss.sum()/args.batch_size
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

        avg_epoch_loss = epoch_loss / global_step
        logger.info("Finish epoch = %s, loss_epoch = %s", i + 1, avg_epoch_loss)

        dev_acc, dev_pred = prediction(dev_data, model, args)

        # Check for early stopping based on loss

        if dev_acc > best_acc_dev:
            best_acc_dev = dev_acc
            test_acc, test_pred = prediction(test_dataset, model, args)
            # print(test_pred)
            best_acc_test = test_acc
            cur_patience = 0
            logger.info("Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)

            # Check and log information about loss improvement
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                logger.info("Better loss, BEST loss = %s & BEST Acc in test = %s.", best_loss, best_acc_test)

        else:
            cur_patience += 1
            if cur_patience == args.patience:
                logger.info("Early Stopping, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)
                break
            else:
                logger.info("Not Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)

    return global_step, tr_loss / global_step, best_loss, best_acc_test, test_pred, dev_pred, best_acc_dev

def get_unique_filename(base_filename):
    counter = 1
    filename = base_filename
    base, ext = os.path.splitext(base_filename)  # Split filename and extension
    while os.path.exists(filename):
        filename = f"{base}_{counter}{ext}"  # Append counter before the extension
        counter += 1
    return filename

class Args:
    def __init__(self):
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument('--train_path', default='../../dataset/train/', help='path to train set')
        args_parser.add_argument("--train_set", default="gpt4o", help="train set to choose")
        args_parser.add_argument('--test_language', default='jv', help='test set language')
        args_parser.add_argument('--num_sent', type=int, default=4, help='number of sentence in context')
        args_parser.add_argument('--max_token_chat', type=int, default=450, help='max token chat for preprocessing')
        args_parser.add_argument('--max_token_resp', type=int, default=50, help='max token response for preprocessing')
        args_parser.add_argument('--batch_size', type=int, default=40, help='total batch size')
        args_parser.add_argument('--learning_rate', type=float, default=5e-6, help='learning rate')
        args_parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Adam optimizer epsilon')
        args_parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
        args_parser.add_argument('--num_train_epochs', type=int, default=20, help='number of training epochs')
        args_parser.add_argument('--warmup_steps', type=int, default=200, help='warmup steps')
        args_parser.add_argument('--logging_steps', type=int, default=200, help='logging steps')
        args_parser.add_argument('--seed', type=int, default=2020, help='random seed')
        args_parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
        args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
        args_parser.add_argument('--no_cuda', action='store_true', help='do not use CUDA')

        # Parse arguments
        self.args = args_parser.parse_args()

        # Set model type to XLM-Roberta
        self.args.model_type = 'xlm-roberta-base'

        # Set device (CUDA or CPU)
        if self.args.local_rank == -1 or self.args.no_cuda:
            self.args.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.args.device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.args.n_gpu = 1

# Set arguments for training or evaluation
args = Args().args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)

logger.info(f"Using model type: {args.model_type}")
set_seed(args)

xlmrdata = XLMRData(args)

scores = {}
# test_predictions = []
for num_sent in [4]:
    args.num_sent = num_sent
    trainset = read_data(f"{args.train_path}/{args.train_set}.csv", args.num_sent)

    trainset = list(zip(*trainset))
    trainset = list(map(list, trainset))  # Convert tuples to lists
    random.shuffle(trainset)
    trainset = list(zip(*trainset))

    print("Train set loaded")
    logger.info(f"Training dataset loaded for num_sent = {num_sent}")

    assert(args.test_language in ['su', 'su_mt', 'su_syn', 'jv', 'jv_mt', 'jv_syn', 'all'])

    if args.test_language in ['jv', 'jv_mt', 'jv_syn']:
        test_path = '../../dataset/test/test_jv.csv'
    elif args.test_language in ['su', 'su_mt', 'su_syn']:
        test_path = '../../dataset/test/test_su.csv'
    elif args.test_language == 'all':
        test_path = '../../dataset/test/test_all.csv'
    else:
        raise ValueError("Unsupported test language")

    original_test_df = pd.read_csv(test_path)

    if args.test_language.endswith('_mt'):
        original_test_df = original_test_df[original_test_df[['topic', 'category']].isnull().all(axis=1)]
    elif args.test_language.endswith('_syn'):
        original_test_df = original_test_df[original_test_df[['topic', 'category']].notnull().all(axis=1)]

    # Process the test data
    testset = read_data(test_path, args.num_sent, args.test_language)

    print("Test set loaded")
    logger.info(f"Test dataset loaded")

    train_dataset = xlmrdata.preprocess(trainset[0], trainset[1], trainset[2])
    test_dataset = xlmrdata.preprocess(testset[0], testset[1], testset[2])
    print("Data preprocessed")

    model = Model(args, args.device)
    model.to(args.device)

    global_step, tr_loss, best_loss, best_acc_test, test_pred, dev_pred, best_acc_dev = train(args, train_dataset, test_dataset, model)

    # test_predictions.extend(list(test_pred))

    print(f'Num Sentences: {num_sent}')
    print(f'Best loss: {best_loss}')
    print(f'Test set accuracy: {best_acc_test}')
    print(f'Dev set accuracy', {best_acc_dev})
    print('-------------------------------------------')

    # Save best accuracy and loss in the scores dictionary
    scores[num_sent] = {'best_acc_test': best_acc_test, 'best_acc_dev': best_acc_dev, 'best_loss': best_loss}

    # Add predictions to the original test dataset
    reconstructed_df = pd.DataFrame({
        "sentence_1": original_test_df["sentence_1"],
        "sentence_2": original_test_df["sentence_2"],
        "sentence_3": original_test_df["sentence_3"],
        "sentence_4": original_test_df["sentence_4"],
        "correct_ending": original_test_df["correct_ending"],
        "incorrect_ending": original_test_df["incorrect_ending"],
        "topic": original_test_df.get("topic", None),
        "category": original_test_df.get("category", None),
        "generated_by": original_test_df.get("generated_by", None),
        "language": original_test_df.get("language", None)
    })

    print("reconstructed_df:",len(reconstructed_df))
    print("test_pred:",len(test_pred))

    # Ensure the length matches
    if len(reconstructed_df) == len(test_pred):
        reconstructed_df["predictions"] = test_pred
    else:
        raise ValueError("Mismatch between test dataset rows and predictions.")

    # Save the reconstructed DataFrame only for num_sent == 4 and test_language == 'jv' or 'su'
    if num_sent == 4 and args.test_language in ['jv', 'su']:
        output_path = get_unique_filename(
            f"result_{args.test_language}/test_xlmr_{args.test_language}_in_{args.train_set}_with_preds.csv"
        )
        reconstructed_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

# Append results to a single file based on train_set
output_filename = f"result_{args.test_language}/xlmr_{args.train_set}_scores.txt"
with open(output_filename, 'a') as f:  # Open file in append mode
    for k, v in scores.items():
        f.write(f"xlmr_{args.train_set}_score {k} --> {v}\n")
print(f"Scores appended to {output_filename}")