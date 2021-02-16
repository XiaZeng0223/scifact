import argparse
import torch
import jsonlines
import random
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='./data//fever/fever_train.jsonl')
parser.add_argument('--dev', type=str, default='./data//fever/fever_dev.jsonl')
parser.add_argument('--model', type=str, default='roberta-large')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')


class FeverLabelPredictionDataset(Dataset):
    def __init__(self, file):
        self.samples = []
        labels = {'SUPPORTS': 2, 'NOT ENOUGH INFO': 1, 'REFUTES': 0}
        for data in jsonlines.open(file):
            if data['label'] == 'NOT ENOUGH INFO':
                if data['sentences']:
                    indices = sorted(random.sample(range(len(data['sentences'])), k=1))
                    sentences = [data['sentences'][i] for i in indices]
                    self.samples.append({
                        'claim': data['claim'],
                        'rationale': ' '.join(sentences),
                        'label': labels['NOT ENOUGH INFO']
                    })
            else:
                for evidence_set in data['evidence_sets']:
                    self.samples.append({
                        'claim': data['claim'],
                        'rationale': ' '.join([data['sentences'][i] for i in evidence_set]),
                        'label': labels[data['label']]
                    })
                # Add negative samples
                non_evidence_indices = set(range(len(data['sentences']))) - set(
                    s for es in data['evidence_sets'] for s in es)
                if non_evidence_indices:
                    non_evidence_indices = random.sample(non_evidence_indices,
                                                         k=random.randint(1, min(1, len(non_evidence_indices))))
                    sentences = [data['sentences'][i] for i in non_evidence_indices]
                    self.samples.append({
                        'claim': data['claim'],
                        'rationale': ' '.join(sentences),
                        'label': labels['NOT ENOUGH INFO']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


trainset = FeverLabelPredictionDataset(args.train)
devset = FeverLabelPredictionDataset(args.dev)
print('Dataset loaded.')
tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
print('Model loaded.')


def encode(claims: List[str], rationale: List[str]):
    encoded_dict = tokenizer.batch_encode_plus(
        zip(rationale, claims),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            zip(rationale, claims),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def evaluate(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset):
            encoded_dict = encode(batch['claim'], batch['rationale'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return targets, outputs


print('Start evaluating')


targets,outputs = evaluate(model, devset)

print(f'Accuracy           {round(sum([outputs[i] == targets[i] for i in range(len(outputs))]) / len(outputs), 4)}')
print(f'Macro F1:          {f1_score(targets, outputs, average="macro").round(4)}')
print(f'Macro F1 w/o NEI:  {f1_score(targets, outputs, average="macro", labels=[0, 2]).round(4)}')
print()
print('                   [C      N      S     ]')
print(f'F1:                {f1_score(targets, outputs, average=None).round(4)}')
print(f'Precision:         {precision_score(targets, outputs, average=None).round(4)}')
print(f'Recall:            {recall_score(targets, outputs, average=None).round(4)}')
print()
print('Confusion Matrix:')
print(confusion_matrix(targets, outputs))
