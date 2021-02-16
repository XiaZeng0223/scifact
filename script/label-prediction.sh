#!/bin/bash
#
# This script recreates the label-prediction results on table 3 of the paper
#
# Usage: bash script/label-prediction.sh [bert-variant] [training-dataset] [dataset]
# [bert-variant] options: "roberta_large", "roberta_base", "scibert", "biomed_roberta_base"
# [training-dataset] options: "scifact", "fever_scifact", "fever", "snopes"
# [dataset] options: "dev", "test"

bert_variant=$1
training_dataset=$2
dataset=$3
model_path=$4
output_path=$5

echo "${model_path}"

echo "${output_path}"

echo "Running pipeline on ${dataset} set."

####################

# Create a prediction folder to store results.
rm -rf "${output_path}"
mkdir -p "${output_path}"

####################

# Run abstract retrieval.
echo; echo "Retrieving oracle abstracts."
python3 verisci/inference/abstract_retrieval/oracle.py \
    --dataset data/claims_${dataset}.jsonl \
    --include-nei \
    --output ${output_path}/abstract_retrieval.jsonl

####################

# Run rationale selection.
echo; echo "Selecting oracle rationales."
python3 verisci/inference/rationale_selection/oracle_tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --abstract-retrieval ${output_path}/abstract_retrieval.jsonl \
    --output ${output_path}/rationale_selection.jsonl

####################

# Run label prediction
echo; echo "Predicting labels."

if [ $training_dataset = "scifact_only_claim" ]
then
    mode="only_claim"
elif [ $training_dataset = "scifact_only_rationale" ]
then
    mode="only_rationale"
else
    mode="claim_and_rationale"
fi

python3 verisci/inference/label_prediction/two-step.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --rationale-selection ${output_path}/rationale_selection.jsonl \
    --model ${model_path} \
    --mode ${mode} \
    --output ${output_path}/label_prediction_${bert_variant}.jsonl

####################

echo; echo "Evaluating."
python3 verisci/evaluate/label_prediction.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --label-prediction ${output_path}/label_prediction_${bert_variant}.jsonl \
    --output ${output_path}/wrong_preds_${bert_variant}.jsonl



