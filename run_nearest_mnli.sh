#bash ./script/label-prediction.sh two-step scifact dev three-models two-step-output

#python verisci/inference/label_prediction/oracle.py \
#    --dataset data/claims_dev.jsonl \
#    --rationale-selection oracle_outputs/rationale_selection.jsonl \
#    --output oracle_outputs/label_prediction.jsonl

#python verisci/inference/label_prediction/sbert.py \
#    --corpus data/corpus.jsonl \
#    --dataset data/claims_dev.jsonl \
#    --rationale-selection oracle_outputs/rationale_selection.jsonl \
#    --output oracle_outputs/label_prediction_sbert.jsonl  \
#    --model ${model_path}

time python nearestneighbor_nli.py --model bert-base-nli-mean-tokens
#time python nearestneighbor_snli.py --model bert-base-nli-mean-tokens
#Traceback (most recent call last):
#  File "nearestneighbor_snli.py", line 50, in <module>
#    X_train = diff(trainset, model)
#  File "nearestneighbor_snli.py", line 24, in diff
#    diffs = claim_embeddings - evidence_embeddings
#numpy.core._exceptions.MemoryError: Unable to allocate 1.57 GiB for an array with shape (549367, 768) and data type floa
#t32
