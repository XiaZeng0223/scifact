python ./verisci/training/label_prediction/transformer_fever.py \
--model prajjwal1/bert-tiny \
--dest ./model/test \
--train ./data/fever/fever_dev.jsonl \
--dev ./data/fever/fever_train.jsonl \
--epoch 1 \
--model_base bert
