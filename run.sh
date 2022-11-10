#!/usr/bin/python
export PYTHONPATH="$PWD/"
pip install -r requirements.txt
pip install clearml

echo "Preparing Training Data"
python3 QCPG/evaluate.py \
--train_file data/mscoco/train.csv.gz \
--dataset_split train \
--predictions_column source \
--references_column target \
--metric metrics/para_metric \
--output_path new_data/mscoco/train.csv.gz

echo "Preparing Validation Data"
python3 QCPG/evaluate.py \
--train_file data/mscoco/validation.csv.gz \
--dataset_split train \
--predictions_column source \
--references_column target \
--metric metrics/para_metric \
--output_path new_data/mscoco/validation.csv.gz

echo "Preparing Testing Data"
python3 QCPG/evaluate.py \
--train_file data/mscoco/test.csv.gz \
--dataset_split train \
--predictions_column source \
--references_column target \
--metric metrics/para_metric \
--output_path new_data/mscoco/test.csv.gz

echo "Training"
python3 QCPG/train.py --model_name_or_path t5-base \
--do_train --do_eval --source_column reference \
--target_column prediction --per_device_eval_batch_size 16 \
--per_device_train_batch_size 16 --predict_with_generate \
--evaluation_strategy epoch --num_train_epochs 6 \
--lr_scheduler_type constant --save_total_limit 1 \
--dataset_generate_mode force_redownload --dataset_keep_in_memory \
--conditions_columns '["semantic_sim", "lexical_div", "syntactic_div", "phonological_div", "morphological_div"]' \
--overwrite_output_dir \
--dataset_map 'semantic_sim = 5 * round(bleurt_score * 100 / 5); lexical_div = 5 * round(set_diversity * 100 / 5); syntactic_div = 5 * round(syn_diversity * 100 / 5); phonological_div = 5 * round(phon_diversity * 100 / 5); morphological_div = 5 * round(morph_diversity * 100 / 5);' \
--train_file new_data/mscoco/train.csv.gz \
--validation_file new_data/mscoco/validation.csv.gz \
--learning_rate 1e-3 \
--output_dir new_data/t5-base-cond-mscoco-bleurt-lr1e-3-v1 \
--dataset_generate_mode force_redownload

echo "Prediction"
python3 QCPG/predict.py \
--per_device_eval_batch_size 256 --per_device_train_batch_size 256 \
--source_column reference --target_column prediction \
--conditions_columns '["semantic_sim", "lexical_div", "syntactic_div", "phonological_div", "morphological_div"]' \
--dataset_map 'semantic_sim = 5 * round(bleurt_score * 100 / 5); lexical_div = 5 * round(set_diversity * 100 / 5); syntactic_div = 5 * round(syn_diversity * 100 / 5); phonological_div = 5 * round(phon_diversity * 100 / 5); morphological_div = 5 * round(morph_diversity * 100 / 5);' \
--train_file new_data/mscoco/validation.csv.gz \
--dataset_split train \
--model_name_or_path new_data/t5-base-cond-mscoco-bleurt-lr1e-3-v1 \
--output_dir new_data/validation/t5-base-cond-mscoco-bleurt-lr1e-3-v1
