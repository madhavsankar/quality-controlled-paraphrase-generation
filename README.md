# Quality Controlled Paraphrase Generation ++
> Paraphrase generation has been widely used in various downstream tasks. Most tasks benefit mainly from high quality paraphrases, namely those that are semantically similar to, yet linguistically diverse from, the original sentence. Generating high-quality paraphrases is challenging as it becomes increasingly hard to preserve meaning as linguistic diversity increases. Recent works achieve nice results by controlling specific aspects of the paraphrase, such as its syntactic tree. However, they do not allow to directly control the quality of the generated paraphrase, and suffer from low flexibility and scalability. 

> Here we propose `QCPG++`, a quality-guided controlled paraphrase generation model that builds on top of [`QCPG`](https://github.com/IBM/quality-controlled-paraphrase-generation), that allows directly controlling the quality dimensions. We add more dimensions to the original model. Now we can control the morphological, phonological and syntactic diversity of the paraphrase while maximizing semantic similarity.

## Trained Models

[`qcpg++`](https://huggingface.co/madhavsankar/qcpg-mscoco-sbert-lr1e-4) (Trained on `data/mscoco`)

## Usage
Use run notebook or run shell script to train the model on mscoco dataset.

### Preprocessing
```python
python QCPG/evaluate.py \
--train_file data/mscoco/train.csv.gz \
--dataset_split train \
--predictions_column source \
--references_column target \
--metric metrics/para_metric \
--output_path new_data/mscoco/train.csv.gz
```

### Train
```python
!python QCPG/train.py --model_name_or_path t5-base \
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
```

### Inference
```python
!python QCPG/predict.py \
--per_device_eval_batch_size 256 --per_device_train_batch_size 256 \
--source_column reference --target_column prediction \
--conditions_columns '["semantic_sim", "lexical_div", "syntactic_div", "phonological_div", "morphological_div"]' \
--dataset_map 'semantic_sim = 0.9; lexical_div = 0.5; syntactic_div = 0.8; phonological_div = 0.5; morphological_div = 0.5;' \
--train_file new_data/mscoco/validation.csv.gz \
--dataset_split train \
--model_name_or_path new_data/t5-base-cond-mscoco-bleurt-lr1e-3-v1 \
--output_dir new_data/validation/t5-base-cond-mscoco-bleurt-lr1e-3-v1
```
    
## Citation
```
@inproceedings{bandel-etal-2022-quality,
    title = "Quality Controlled Paraphrase Generation",
    author = "Bandel, Elron  and
      Aharonov, Ranit  and
      Shmueli-Scheuer, Michal  and
      Shnayderman, Ilya  and
      Slonim, Noam  and
      Ein-Dor, Liat",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.45",
    pages = "596--609",
    abstract = "Paraphrase generation has been widely used in various downstream tasks. Most tasks benefit mainly from high quality paraphrases, namely those that are semantically similar to, yet linguistically diverse from, the original sentence. Generating high-quality paraphrases is challenging as it becomes increasingly hard to preserve meaning as linguistic diversity increases. Recent works achieve nice results by controlling specific aspects of the paraphrase, such as its syntactic tree. However, they do not allow to directly control the quality of the generated paraphrase, and suffer from low flexibility and scalability. Here we propose QCPG, a quality-guided controlled paraphrase generation model, that allows directly controlling the quality dimensions. Furthermore, we suggest a method that given a sentence, identifies points in the quality control space that are expected to yield optimal generated paraphrases. We show that our method is able to generate paraphrases which maintain the original meaning while achieving higher diversity than the uncontrolled baseline. The models, the code, and the data can be found in https://github.com/IBM/quality-controlled-paraphrase-generation.",
}
```