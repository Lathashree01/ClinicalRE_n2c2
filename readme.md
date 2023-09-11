# Unified Framework for Clinical Relation Extraction

## Aim
This package is developed for researchers to easily use state-of-the-art transformer models for extracting relations from clinical notes. 
No prior knowledge of transformers is required. We handle the whole process, from data preprocessing to training to prediction.

## Dependency
The package is built on top of the Transformers developed by the HuggingFace. 
We have the requirement.txt to specify the packages required to run the project.

## Background
Original repository: https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction

## Further development
This repository is attributed to - https://github.com/Lathashree01/LlamaClinicalRE.
In this project, we perform domain adaptive pretraining of LLAMA models to the clinical domain. The clinical language understanding is evaluated based on evaluation datasets.
This repository is used to evaluate on n2c2 2018 dataset.

## Base models
- LLaMA 1
- LLaMA 2
- Our clinical LLaMA models


## Usage and example
- prerequisite
> The package is only for relation extraction, thus, the entities must be provided. 
> You have to conduct NER first to get all entities, then run this package to get the end-to-end relation extraction results

- data format
> see sample_data dir (train.tsv and test.tsv) for the train and test data format

> The sample data is a small subset of the data prepared from the 2018 umass made1.0 challenge corpus

```
# data format: tsv file with 8 columns:
1. relation_type: adverse
2. sentence_1: ALLERGIES : [s1] Penicillin [e1] .
3. sentence_2: [s2] ALLERGIES [e2] : Penicillin .
4. entity_type_1: Drug
5. entity_type_2: ADE
6. entity_id_1: T1
7. entity_id2: T2
8. file_id: 13_10

note: 
1) the entity between [s1][e1] is the first entity in a relation; the second entity in the relation is inbetween [s2][e2]
2) even the two entities in the same sentenc, we still require to put them separately
3) in the test.tsv, you can set all labels to neg or no_relation or whatever, because we will not use the label anyway
4) We recommend to evaluate the test performance in a separate process based on prediction. (see **post-processing**)
5) We recommend using official evaluation scripts to do evaluation to make sure the results reported are reliable.
```

- preprocess data (see the preprocess.ipynb script for more details on usage)
> We did not provide a script for training and test data generation

> We have a jupyter notebook with preprocessing 2018 n2c2 data as an example

> You can follow our example to generate your own dataset

- special tags
> We use 4 special tags to identify two entities in a relation
```
# The default tags we defined in the repo are

EN1_START = "[s1]"
EN1_END = "[e1]"
EN2_START = "[s2]"
EN2_END = "[e2]"

If you need to customize these tags, you can change them in
config.py
```

- training
> Please refer to the original page for all details of the parameters
> [flag details](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction/wiki/all-parameters)

sh run_training.sh

- prediction
sh run_testing.sh

- post-processing (we only support transformation to brat format)
```shell script
data_dir=./sample_data
pof=./predictions.txt

python src/data_processing/post_processing.py \
		--mode mul \
		--predict_result_file $pof \
		--entity_data_dir ./test_data_entity_only \
		--test_data_file ${data_dir}/test.tsv \
		--brat_result_output_dir ./brat_output
```
- Running evaluation script (n2c2 2018 challenge)
```shell script
python src/brat_eval.py --f1 /path_to_test_files_brat/ \
		--f2 path_to_brat_output -v
```
f1 -> Folder path to Gold standard 
f2 -> Folder path to Model predicted brat files
  

## Acknowledgements

This project is mainly developed based on the below open-source repository.
- https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction

## Issues
Please raise a GitHub issue if you have a problem or check the original repository.


