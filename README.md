# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija `
Authors: [**`Miha Lazić`**](https://github.com/Lazzo23), [**`Luka Gulič`**](https://github.com/GulicLuka), [**`Matevž Crček`**](https://github.com/Matslo1234)

## About
Traffic reporting is a crucial component of public broadcasting, providing real-time updates on road conditions, accidents, and congestion. At RTV Slovenija, traffic news is currently compiled manually by students who review traffic reports from the [_promet.si_](https://www.promet.si/sl) portal and transcribe them into structured news segments, which are then broadcast every 30 minutes. This manual approach is time-consuming, prone to human error, and inefficient given the volume of traffic data that needs to be processed. 

With the advancement of large language models (LLMs), there is an opportunity to automate the generation of structured and readable traffic news. This project aims to leverage an existing LLM, apply prompt engineering techniques and fine-tune the model to automatically generate traffic reports that adhere to the broadcasting standards of RTV Slovenija. In the next chapter, we present various existing solutions that we consider using in this project.

## Project Structure
**`/Code`** contains project source code.  
**`/Data`** contains project raw data (used for fine-tuning, evaluation, ...).  
**`/Images`** is output folder for project source code.  
**`/Instructions`** contains given project instructions and traffic report rules.  
**`/Report`** contains temporary project report written in LaTeX.   
**`/Models`** is output folder for trained models. They are too big to be included in the repository, so you will need to get them from drive, see section [Generating Text](#generating-text).

## Required Python Packages
The required python packages are listed in the [requirements.txt](Code%2Frequirements.txt) file. You can install them using pip:

```pip install -r requirements.txt```

## Generating the Datasets
To generate the datasets, we use the [generate_datasets.ipynb](Code%2Fgenerate_datasets.ipynb) notebook. This script processes the raw traffic data from the `Data` folder and generates structured datasets.
Running the notebook generates following files (they are also available on OneDrive):
- [train_dataset.jsonl](https://unilj-my.sharepoint.com/:u:/g/personal/mc6460_student_uni-lj_si/EWqrfb_Hu15OqeVNs2mw7yEBb-axmBkSEHd7cwzetP_Cug?e=FPLqqW)
- [valid_dataset.jsonl](https://unilj-my.sharepoint.com/:u:/g/personal/mc6460_student_uni-lj_si/EdfU2XlWuuREibTrJu2sgegB8m-Q49zMI80_HbvJeXVn_w?e=a2Ok8K)
- [test_dataset.jsonl](https://unilj-my.sharepoint.com/:u:/g/personal/mc6460_student_uni-lj_si/EdfkonJ2e6NCu85s0dkP5FYBzgnglji7ttw-G12YcaBM1w?e=KBkeH1)
- [train_dataset_generated_reports.jsonl](https://unilj-my.sharepoint.com/:u:/g/personal/mc6460_student_uni-lj_si/EfsLnoVi-w5GhtMXnwomKRQB1VP61ZAVi0mHi6L7MBdXmw?e=5RLysx)

## Training the Model
To train the model use [train.py](Code%2Ftrain.py). It takes the following arguments:
- **model_name**: Name of the base model to train. Default is `cjvt/GaMS-9B-Instruct` which we used in our project.
- **output_dir**: Directory to save the trained model. Required.
- **dataset**: Path to the training dataset in JSONL format. Required. Use the train datasets generated in the previous step.
- **max_steps**: Maximum number of training steps. Required.

Example command to train the model:

```python train.py --output_dir ./models/original_reports --dataset train_dataset.jsonl --max_steps 50000```

```python train.py --output_dir ./models/generated_reports --dataset train_dataset_generated_reports.jsonl --max_steps 50000```

## Generating Text
To generate text using the model use [eval.py](Code%2Feval.py). It takes the following arguments:
- **model_name**: Name of the base model to train. Default is `cjvt/GaMS-9B-Instruct` which we used in our project.
- **trained_model_path**: Path to the trained model. To evaluate only base model, set to "None". The path must be a folder which contains `adapter_config.json` and `adapter_model.safetensors`.
- **dataset**: Path to the training dataset in JSONL format. Required. Use the test dataset generated in the previous step.
- **output_path**: Path to file where evaluation results will be saved. Default is `eval_results.csv`.

Trained model for original reports is available on [OneDrive](https://unilj-my.sharepoint.com/:u:/g/personal/mc6460_student_uni-lj_si/Ecn0hprHjCxPi96hwJWAFfkBoOeJlw4tSBkBCxeqo6Uj0g?e=ZfWsax).

Trained model for gemini generated reports is available on [OneDrive](https://unilj-my.sharepoint.com/:u:/g/personal/mc6460_student_uni-lj_si/Ee568ZQ-nQZKtnfIcmPHWWcBnWSqDd-wpiibAgm1WSmRhw?e=xJ3SlX).

Extract the files to `Models/original_reports` and `Models/generated_reports` folders respectively.

[eval.py](Code%2Feval.py) will print the prompt currently being evaluated and the generated text. It will also save the results to the specified output path.
Example command to evaluate the model:

```python eval.py --trained_model_path ./Models/original_reports --dataset test_dataset.jsonl --output_path eval_results_original.csv```

```python eval.py --trained_model_path ./Models/generated_reports --dataset test_dataset.jsonl --output_path eval_results_generated.csv```

```python eval.py --trained_model_path "None" --dataset test_dataset.jsonl --output_path eval_results_base_gams.csv```

## Evaluation
Evaluation in done in [eval_bert_score.ipynb](Code%2Feval_bert_score.ipynb) notebook. It will calculate the BERTScore for each of the generated reports and print them out.

The notebook assumes that you have all of the results from the previous step in the `Code` folder. If you are missing any of the files, you can comment out the code that loads them and run the notebook again.

