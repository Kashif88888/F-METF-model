# F-METF-model

>Muhammad Kashif Jabbar; Huang Jianjun; Ayesha Jabbar; Zaka Ur Rehman; Khalil ur Rehman. Federated Multi-Expert Temporal Fusion for Privacy-Preserving Disease Onset Prediction in Multi-Institutional Health
Networks

## Requirements

```
recbole==1.0.1
python==3.8.13
cudatoolkit==11.3.1
pytorch==1.11.0
```

## Dataset

You should download the We have explicitly listed the exact sources of each dataset used in this work, with direct hyperlinks for reproducibility:
ECG Dataset: https://www.kaggle.com/datasets/mehranrezvani/electrocardiogram-ecg
EEG Dataset: https://www.kaggle.com/datasets/fabriciotorquato/eeg-data-from-hands-movement
PPG Dataset: https://www.kaggle.com/datasets/ucimachinelearning/photoplethysmography-ppg-dataset

## Quick Start

### Data Preparation

Preparing data:

```bash
python process_ECG_Dataset.py, EEG_Dataset.py, PPG_Dataset --output_path your_dataset_path
```


### Federated pre-train

```bash
python FMETF.py
```
Before train, you need to modify the relevant configuration in the configuration files `props/F-METF-model.yaml` and `props/pretrain.yaml`. 

Here are some important parameters in `props/pretrain.yaml` you may need to modify:

1.`data_path`: The path of the dataset you want to use for pre-training.

2.`cluster_centroids`: The number of cluster centroids for clustering in the server.

3.`cluster_iters`: Maximum number of rounds of clustering, beyond which clustering will be paused if not yet completed.

### Fine-tuning after federated pre-training
Finetune pre-trained recommender of "Pantry":

```bash
python finetune.py --d=Pantry --p=your_pretrained_model.pth
```
You can adjust the corresponding parameters for fine-tuning in  `props/finetune.yaml`.
