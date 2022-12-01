----------------
# Developed with

- Operating System: Ubuntu 20.04.4 LTS
	- Kernel: Linux 5.4.0-110-generic
	- Architecture: x86-64
- Python:
	- 3.9.7

----------------
# Prerequisites

## Installation of required libraries

You can install external libraries by running:

```pip install -r requirements.txt```

You can also create a dedicated environment using conda to avoid dependency issues:

```bash
# Create the environment.
conda create -n JNRF python=3.9.7

# Activate the environment.
conda activate JNRF

# Go to the project root folder.
cd <path to the project root folder>

# Install external libraries.
pip install -r requirements.txt

# Do stuff: See the "Typical pipeline" section.

# Deactivate the environment.
conda deactivate

# If you wish, you can remove the environment.
conda env remove --name JNRF
```

----------------
# Structure

```bash
├── configs
│   ├── inference_config.json (configuration required for inference)
│   ├── preprocessing_config.json (configuration required for preprocessing)
│   └── training_config.json (configuration required for training)
│
├── data
│   ├── loop_pred (folder required for evaluation during training)
│   ├── predictions (folder required to store prediction file before evaluation)
│   ├── test (folder with original raw test data .txt and .ann files)
│   └── train (folder with original raw train data .txt and .ann files)
│
├── runs (folder for tensorboard logs)
│
├── saved_models (folder to save best model during training)
│
├── scr (source code)
│   ├── dataloader_helper.py
│   ├── evaluation_script.py
│   ├── JNRF.py
│   ├── loss.py
│   ├── postprocessing_helper.py
│   └── preprocessing_helper.py
│
├── 1_preprocessing.py (used to preprocess raw data, see configs/preprocessing_config.json for configuration)
│
├── 2_training.py (used to train a JNRF model, see configs/training_config.json for configuration)
│
├── 3_inference.py (used to predict with JNRF a model, see configs/inference_config.json for configuration)
│
├── 4_evaluation.py (used to compute evaluation metrics on the model used in the inference step)
│
└── requirements.txt (used to install dependencies)
```

----------------
# Typical pipeline

## Access to data

Apply and download [data](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2018-t2/). Then, put orginial raw test data (.txt and .ann files) in "data/test", and orginial raw train data (.txt and .ann files) in "data/train".

## Download Spacy English pipeline

```python -m spacy download en_core_web_sm```

## Step 1: Preprocessing

see configs/preprocessing_config.json for configuration, then:

```python 1_preprocessing.py```

## Step 2: Model training

see configs/training_config.json for configuration. Make sure to change the name of the model to your convenience, then:

```python 2_training.py```

## Step 3: Model prediction

configs/inference_config.json for configuration. Make sure to select the correct model name, then:

```python 3_inference.py```

## Step 4: Evaluation

```python 4_evaluation.py```

----------------
