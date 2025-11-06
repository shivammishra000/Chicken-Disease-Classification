# Chicken-Disease-Classification

This project is a **Deep Learning-based pipeline** to classify chicken fecal images into two categories: **Healthy** and **Coccidiosis**. The pipeline includes **data ingestion, base model preparation, model training, evaluation**, and **real-time prediction via a Flask web application**.


## Workflow

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

### How to run?

STEPS:

Clone the repository

https://github.com/entbappy/Chicken-Disease-Classification--Project

STEP 01- Create a conda environment after opening the repository

conda create -n cdcl python=3.8 -y

conda activate cdcl

STEP 02- install the requirements

pip install -r requirements.txt

# Finally run the following command

python app.py

Now,

open up you local host and port
http://127.0.0.1:8080/

### DVC cmd
1. dvc init
2. dvc repro
3. dvc dag

##  Features

- Automated **data ingestion** from image folders.
- **Transfer learning** using a pre-trained CNN model.
- Supports **data augmentation** for better generalization.
- **Early stopping** and **model checkpointing** during training.
- Model evaluation on validation data with **loss and accuracy** metrics.
- Real-time **prediction API** with Flask and a web interface.

## Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **Flask** for REST API and web UI
- **DVC** for data and model versioning
- **Bootstrap 4** for front-end
- **NumPy & OpenCV** for image processing

## Author
Shivam Mishra

Github - https://github.com/shivammishra000/Chicken-Disease-Classification

LinkedIn - https://www.linkedin.com/in/shivam-mishra-38322b260/