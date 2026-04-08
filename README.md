video link https://youtu.be/x2gGw9bf-K0

Handwritten Character Recognition ML Pipeline
Overview

This project implements an end-to-end Machine Learning pipeline for recognizing handwritten English characters and digits. The system allows users to upload images of handwritten characters and receive predictions from a trained deep learning model through a FastAPI web application.

The pipeline includes:

Model inference
Batch predictions
Prediction logging
Model performance monitoring
Retraining functionality
A metrics dashboard

The goal of this project is to demonstrate the deployment and monitoring of a machine learning model in a production-like environment.

Project Features
Image Prediction

Users can upload an image of a handwritten character and the system will:

Preprocess the image
Run it through the trained CNN model
Return the predicted character
Display prediction confidence
Batch Prediction

Users can upload multiple images at once for prediction.

The system processes each image and returns:

Predicted character
Prediction confidence
Inference time

All predictions are stored in the database.

Prediction Logging

Each prediction is saved in the database with the following information:

Image name
Predicted character
True label (if provided)
Confidence score
Inference time
Timestamp

This allows monitoring of model performance over time.

Model Metrics Endpoint

The API provides a metrics endpoint that tracks model performance.

Metrics include:

Total predictions
Average inference time
Model accuracy
Precision
Recall
F1 score
API uptime

These metrics are calculated using stored predictions in the database.

Metrics Dashboard

A web dashboard allows users to view model performance metrics without manually querying the API.

The dashboard displays:

Total predictions
Average inference time
Model accuracy
Precision
Recall
F1 score
System uptime
Model Retraining

The system supports model retraining using newly uploaded data.

Retraining includes:

Uploading new labeled data
Preprocessing the data
Fine-tuning the pretrained model
Saving the updated model
Loading the retrained model into the API

This allows the system to continuously improve with new data.

Technology Stack

The following technologies were used to build the system.

Backend

Python
FastAPI
Uvicorn

Machine Learning

TensorFlow / Keras
NumPy
OpenCV

Database

MySQL
SQLAlchemy

Frontend

HTML
JavaScript
Bootstrap

Deployment

Render
Project Structure
ML_Pipeline_Summative
│
├── api.py
├── requirements.txt
├── README.md
│
├── models
│   ├── final_alphanumeric_model.h5
│   ├── class_names.npy
│   └── model_metadata.json
│
├── src
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── model.py
│   └── retrain.py
│
├── static
│   ├── index.html
│   ├── dashboard.html
│   └── retrain.html
│
├── uploads
├── data
│   └── retrain
│
├── database.py
└── database_model.py
Installation
Clone the Repository
git clone https://github.com/yourusername/handwritten-character-recognition.git
cd handwritten-character-recognition
Create Virtual Environment
python -m venv rl-env

Activate the environment:

Windows

rl-env\Scripts\activate

Mac/Linux

source rl-env/bin/activate
Install Dependencies
pip install -r requirements.txt


Running the Application


Start the FastAPI server:

uvicorn api:app --reload

The API will run at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs


API Endpoints
Predict Single Image
POST /predict

Uploads a single image and returns the predicted character.

Batch Prediction
POST /batch-predict

Uploads multiple images and returns predictions for each.

Metrics Endpoint
GET /metrics

Returns model performance statistics.

Example response:

{
  "total_predictions": 9,
  "average_inference_time_ms": 144.78,
  "uptime_seconds": 322,
  "model_accuracy": 1.0,
  "model_precision": 1.0,
  "model_recall": 1.0,
  "model_f1_score": 1.0
}
Retrain Model
POST /retrain

Retrains the model using new labeled data.

Parameters include:

number of epochs
learning rate
Dataset

The model was trained on a dataset containing handwritten English characters and digits.

Images were organized into folders representing each character class.

Example structure:

dataset
│
├── train
│   ├── A
│   ├── B
│   ├── C
│   └── ...
│
└── test
Deployment

The application is deployed on Render.

Deployment steps:

Push the project to GitHub
Create a Web Service on Render
Set build command:
pip install -r requirements.txt
Set start command:
uvicorn api:app --host 0.0.0.0 --port $PORT
Future Improvements

Possible improvements include:

Improved UI for prediction results
Automated model retraining pipeline
Support for more handwriting styles
Advanced monitoring and logging
Cloud storage for uploaded images
Author

Florence Kabeya
African Leadership University
Machine Learning Pipeline Project