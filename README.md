# Customer-Support-Ticket-Auto-Triage

An end-to-end Machine Learning application that automatically classifies support tickets based on subject + description text.

The project includes model training, evaluation, a REST API, and a browser-based frontend for real-time predictions.

## Features

Text-based ticket classification

TF-IDF + Logistic Regression model

RESTful API using Flask

Simple frontend UI for testing

Model persistence with joblib

Real-time inference with latency measurement

## Dataset

IT Service Ticket Classification Dataset

Document → Subject + Description (input)

Topic_group → Ticket Category (target)

## Model Performance

Accuracy: ~85%

Precision (macro): ~0.83

Recall (macro): ~0.87

F1-score (macro): ~0.85

Avg Latency: ~0.00006 seconds

## Tech Stack

Python

scikit-learn

Flask

HTML / CSS / JavaScript

## Usage

Frontend UI

http://127.0.0.1:5000/ui

## API Endpoint (Sample Input)

POST /predict

{
  "subject": "Laptop not turning on
  
  "description": "System does not boot after update"
}

## Sample Output
{
  "predicted_category": "Hardware",
  
  "latency_seconds": 0.0019
}
