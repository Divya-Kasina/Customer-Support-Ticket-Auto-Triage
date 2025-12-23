# Customer Support Ticket Auto-Triage 

## Overview

This project implements an **AI-powered Support Ticket Auto-Triage system** that automatically classifies incoming IT service/support tickets into predefined categories. 

The goal is to reduce manual effort, improve response time, and route tickets to the correct support teams efficiently.

The system uses **machine learning-based text classification** trained on historical IT service ticket data and exposes the model through a simple execution pipeline.

## Features

* Automatic classification of support tickets based on text description
  
* Trained ML model saved as a reusable checkpoint

* End-to-end pipeline: data loading → preprocessing → training → prediction
  
* Easy-to-run Python implementation
  
* Suitable for integration with customer support platforms

## Tech Stack

* **Programming Language:** Python
  
* **Machine Learning:** Scikit-learn
  
* **Data Processing:** Pandas, NumPy
  
* **Environment:** Virtual Environment

---

## Repository Structure

```
Customer-Support-Ticket-Auto-Triage/
│
├── IT_Service_Ticket_Classification.csv   # Dataset
├── ticket_classifier_it_dataset.py        # Main training & inference script
├── ticket_classifier.pkl                  # Trained model checkpoint
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Divya-Kasina/Customer-Support-Ticket-Auto-Triage.git
cd Customer-Support-Ticket-Auto-Triage
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model & Generate Checkpoint

```bash
python ticket_classifier_it_dataset.py
```

This will:

* Load and preprocess the dataset
* Train the ticket classification model
* Save the trained model as `ticket_classifier.pkl`

---

### Input

* Raw ticket text (string)

### Output

* Predicted ticket category (label)

---

## Dataset Information

* **File:** `IT_Service_Ticket_Classification.csv`
* **Description:** Historical IT service/support tickets
* **Fields:** Ticket text and corresponding category labels

The dataset is used for supervised learning to train the classification model.

---

## Model Details

* Text preprocessing includes cleaning and vectorization
* Supervised classification using Scikit-learn
* Model serialized using Pickle for reuse
  
<img width="967" height="832" alt="image" src="https://github.com/user-attachments/assets/a618f026-67eb-4e71-b8f8-b064dcd5efc1" />
