# Symptoms Based Disease Prediction System

## Overview

The **Symptoms Based Disease Prediction System** is a Machine Learning web application that predicts possible diseases based on user-provided symptoms. The system uses a **Random Forest Classifier** trained on a symptoms dataset to identify diseases and provide additional information such as **disease descriptions and precautions**.

The application is built using **Python, Flask, and Scikit-Learn**, and the model is deployed as an API endpoint that can be tested using tools like **Postman**.

---

## Features

* Predicts diseases based on user symptoms
* Provides disease **description**
* Suggests **precautions** for the predicted disease
* Machine Learning model using **Random Forest**
* REST API built with **Flask**
* Easy API testing using **Postman**

---

## Project Structure

```
Symptoms-Disease-Prediction
│
├── Summary.py
├── README.md
├── requirements.txt
│
├── Symptoms_dataset.csv
├── symptom_Description.csv
├── symptom_precaution.csv
├── Symptom-severity.csv
│
└── .gitignore
```

---

## Technologies Used

* Python
* Flask
* NumPy
* Pandas
* Scikit-learn
* Pickle
* Postman (API Testing)

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

### 2. Navigate to Project Folder

```
cd Symptoms-Disease-Prediction
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask server:

```
python Summary.py
```

The server will start at:

```
http://127.0.0.1:5000
```

---

## API Endpoint

### Predict Disease

**Endpoint**

```
POST /predict
```

**Request Body (JSON)**

```
{
  "Symptoms": [
    "itching",
    "skin rash",
    "nodal skin eruptions"
  ]
}
```

**Response**

```
{
  "Disease": "fungalinfection",
  "Description": "A fungal infection is a disease caused by fungus.",
  "Precautions": [
    "bath twice",
    "use dettol",
    "keep infected area dry",
    "consult doctor"
  ]
}
```

---

## Model Training

The machine learning model is trained using:

* **Random Forest Classifier**
* Dataset containing **symptoms and diseases**
* Label Encoding for disease classification
* Train/Test split for model evaluation

The trained model and encoder are saved using **Pickle**.

---

## Datasets Used

* Symptoms Dataset
* Symptom Description Dataset
* Symptom Precaution Dataset
* Symptom Severity Dataset

These datasets help the system to:

* Predict diseases
* Provide disease descriptions
* Suggest precautions

---

## Future Improvements

* Add a web frontend using **React or HTML/CSS**
* Improve model accuracy with larger datasets
* Deploy API on **AWS / Render / Heroku**
* Add symptom autocomplete suggestions
* Build a full **health assistant system**

---

## Author

Aman Sharma

Machine Learning Enthusiast | Python Developer

---

## License

This project is created for **educational and research purposes**.
