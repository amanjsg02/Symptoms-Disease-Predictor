This project is an AI-powered disease prediction system that takes a list of symptoms from the user and predicts
the most probable disease using a Random Forest Classifier trained on medical symptom data. Once a disease is 
predicted, the system also returns a brief description of the disease and a set of recommended precautions to be followed.
To make the model accessible and user-friendly, the entire system is deployed using Flask, a lightweight web framework in Python. 
Flask serves as the backend API that receives user symptoms through POST requests, processes the input, uses the trained ML
model to make predictions, and responds with useful health information. 
