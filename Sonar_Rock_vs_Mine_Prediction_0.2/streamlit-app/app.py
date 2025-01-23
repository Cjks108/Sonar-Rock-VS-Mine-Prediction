import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
sonar_data = pd.read_csv('C:/Users/98chi/PycharmProjects/Sonar_Rock_vs_Mine_Prediction/streamlit-app/sonar_data.csv', header=None)

# Data preparation
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Function to make a prediction
def predict_object(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_numpy_array)
    return prediction[0]

# Streamlit App UI
st.title("Rock vs Mine Classifier 🪨💣")
st.write("Upload a sonar signal dataset or input feature values to predict whether the object is a Rock or a Mine.")

# Input for feature values
input_data = st.text_area(
    "Enter feature values separated by commas (e.g., 0.1083,0.107,...):",
    "0.1083,0.107,0.0257,0.0837,0.0748,0.1125,0.3322,0.459,..."
)

if st.button("Predict"):
    try:
        # Convert input data to float array
        input_data_list = [float(x) for x in input_data.split(",")]
        prediction = predict_object(input_data_list)

        if prediction == 'R':
            st.success("The Object is a Rock! 🪨")
            st.image("C:/Users/98chi/PycharmProjects/Sonar_Rock_vs_Mine_Prediction/streamlit-app/assets/rock_image.jpg", caption="Rock", width=300)  # Replace with your rock image file
        else:
            st.success("The Object is a Mine! 💣")
            st.image("C:/Users/98chi/PycharmProjects/Sonar_Rock_vs_Mine_Prediction/streamlit-app/assets/mine_image.jpg", caption="Mine", width=300)  # Replace with your mine image file
    except Exception as e:
        st.error("Invalid input data! Please enter the correct number of feature values.")

# Display accuracy metrics
st.sidebar.header("Model Accuracy Metrics")
training_accuracy = accuracy_score(model.predict(X_train), Y_train) * 100
testing_accuracy = accuracy_score(model.predict(X_test), Y_test) * 100

st.sidebar.write(f"*Training Accuracy:* {training_accuracy:.2f}%")
st.sidebar.write(f"*Testing Accuracy:* {testing_accuracy:.2f}%")
