# Sonar-Rock-VS-Mine-Prediction_0.2
A Streamlit-based web application that uses a Logistic Regression model to classify objects as either a Rock or a Mine based on sonar signal data. Includes interactive UI, prediction visualizations, and model accuracy metrics.

## Video 
[![SONAR Rock vs Mine Prediction with Logistic Regression ]()

Click on the thumbnail above to watch the complete tutorial on YouTube.

## Dataset
- The dataset used is the Sonar dataset from the UCI Machine Learning Repository.
- It contains 60 features representing sonar signals and a label (`R` for rock and `M` for mine).

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (optional for visualization)

## Installation
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage
```bash
python sonar_rock_vs_mine.py
```

## Results
- Achieved an accuracy of approximately 83% on the test dataset.
- Confusion matrix and classification report are generated for evaluation.

## Conclusion
Logistic Regression effectively distinguishes between sonar signals of rocks and mines. Further improvements can be made using hyperparameter tuning and ensemble models.
