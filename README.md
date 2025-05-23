# Iris Flower Classification System 🌿🌸

This project is a machine learning-based **Iris Flower Classification System** that uses the **Random Forest** algorithm to classify iris species based on their features. It includes **data preprocessing, visualization, model training, evaluation, and prediction**.

## 📌 Features
- Loads data from the **Iris dataset** or a custom CSV file.
- Provides **data exploration** using statistics and visualizations (pairplots & boxplots).
- **Preprocesses** data with train-test splitting and feature scaling.
- **Trains a Random Forest classifier** for accurate species prediction.
- **Evaluates the model** using accuracy, classification reports, and confusion matrix.
- **Predicts species** based on new measurements with confidence probabilities.

## 🚀 Installation
To use this project, first install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

📂 Files Overview
- iris_classifier.py - Main Python script for the classification system.
- datapr.csv - Sample dataset (if applicable).
- iris_pairplot.png - Pairplot visualization.
- iris_boxplots.png - Boxplot visualization.
- feature_importance.png - Feature importance plot.
- confusion_matrix.png - Confusion matrix heatmap.

🔬 Usage
Run the script using:
python iris_classifier.py
Or customize it by loading your own dataset:
classifier = IrisClassifier()
df = classifier.load_data(csv_file="custom_data.csv")

Example Prediction
sample_measurements = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [7.9, 3.8, 6.4, 2.0]]
re

📊 Model Evaluation
The classifier provides:
- Accuracy Score
- Feature Importance Rankings
- Confusion Matrix Visualization
- Detailed Classification Report
💡 Future Enhancements
- Support for additional flower datasets.
- Integration of deep learning models.
- Deployment as a web-based application.
