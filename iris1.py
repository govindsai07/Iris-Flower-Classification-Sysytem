import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class IrisClassifier:
    def __init__(self):
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
    
    def load_data(self, csv_file=None):
        """Load data from CSV file or default iris dataset"""
        if csv_file:
            try:
                data = pd.read_csv(csv_file)
                self.X = data.iloc[:, :-1].values
                self.y = data.iloc[:, -1].values
                self.feature_names = list(data.columns[:-1])
                self.target_names = sorted(data.iloc[:, -1].unique())
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return None
        else:
            iris = load_iris()
            self.X = iris.data
            self.y = iris.target
            self.feature_names = iris.feature_names
            self.target_names = iris.target_names
        
        return pd.DataFrame(data=self.X, columns=self.feature_names)
    
    def preprocess_data(self, test_size=0.3, random_state=42):
        """Split and scale data"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def train_model(self, n_estimators=100, random_state=42):
        """Train RandomForest classifier"""
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """Evaluate performance"""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.target_names))
    
    def predict(self, samples):
        """Predict species for new samples"""
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
        
        samples_scaled = self.scaler.transform(samples)
        pred_indices = self.model.predict(samples_scaled)
        predictions = [self.target_names[idx] for idx in pred_indices]
        
        return predictions

def plot_data(data):
    """Visualize pairplots and boxplots"""
    plt.figure(figsize=(10, 8))
    sns.pairplot(data, hue='species', height=2.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(data.columns[:-2]):  
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='species', y=feature, data=data)
        plt.title(f'{feature} by Species')
    plt.tight_layout()
    plt.show()

def main():
    classifier = IrisClassifier()
    df = classifier.load_data("datapr.csv")
    if df is not None:
        df['species'] = df['target'].map({i: name for i, name in enumerate(classifier.target_names)})
        plot_data(df)

        classifier.preprocess_data()
        classifier.train_model()
        classifier.evaluate_model()
        
        samples = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [7.9, 3.8, 6.4, 2.0]]
        predictions = classifier.predict(samples)
        for sample, pred in zip(samples, predictions):
            print(f"Sample {sample}: Predicted Species -> {pred}")

if __name__ == "__main__":
    main()