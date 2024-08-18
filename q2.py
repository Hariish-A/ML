import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Log Loss function
def log_loss(y, y_pred):
    return -(y * np.log(y_pred)) - ((1 - y) * np.log(1 - y_pred))

# Compute cost
def compute_cost(y, y_pred):
    m = len(y)
    cost = np.sum(log_loss(y, y_pred)) / m
    return cost

# Gradient Descent
def gradient_descent(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    cost_history = []

    for i in range(epochs):
        z = np.dot(X, w) + b 
        y_pred = sigmoid(z) 

        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        w -= alpha * dw
        b -= alpha * db

        cost = compute_cost(y, y_pred)
        cost_history.append(cost)

    return w, b, cost_history

# Prediction
def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    pred_class = [1 if p >= 0.50 else 0 for p in y_pred]
    return np.array(pred_class)

# Load and preprocess the data
df = pd.read_csv(r"C:\Users\Aparna\Downloads\DIABETICS DATASET.csv")

df.corr()

scaler = StandardScaler()
columns_to_scale = ['Glucose', 'BMI', 'Age']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

X = df[['Glucose', 'BMI', 'Age']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
w, b, cost_history = gradient_descent(X_train, y_train, alpha=0.1, epochs=1000)

# Make predictions
y_pred = predict(X_test, w, b)

# Evaluate the model
print("Weights and Bias:")
for feature, weight in zip(X.columns, w):
    print(f"{feature}: {weight:.4f}")
print(f"\nBias: {b:.4f}")


# Accuracy
def accuracy(y, y_pred):
    correct_predictions = np.sum(y == y_pred)
    return correct_predictions / len(y)

# Precision
def precision(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == 0) & (y_pred == 1))
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)

# Recall
def recall(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)

# F1 Score
def f1_score(y, y_pred):
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    if (prec + rec) == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    cm = np.array([[TN, FP],
                   [FN, TP]])
    return cm

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot ROC AUC
def plot_roc_auc(X_test, y_test, w, b):
    y_prob = sigmoid(np.dot(X_test, w) + b)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

acc = accuracy(y_test, y_pred)
prec = precision(y_test, y_pred)
rec = recall(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)

# Plot ROC AUC
plot_roc_auc(X_test, y_test, w, b)
