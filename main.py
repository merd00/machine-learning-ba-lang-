import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score

# Load the data
data = pd.read_csv("C:/Users/mertt/Downloads/wine+quality/winequality-white.csv", delimiter=';')

# Check for missing values
print(data.isnull().sum())

# Preprocess the data
X = data.drop('quality', axis=1)
y = data['quality']

# Binarize the output variable (if needed, for binary classification)
y = (y > 5).astype(int)  # Assuming quality > 5 is 'good' wine and <= 5 is 'bad' wine

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data (important for some algorithms like SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC()
}

# Train and evaluate each classifier
results = {}
metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = cm.ravel()  # Flatten the confusion matrix into 1D array
    print(f"Confusion Matrix for {name}:\n{cm}\n")

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics['Accuracy'].append(acc)
    metrics['Precision'].append(prec)
    metrics['Recall'].append(rec)
    metrics['F1 Score'].append(f1)

    # Print metrics
    print(f"Metrics for {name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}\n")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

# Summary of results in a table format
summary_table = pd.DataFrame(results, index=['True Negative', 'False Positive', 'False Negative', 'True Positive'])
print(summary_table)

# Plot the summary table as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(summary_table, annot=True, cmap="YlGnBu", cbar=False)
plt.title('Confusion Matrix Summary Table')
plt.show()

# Create a DataFrame for metrics
metrics_df = pd.DataFrame(metrics, index=classifiers.keys())

# Plot metrics as bar charts
metrics_df.plot(kind='bar', figsize=(14, 8), fontsize=12)
plt.title('Classification Metrics', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1, fontsize=12)
plt.tight_layout()
plt.show()
