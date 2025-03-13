import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import shap

# Dataset Paths
datasets = {
    "ant-1.7": "/content/ant-1.7.csv",
    "camel-1.6": "/content/camel-1.6.csv",
    "forrest-0.8": "/content/forrest-0.8.csv",
    "ivy-2.0": "/content/ivy-2.0.csv",
    "log4j-1.2": "/content/log4j-1.2.csv",
    "poi-3.0": "/content/poi-3.0.csv",
    "synapse-1.2": "/content/synapse-1.2.csv",
    "velocity-1.6": "/content/velocity-1.6.csv",
    "xalan-2.7": "/content/xalan-2.7.csv",
    "xerces-1.4": "/content/xerces-1.4.csv",
}

# Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# Preprocessing Function
def preprocess_data(file_path):
    X, y = load_data(file_path)
    class_counts = np.bincount(y)
    minority_count = np.min(class_counts)
    smote_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

    try:
        if minority_count > 1:
            smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            raise ValueError("Not enough samples for SMOTE.")
    except ValueError:
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    return X_scaled, y_resampled

# Graph Neural Network (GCN + GAT)
class GNNFeatureTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNFeatureTransformer, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.gat = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        x = self.gat(x, edge_index)
        return x

# Create adjacency matrix for graph data
def create_graph(X):
    num_nodes = X.shape[0]
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
    return edge_index

# SHAP Feature Selection
def shap_feature_selection(model, X_train, y_train):
    model.fit(X_train, y_train)  # Ensure model is trained before SHAP
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    top_features = np.argsort(feature_importance)[-10:]  # Select top 10 features
    return top_features

# Deep Neural Network with Focal Loss
class DNNWithFocalLoss(nn.Module):
    def __init__(self, input_dim):
        super(DNNWithFocalLoss, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def focal_loss(self, outputs, targets, alpha=0.25, gamma=2):
        bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

# Augment Data (10% merge + SMOTE)
def augment_data(X_train, y_train, X_test, y_test):
    test_sample_count = int(0.1 * len(X_test))
    X_train_augmented = np.vstack((X_train, X_test[:test_sample_count]))
    y_train_augmented = np.hstack((y_train, y_test[:test_sample_count]))

    smote = SMOTE(random_state=42)
    X_train_doubled, y_train_doubled = smote.fit_resample(X_train_augmented, y_train_augmented)

    return X_train_doubled, y_train_doubled

# Train and evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42)),
    ]

    base_predictions = []
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        base_predictions.append(preds)

    base_predictions = np.column_stack(base_predictions)

    # Define and train DNN with Focal Loss
    dnn_model = DNNWithFocalLoss(input_dim=base_predictions.shape[1])
    optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)

    X_train_torch = torch.tensor(base_predictions, dtype=torch.float32)
    y_train_torch = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = dnn_model(X_train_torch)
        loss = dnn_model.focal_loss(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

    final_preds = (outputs.detach().numpy() > 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, final_preds),
        "Precision": precision_score(y_test, final_preds, average="macro"),
        "Recall": recall_score(y_test, final_preds, average="macro"),
        "F1-Score": f1_score(y_test, final_preds, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, final_preds),
        "Balanced Accuracy": balanced_accuracy_score(y_test, final_preds),
        "G-Mean": np.sqrt(recall_score(y_test, final_preds, pos_label=0) * recall_score(y_test, final_preds, pos_label=1))
    }
    return metrics

# Run the CPDP process with dataset integration
def run_cpdp(datasets):
    results = []
    for train_name, train_path in datasets.items():
        X_train, y_train = preprocess_data(train_path)

        # Apply Graph-based Feature Transformation
        edge_index = create_graph(X_train)  # Create adjacency matrix
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        gnn = GNNFeatureTransformer(input_dim=X_train.shape[1], hidden_dim=16, output_dim=X_train.shape[1])
        X_train_transformed = gnn(X_train_torch, edge_index).detach().numpy()  # Get transformed features

        for test_name, test_path in datasets.items():
            if train_name != test_name:
                X_test, y_test = preprocess_data(test_path)

                # Apply Data Augmentation
                X_train_augmented, y_train_augmented = augment_data(X_train_transformed, y_train, X_test, y_test)

                # Apply SHAP Feature Selection
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                top_features = shap_feature_selection(model, X_train_augmented, y_train_augmented)
                X_train_selected = X_train_augmented[:, top_features]  # Select top features
                X_test_selected = X_test[:, top_features]  # Select top features for test

                # Train and evaluate model
                metrics = train_and_evaluate(X_train_selected, y_train_augmented, X_test_selected, y_test)

                results.append({
                    "Train Dataset": train_name,
                    "Test Dataset": test_name,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1-Score": metrics["F1-Score"],
                    "ROC-AUC": metrics["ROC-AUC"],
                    "Balanced Accuracy": metrics["Balanced Accuracy"],
                    "G-Mean": metrics["G-Mean"]
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/content/cpdp_results.csv", index=False)
    return results_df

if __name__ == "__main__":
    results = run_cpdp(datasets)
    print(results)
    print("Complete CPDP model with dataset integration, GCN, GAT, SHAP feature selection, and ensemble learning using DNN with Focal Loss. Results saved to 'cpdp_results.csv'.")
