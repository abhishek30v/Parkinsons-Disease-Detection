import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from algorithms import evaluate_algorithms
from utils import load_data

# Set the environment variable to suppress joblib warning
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def preprocess_data(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(['sex', 'subject#'], axis=1))
    preprocessed_df = pd.DataFrame(scaled_features, columns=df.columns.drop(['sex', 'subject#']))
    preprocessed_df['sex'] = df['sex'].values
    return preprocessed_df, scaler

def evaluate_clusters(preprocessed_df):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(preprocessed_df.drop('sex', axis=1))
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

def apply_clustering(preprocessed_df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    preprocessed_df['cluster'] = kmeans.fit_predict(preprocessed_df.drop('sex', axis=1))
    return preprocessed_df, kmeans.cluster_centers_

def visualize_clusters(preprocessed_df):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=preprocessed_df.iloc[:, 0], y=preprocessed_df.iloc[:, 1], hue=preprocessed_df['cluster'], palette='viridis')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Cluster')
    plt.show()
    pass

def train_best_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)  # Updated to Random Forest
    model.fit(X_train, y_train)
    return model

def drop_unexpected_columns(df, expected_columns):
    current_columns = set(df.columns)
    unexpected_columns = current_columns - set(expected_columns)
    if unexpected_columns:
        print(f"Dropping unexpected columns: {unexpected_columns}")
        df = df.drop(unexpected_columns, axis=1)
    return df

def visualize_sample_input(file_path):
    sample_input_df = pd.read_csv(file_path)
    # sample_input_df.hist(bins=15, figsize=(15, 15), edgecolor='black')
    # plt.suptitle("Sample Input Data Distributions")
    # plt.show()

def predict_status_from_file(file_path, model, scaler, expected_columns):
    sample_input_df = pd.read_csv(file_path)
    sample_input_df = drop_unexpected_columns(sample_input_df, expected_columns)

    sample_input_scaled = scaler.transform(sample_input_df)
    predictions = model.predict(sample_input_scaled)
    status_labels = {0: 'Parkinsons Early Stage', 1: 'Normal', 2: 'Parkinsons Advanced Stage'}
    predicted_statuses = [status_labels[prediction] for prediction in predictions]
    return predicted_statuses

def main():
    raw_data_path = 'C:/Users/av725/Downloads/Parkinson-s-Disease-Detection/dataset/parkinsons_raw.data'
    csv_data_path = 'C:/Users/av725/Downloads/Parkinson-s-Disease-Detection/dataset/parkinsons.csv'
    model_path = 'C:/Users/av725/Downloads/Parkinson-s-Disease-Detection/model/best_model.pkl'
    scaler_path = 'C:/Users/av725/Downloads/Parkinson-s-Disease-Detection/model/scaler.pkl'
    sample_input_path = 'C:/Users/av725/Downloads/Parkinson-s-Disease-Detection/dataset/sample_input.csv'

    df = pd.read_csv(raw_data_path, header=0)
    df.to_csv(csv_data_path, index=False)
    df = pd.read_csv(csv_data_path)
    
    sns.countplot(data=df, x='sex')
    plt.title('Count of gender in the Dataset')
    plt.xlabel('gender (0 = male, 1 = Female)')
    plt.ylabel('Count')
    plt.show()

    # df.hist(bins=15, figsize=(15, 15), edgecolor='black')
    # plt.suptitle("Feature Distributions")
    # plt.show()

    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()
    
    preprocessed_df, scaler = preprocess_data(df)
    evaluate_clusters(preprocessed_df)
    preprocessed_df, cluster_centers = apply_clustering(preprocessed_df)
    visualize_clusters(preprocessed_df)
    
    df['cluster'] = preprocessed_df['cluster']
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cluster')
    plt.title('Distribution of Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()
    
    X = preprocessed_df.drop(['sex', 'cluster'], axis=1).values
    y = preprocessed_df['cluster'].values
    data, preprocessed_df, scaler = load_data(csv_data_path) 
    X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.2, random_state=42)
    model = train_best_model(X_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    expected_columns = preprocessed_df.drop(['sex', 'cluster'], axis=1).columns.tolist()
    
    # Evaluate the model on the test set 
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")

    # print("\nClassification Report:") 
    # print(classification_report(y_test, y_pred))
    # visualize_sample_input(sample_input_path)

    predicted_statuses = predict_status_from_file(sample_input_path, model, scaler, expected_columns)
    
    print("\nEvaluating different algorithms:")
    evaluate_algorithms()

    print("\nFinal Predicted Disease Conditions for Sample Input:")
    for i, status in enumerate(predicted_statuses):
        print(f"Sample {i+1}: {status}")

if __name__ == "__main__":
    main()
