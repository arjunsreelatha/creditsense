import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import load_csv

def eda_pearson_corrcoef(X:np.ndarray)->np.ndarray:
    """computes the pearson correlation coeefient matrix for a dataset"""
    X_centered = X - np.mean(X,axis = 0)
    stds = X_centered.std(axis=0)
    stds[stds == 0] = 1e-8
    X_normalized = X_centered / stds
    corr_matrix = np.dot(X_normalized.T, X_normalized) / (X_normalized.shape[0] - 1)
    return corr_matrix

def eda_correlation_matrix(headers:list, data:np.ndarray) -> None:
    """compute and visualie the coreelation matrix and heatmap for a dataset"""
    features = data[:, 1:]  # Exclude target variable
    corr_matrix = eda_pearson_corrcoef(features)
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(headers)-1), labels=headers[1:], rotation=90)
    plt.yticks(ticks=np.arange(len(headers)-1), labels=headers[1:], rotation=0)
    plt.title("Feature Correlation Matrix")
    for i in range(len(headers)-1):
        for j in range(len(headers)-1):
            plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8) 
    plt.tight_layout()
    plt.savefig("C:\\Users\\Lenovo\\Desktop\\HOPE\\creditsense\\results\\correlation_matrix.png")
    plt.show()
    

def eda_correlation(filepath:str) -> None:
    """Loads dataset and computes correlation matrix."""
    headers, data = load_csv(filepath)
    eda_correlation_matrix(headers, data)

if __name__ == "__main__":
    eda_correlation(r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\cleaned\cs-training-cleaned.csv")    