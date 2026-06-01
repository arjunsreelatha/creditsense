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

def eda_feature_distribution(filepath:str) -> None:
    """loads dataset and visualises the distribution of each feature by target class"""
    headers, data = load_csv(filepath)
    target = data[:,0].astype(int)
    features = data[:,1:]
    feat_names = headers[1:]
   
    for i in range(features.shape[1]):
        vals0 = features[target == 0,i]
        vals1 = features[target == 1,i]
        plt.figure(figsize=(8,6))
        plt.hist(vals0,bins = 30,alpha = 0.5,color = "blue",label = "No Default")
        plt.hist(vals1,bins = 30,alpha = 0.5,color = "red",label = "Default")
        plt.axvline(np.mean(vals0),color = "blue",linestyle = "dashed",linewidth = 1)
        plt.axvline(np.mean(vals1),color = "red",linestyle = "dashed",linewidth = 1)
        plt.title(f"Distribution of {feat_names[i]} by Target Class")
        plt.xlabel(feat_names[i])
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"C:\\Users\\Lenovo\\Desktop\\HOPE\\creditsense\\results\\{feat_names[i]}_distribution.png")
        plt.show()  
        plt.close()  
def eda_variance_ranking(filepath:str) -> None:
    """loads dataset and rank features by variance"""
    headers,data = load_csv(filepath)
    features = data[:,1:]
    feat_names = headers[1:]

    variances = np.var(features,axis = 0)

    pairs = list(zip(feat_names,variances))
    pairs.sort(key = lambda x: x[1],reverse = True)
    print("\nFeature Variance Ranking:")
    print(f"{'feature':40} {'variance':>10}")
    for feature, variance in pairs:
        print(f"{feature:40} {variance:>10.4f}")

def eda_variance_per_class(filepath:str) -> None:
    """loads dataset and rank features by variance per class"""
    headers,data = load_csv(filepath)
    target = data[:,0].astype(int)
    features = data[:,1:]
    feat_names = headers[1:]

    variances_0 = np.var(features[target == 0],axis = 0)
    variances_1 = np.var(features[target == 1],axis = 0)

    pairs_0 = list(zip(feat_names,variances_0))
    pairs_1 = list(zip(feat_names,variances_1))

    pairs_0.sort(key = lambda x: x[1],reverse = True)
    pairs_1.sort(key = lambda x: x[1],reverse = True)

    print("\nFeature Variance Ranking for Class 0 (No Default):")
    print(f"{'feature':40} {'variance':>10}")
    for feature, variance in pairs_0:
        print(f"{feature:40} {variance:>10.4f}")

    print("\nFeature Variance Ranking for Class 1 (Default):")
    print(f"{'feature':40} {'variance':>10}")
    for feature, variance in pairs_1:
        print(f"{feature:40} {variance:>10.4f}")

def eda_top5_features(filepath: str) -> None:
    """
    Rank features by absolute Pearson correlation with the target.
    Plot top 5 as a horizontal bar chart sorted by correlation strength.
    """
    headers,data = load_csv(filepath)
    target = data[:,0].astype(int)
    features = data[:,1:]
    feat_names = headers[1:]
    corr_matrix = eda_pearson_corrcoef(data)
    target_corr = corr_matrix[0,1:]
    abs_corr = np.abs(target_corr)
    top5_indices = np.argsort(abs_corr)[-5:]
    top5_features = [feat_names[i] for i in top5_indices]
    top5_corr = target_corr[top5_indices]
    plt.figure(figsize=(8,6))
    bars = plt.barh(top5_features, top5_corr, color=['red' if c < 0 else 'blue' for c in top5_corr])
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("Pearson Correlation with Target")
    plt.title("Top 5 Features by Absolute Correlation with Target")
    plt.tight_layout()
    plt.savefig(f"C:\\Users\\Lenovo\\Desktop\\HOPE\\creditsense\\results\\top5_features.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    filepath = r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\cleaned\cs-training-cleaned.csv"
    
    eda_top5_features(filepath)