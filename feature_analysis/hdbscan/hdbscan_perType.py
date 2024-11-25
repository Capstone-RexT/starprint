import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

dataset_dir = ' '   # TODO : path for dataset 

with open(dataset_dir + 'TAM1D_FS_X.pkl', 'rb')  as handle:
    X = pickle.load(handle, encoding='latin1')
    X = np.array(X, dtype=object)

with open(dataset_dir + 'TAM1D_FS_Y.pkl', 'rb')  as handle:
    y = pickle.load(handle, encoding='latin1')
    y = np.array(y, dtype=object)

def makeHDBSCAN(target_class):
    # 1. 데이터 준비
    class_indices = np.where(y == target_class)  # Y에서 해당 클래스에 해당하는 인덱스 찾기
    X_class = X[class_indices]  # 해당 인덱스에 맞는 X 데이터 추출

    # 2. 데이터 전처리
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_class)

    # 3. HDBSCAN 클러스터링 적용
    # min_cluster_size는 데이터에 맞게 조정
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)  # 조정
    labels = clusterer.fit_predict(X_scaled)

    # 4. 결과 시각화 (PCA로 2D 차원 축소 후 시각화)
    # 시각화를 위해 PCA로 2차원으로 축소
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 클러스터링 결과 플로팅
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar(label='Cluster Label')
    plt.title("HDBSCAN Clustering Results")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # 그래프 저장
    plt.savefig(f"  /hdbscan_class_{target_class}.png", dpi=300)  # TODO : set the path to save files
    plt.close()

    print(f'{target_class} 완료')


for i in range(75):
    makeHDBSCAN(i)
