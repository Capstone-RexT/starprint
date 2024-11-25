import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

# 데이터 경로 설정
dataset_dir = '/scratch4/starlink/baseline/feature/time_direction/'

# 데이터 로드
with open(dataset_dir + 'TikTok_FF_X_direction.pkl', 'rb') as handle:
    X = pickle.load(handle, encoding='latin1')
    X = np.array(X, dtype=object)

with open(dataset_dir + 'TikTok_FF_Y_direction.pkl', 'rb') as handle:
    y = pickle.load(handle, encoding='latin1')
    y = np.array(y, dtype=object)

def makeHDBSCAN(target_classes):
    # 1. 데이터 준비
    # 선택된 클래스들에 해당하는 인덱스 찾기 및 데이터 병합
    class_indices = np.isin(y, target_classes)
    X_class = X[class_indices]
    y_class = y[class_indices]

    # 2. 데이터 전처리
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_class)

    # 3. HDBSCAN 클러스터링 적용
    # min_cluster_size는 데이터에 맞게 조정
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X_scaled)

    # 4. 결과 시각화 (PCA로 2D 차원 축소 후 시각화)
    # 시각화를 위해 PCA로 2차원으로 축소
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 클러스터링 결과 플로팅
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(f"HDBSCAN Clustering Results for Classes {target_classes[0]} ~ {target_classes[-1]}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # 그래프 저장
    #plt.savefig(f"/home/jiwoo0914/starlink/traffic_analysis/HDBSCAN/6._perAllClasses/hdbscan_classes_{'_'.join(map(str, target_classes))}.png", dpi=300)
    plt.savefig(f"/home/jiwoo0914/starlink/traffic_analysis/HDBSCAN/6._perAllClasses/hdbscan_2.png", dpi=300)
    plt.close()

    print(f'Classes {target_classes} 완료')

# 예시: 클래스 0에서 9까지를 하나의 그래프에 나타내기
target_classes = list(range(40))
makeHDBSCAN(target_classes)
