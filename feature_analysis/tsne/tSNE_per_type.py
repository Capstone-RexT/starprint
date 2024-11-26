import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# X: 입력 데이터 (network traffic direction, 2차원)
# Y: 각 인스턴스의 클래스 레이블

dataset_dir = ' '

with open(dataset_dir + 'FF_X_ipd.pkl', 'rb')  as handle:
    Xf = pickle.load(handle, encoding='latin1')
    Xf = np.array(Xf, dtype=object)

with open(dataset_dir + 'FF_Y_ipd.pkl', 'rb')  as handle:
    yf = pickle.load(handle, encoding='latin1')
    yf = np.array(yf, dtype=object)
    
with open(dataset_dir + 'FS_X_ipd.pkl', 'rb')  as handle:
    Xs = pickle.load(handle, encoding='latin1')
    Xs = np.array(Xs, dtype=object)

with open(dataset_dir + 'FS_X_ipd.pkl', 'rb')  as handle:
    ys = pickle.load(handle, encoding='latin1')
    ys = np.array(ys, dtype=object)

# 1. 특정 클래스 선택 (예: 클래스 3에 해당하는 데이터)

def makeTSNE(target_class):
    fiber_class_indices = np.where(yf == target_class)  # Y에서 해당 클래스에 해당하는 인덱스 찾기
    fiber_X_class = Xf[fiber_class_indices]  # 해당 인덱스에 맞는 X 데이터 추출

    starlink_class_indices = np.where(ys == target_class)  # Y에서 해당 클래스에 해당하는 인덱스 찾기
    starlink_X_class = Xs[starlink_class_indices]  # 해당 인덱스에 맞는 X 데이터 추출

    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42)

    # Fiber 데이터에 대한 t-SNE
    fiber_X_tsne = tsne.fit_transform(fiber_X_class)
    
    # Starlink 데이터에 대한 t-SNE
    starlink_X_tsne = tsne.fit_transform(starlink_X_class)

    # t-SNE 그래프 시각화
    plt.figure(figsize=(10, 8))  # 그래프 크기 설정
    plt.scatter(fiber_X_tsne[:, 0], fiber_X_tsne[:, 1], color='red', label='Fiber Data', alpha=0.6)  # Fiber 데이터 포인트
    plt.scatter(starlink_X_tsne[:, 0], starlink_X_tsne[:, 1], color='blue', label='Starlink Data', alpha=0.6)  # Starlink 데이터 포인트
    plt.title(f't-SNE Comparison for Class {target_class}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()  # 범례 추가
    plt.grid(True)  # 격자 추가
    plt.savefig(f"  /comparison_class_{target_class}.png", dpi=300)  # 그래프 저장
    plt.show()  # 그래프 표시

for i in range(75):
    makeTSNE(i)
