import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np

# X: 입력 데이터 (network traffic direction, 2차원)
# Y: 각 인스턴스의 클래스 레이블

dataset_dir = 'path for dataset'

with open(dataset_dir + 'FS_X_ipd.pkl', 'rb')  as handle:
    X = pickle.load(handle, encoding='latin1')
    X = np.array(X, dtype=object)

with open(dataset_dir + 'FS_Y_ipd.pkl', 'rb')  as handle:
    y = pickle.load(handle, encoding='latin1')
    y = np.array(y, dtype=object)

# 1. 특정 클래스 선택 (예: 클래스 3에 해당하는 데이터)

def makeTSNE(target_class):
    class_indices = np.where(y == target_class)  # Y에서 해당 클래스에 해당하는 인덱스 찾기
    X_class = X[class_indices]  # 해당 인덱스에 맞는 X 데이터 추출

    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_class)  # 해당 클래스 데이터로 t-SNE 변환

    # t-SNE 그래프 시각화
    plt.figure()  # 새로운 figure 생성
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.title(f't-SNE for Class {target_class} - starink_IPD')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    # 그래프 저장
    plt.savefig(f"path_to_save_the_image/tsne_class_{target_class}.png", dpi=300)  # 그래프 저장
    plt.close()

for i in range(75):
    makeTSNE(i)
