import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import cm

# X: 입력 데이터 (network traffic direction, 2차원)
# Y: 각 인스턴스의 클래스 레이블
with open('path_for_dataset/FF_X_burst_size.pkl', 'rb') as handle:
    X = pickle.load(handle, encoding='latin1')
    X = np.array(X, dtype=object)

with open('path_for_dataset//FF_Y_burst_size.pkl', 'rb') as handle:
    y = pickle.load(handle, encoding='latin1')
    y = np.array(y, dtype=object)

# 1. 여러 클래스를 t-SNE로 변환하고 시각화하는 함수
def makeTSNE_for_multiple_classes(target_classes):
    # 선택한 클래스들의 데이터를 모두 모아 X_subset에 저장
    X_subset = []
    y_subset = []
    
    for target_class in target_classes:
        class_indices = np.where(y == target_class)  # Y에서 해당 클래스에 해당하는 인덱스 찾기
        X_class = X[class_indices]  # 해당 인덱스에 맞는 X 데이터 추출
        X_subset.append(X_class)
        y_subset.append(np.full(X_class.shape[0], target_class))  # 해당 클래스 레이블 추가

    # 데이터를 합치기
    X_subset = np.concatenate(X_subset, axis=0)
    y_subset = np.concatenate(y_subset, axis=0)
    
    # 2. t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_subset)  # t-SNE 변환

    # 3. t-SNE 그래프 시각화
    plt.figure(figsize=(10, 8))

    # 4. 클래스별로 색상을 달리해서 시각화
    #colors = cm.get_cmap('tab10', len(target_classes))  # 10개의 클래스에 맞는 색상 팔레트
    colors = cm.get_cmap('Set1', len(target_classes)) 
    for i, target_class in enumerate(target_classes):
        indices = np.where(y_subset == target_class)
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], color=colors(i), label=f'Class {target_class}', alpha=0.6)

    plt.title(f't-SNE Comparison for Classes {target_classes}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()  # 범례 추가
    plt.grid(True)  # 격자 추가
    
    plt.savefig(f"path_for_save_files/FF_burst_multiple_classes.png", dpi=300)  # 그래프 저장
    plt.show()  # 그래프 표시

# 10개의 클래스 선택
target_classes = [1,22,45,57,70]  # 비교할 클래스 번호
makeTSNE_for_multiple_classes(target_classes)
