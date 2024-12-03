# **Starprint: 위성인터넷 해킹을 통한 프라이버시 유출 방지를 위한 Fingerprinting 기법을 활용한 Starlink 네트워크 취약점 분석**  
**2024~2025 Ewha Womans University Capstone Design Project - Team16 RexT**


## **소개**  
Starprint는 **위성인터넷 해킹으로 인한 프라이버시 유출 방지**를 목표로, **Website Fingerprinting 기법**을 활용하여 **Starlink 네트워크의 보안 취약점**을 분석하는 프로젝트입니다.  
본 연구는 위성 네트워크의 보안 강화를 위한 학문적, 실질적 기여를 목표로 하고 있습니다.  

---

## **프로젝트 개요**  
- **프로젝트 기간**: 2024년  
- **팀명**: Team16 RexT  
- **소속**: 이화여자대학교 캡스톤디자인 프로젝트  
- **주요 목표**:  
  1. Starlink 위성 네트워크 트래픽 패턴 분석
  2. Website Fingerprinting 기술을 적용해 Starlink의 프라이버시 노출을 비롯한 보안 취약성 검증
  3. Starlink를 비롯한 위성인터넷을 위한 새로운 보안 솔루션 및 보안 프로토콜 제시  

---

## **기술 및 도구**  
- **프로그래밍 언어**: Python
- **데이터 분석**: TensorFlow, Scikit-learn  
- **데이터 전처리**: Torch 
- **모델 학습**: Keras

---

## **모델 소개**  
![image](https://github.com/user-attachments/assets/1d5a4ef5-94ef-4484-96b3-2a2140a0688f)

---

## **시작하기 (Getting Started)**  
### **프로젝트 폴더 구성 및 역할**  
1. **`feature_analysis`**
   - **역할**: 가지고 있는 네트워크 트래픽의 분포를 확인하고 분석합니다.
    
3. **`feature_extractor`**  
   - **역할**: 네트워크 트래픽에서 주요 특징(feature)을 추출하는 스크립트를 포함합니다.  
   - **주요 파일**:
     - `functions.py`: 각 메타데이터마다 feature를 추출하는 함수들을 모아놓은 모듈입니다.
     - `pkl_extractor.py`: 수집된 네트워크 트래픽 데이터를 전처리하고, pickle 형식으로 저장합니다. 

4. **`embedding_extractor`**  
   - **역할**: 앞에서 추출된 트래픽의 feature를 입력으로 받아, 모델 학습에 사용될 임베딩 벡터를 생성합니다.  
   - **주요 파일**:  
     - `llama_extractor.py`: Llama 모델을 사용하여 트래픽 데이터를 임베딩 공간으로 매핑합니다.  

5. **`models/star_laserbeak`**  
   - **역할**: 학습된 모델을 사용하여 위성 네트워크 트래픽의 분류를 수행합니다.  
   - **주요 파일**:  
     - `laserbeak_1d_main.py`: 1D CNN 기반 분류 모델을 실행하여 네트워크 트래픽 분석 결과를 출력합니다.  

---
### **설치 및 실행**
1. **필수 요구사항**:
   - python 3.7 이상
   - cuda 11.4
   - cnDNN

3. **설치 및 실행**:  
   ```bash
   # 프로젝트 클론
   git clone https://github.com/Capstone-RexT/starprint
   
   # 환경 세팅
   cd starprint
   pip install -r requirements.txt
   ```

   ```bash
   # Feature 추출
   cd feature_extractor
   python pkl_extractor.py
   ```

   ```bash
   # Embedding vector 추출
   cd embedding_extractor
   python llama_extractor.py
   ```
   
   ```bash
   # Classification 수행
   cd models/star_laserbeak
   python laserbeak_1d_main.py
   ```
---

## 기술 시연 영상
[![Watch the video](https://img.youtube.com/vi/TnjwFFnJn-4/maxresdefault.jpg)](https://www.youtube.com/watch?v=TnjwFFnJn-4)

---

## **구성원 및 역할**  
- **팀원**:  
  - **곽현정**: 프로젝트 총괄, 모델 아키텍처 탐색 및 Llama 모델 구현
  - **강호성**: Fingerprinting 모델 설계 및 구현  
  - **홍지우**: 네트워크 취약점 분석 및 traffic analysis, Fingerprinting 모델 설계 및 구현
---

## **성과 및 기대 효과**  
- 위성 네트워크의 취약점을 구체적으로 규명하여 관련 보안 연구에 기여  
- 데이터 기반 분석 결과를 바탕으로 Starlink의 보안 강화 방향 제시  

---

## **문의**  
프로젝트에 대한 문의는 이메일로 연락해 주세요.  
- **이메일**: 2171003@ewha.ac.kr  
