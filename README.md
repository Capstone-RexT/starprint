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

## **구성원 및 역할**  
- **팀원**:  
  - **곽현정**: 프로젝트 총괄, 모델 아키텍처 탐색 및 Llama 모델 구현
  - **강호성**: Fingerprinting 모델 설계 및 구현  
  - **홍지우**: 네트워크 취약점 분석 및 보고서 작성  
---

## **성과 및 기대 효과**  
- 위성 네트워크의 취약점을 구체적으로 규명하여 관련 보안 연구에 기여  
- 데이터 기반 분석 결과를 바탕으로 Starlink의 보안 강화 방향 제시  

---

## **시작하기 (Getting Started)**  
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

## **문의**  
프로젝트에 대한 문의는 이메일로 연락해 주세요.  
- **이메일**: 2171003@ewha.ac.kr  
