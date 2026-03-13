# 🧠 스트레스 지수 예측 AI 알고리즘 개발

## 1. 프로젝트 개요
**데이콘 Basic 스트레스 지수 예측 : 건강 데이터로 마음의 균형을 찾아라!**

- **배경:** 현대인들의 일상에 깊숙이 자리 잡은 스트레스는 신체적, 정신적 건강에 심각한 영향을 미치고 있으나, 많은 사람들이 자신의 스트레스 수준을 객관적으로 인식하지 못하고 있습니다. 본 해커톤에서는 신체 정보, 수면 패턴, 활동량 등 다양한 건강 데이터를 활용하여 개인의 스트레스 점수를 예측하는 AI 알고리즘을 개발합니다.
- **주제:** 신체 데이터를 기반으로 스트레스 점수를 예측하는 AI 알고리즘 개발
- **목표:** 데이터를 기반으로 한 객관적인 스트레스 점수 예측 모델을 구축하여, 현대인들이 더 건강하고 균형 잡힌 삶을 영위할 수 있도록 기여합니다.

## 2. 개발 환경 및 기술 스택
- **Language:** Python
- **Modeling:** 
- **Environment:** 
- **Version Control:** Git, GitHub

## 3. 프로젝트 구조
```text
├── configs/
│   └── config.yaml        # 데이터 경로 및 모델 하이퍼파라미터 설정 파일
├── data/
│   ├── raw/               # 원본 데이터 (train.csv, test.csv 등)
│   └── processed/         # 전처리 완료된 데이터
├── outputs/
│   └── submissions/       # 최종 예측 결과 제출용 csv 파일
├── src/                   # 주요 소스 코드
│   ├── eda.ipynb          # 탐색적 데이터 분석(EDA) 노트북
│   ├── preprocess.py      # 데이터 전처리 스크립트
│   └── train.py           # 모델 학습 및 평가 스크립트
├── requirements.txt       # 필요한 파이썬 패키지 목록
└── README.md              # 프로젝트 설명서
```

## 4. Git 컨벤션 (Git Convention)
원활한 협업과 버전 관리를 위해 아래의 커밋 메시지 규칙과 브랜치 전략을 따릅니다.

### 📌 커밋 메시지 규칙
태그: 작업 내용 형식으로 작성합니다. (예: feat: 결측치 처리 로직 추가)
- feat : 새로운 기능 추가, 데이터 전처리 로직 구현, 새 모델 적용
- fix : 버그 및 에러 수정
- refactor : 결과의 변경 없이 코드 구조 재작성 및 최적화
- chore : 패키지 설치, 디렉토리 구조 변경 등 기타 작업
- docs : README, 주석 등 문서 수정
- experiment : 하이퍼파라미터 튜닝 등 실험적인 모델 변경사항

### 🌿 브랜치(Branch) 전략
- main : 최종 제출 가능한 안정적인 코드가 유지되는 브랜치
- develop : 기능 개발이 병합되는 테스트 브랜치
- feat/이름 : 개인별 기능 개발 브랜치 (예: feat/eda, feat/lgbm-tuning)
    - 작업 완료 후 develop 브랜치로 Pull Request(PR) 진행
    - *개인별 기능 검토해서 테스트 후 PR 진행 여부 결정

## 5. 실행 방법 (How to Run)

1. 환경 세팅: pip install -r requirements.txt
2. 설정 확인: config.yaml 파일에서 데이터 경로와 실험 세팅 확인
3. 모델 학습: python src/train.py

## 6. 팀원 소개 (Team Members)

<div align="center">
  <table>
    <tr>
      <th>팀원</th>
      <th>팀원</th>
      <th>팀원</th>
    </tr>
    <tr align="center">
      <td><img src="https://avatars.githubusercontent.com/u/156976285?v=4" width="100"/><br/>안은지</td>
      <td><img src="https://avatars.githubusercontent.com/u/156976285?v=4" width="100"/><br/>이설미</td>
      <td><img src="https://avatars.githubusercontent.com/u/156976285?v=4" width="100"/><br/>조영현</td>
    </tr>
  </table>
</div>