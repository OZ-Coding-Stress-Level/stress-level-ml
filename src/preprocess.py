import pandas as pd
import numpy as np
from typing import Tuple # Type Hinting을 위해
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 결측치 처리
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 수치형 변수 결측치 처리
    # mean_working 변수의 결측치 처리
    if 'mean_working' in df.columns:
        # 중앙값과 평균값 비교
        median_val = df['mean_working'].fillna(df['mean_working'].median())
        mean_val = df['mean_working'].fillna(df['mean_working'].mean())
        print(f"중앙값 : {median_val}")
        print(f"평균값 : {mean_val}")
        # [근무시간 결측치 중앙값] XGBoost + CatBoost MAE: 0.1710 / R2 Score: 0.3533
        # [근무시간 결측치 평균값] XGBoost + CatBoost MAE: 0.1739 / R2 Score: 0.3361
        # [근무시간 결측치 0] XGBoost + CatBoost MAE: 0.1639 / R2 Score: 0.3497
        df['mean_working'] = 0 # 오히려 근무를 하지 않는다고 판단하면 MAE 점수가 올라간다.

    # 범주형 변수 결측치 처리
    # medical_history, family_medical_history, edu_level 변수의 결측치 처리
    # medical_history, family_medical_history : 병력 데이터를 나타내서 해당 없는 것을 뜻하기 위해 None이라는 것으로 새로 채우기
    for col in ['medical_history', 'family_medical_history']:
        if col in df.columns:
            df[col] = df[col].fillna('None') # None : 병력이 없다.

    # edu_level : 학력 데이터로 아예 응답을 하지 않은 사람도 구분을 하기 위해서 UnKnown으로 새로 채우기
    if 'edu_level' in df.columns:
        df['edu_level'] = df['edu_level'].fillna('Unknown')

    # 방어적 프로그래밍(Defensive Programming) : 알 수 없는 데이터에 대비
    # 수치형 변수는 결측치를 중앙값으로 처리
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 범주형 변수 결측치는 최빈값으로 처리
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0]) # 최빈값이 여러 개일 수 있으므로 첫 번째 값(.iloc[0])을 명시적으로 가져옴.

    return df

# 피셍 변수 생성
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # BMI (체질량지수) 생성: 몸무게(kg) / (키(m)의 제곱)
    if 'weight' in df.columns and 'height' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
                              
    return df

# StandardScaler를 통해 숫자 단위의 차이로 인해 모델에 영향을 주는 것을 방지하기 위해
def scale_numerical(df: pd.DataFrame, is_train=True, scaler=None, target_col='stress_score') -> Tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()

    # 실전 예측(Test)인데 스케일러 도구를 안 가져왔다면 에러를 띄웁니다.
    if not is_train and scaler is None:
        raise ValueError("테스트 데이터를 변환하려면 학습 때 만든 scaler가 필요합니다!")
        
    # 학습(Train) 중일 때만 새 스케일러 도구를 포장지에서 꺼냅니다.
    if is_train:
        scaler = StandardScaler()

    
    # 정답지(target_col)와 0/1로 이루어진 이진 변수는 스케일링에서 제외
    num_cols = df.select_dtypes(include=['number']).columns
    cols_to_scale = [c for c in num_cols if c not in [target_col]]
    
    if len(cols_to_scale) > 0:
        if is_train:
            # 학습할 때는 기준을 세우고(fit) + 변환(transform)
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        else:
            # 실전에서는 학습 때 만든 기준을 그대로 써서 변환(transform)
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
    return df, scaler

# Label Encoder를 통해 문자열을 숫자로 변경
# encoders라는 복수형을 쓴 이유 : 데이터내에 범주형인 데이터가 여러개 이므로 각각에 대한 label encoder가 생성되기 때문에 복수형으로 사용
# def encode_categorical(df: pd.DataFrame, is_train=True, encoders=None) -> Tuple[pd.DataFrame, LabelEncoder]:
#     df = df.copy()
    
#     if encoders is None:
#         encoders = {}

#     cat_cols = df.select_dtypes(include=['object']).columns
    
#     for col in cat_cols:
#         if is_train:
#             le = LabelEncoder()
#             # 1. 문자를 숫자로 변환
#             # 2. 정수형(int)으로 변경
#             df[col] = le.fit_transform(df[col].astype(str)).astype(int)
#             encoders[col] = le
#         else:
#             if col in encoders:
#                 le = encoders[col]
#                 # 실전(Test) 데이터에 처음 보는 단어가 등장하면 에러가 나므로,
#                 # 학습 때 보았던 가장 흔한 값(le.classes_[0])으로 덮어씌우는 방어적 코드
#                 df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
#                 df[col] = le.transform(df[col]).astype(int)
        
#     return df, encoders

# 범주형 데이터 인코딩 (순서형은 map, 명목형은 One-Hot)
def encode_categorical(df: pd.DataFrame, is_train=True, train_columns=None) -> Tuple[pd.DataFrame, list]:
    df = df.copy()
    
    # 순서가 있는 변수 - 직접 딕셔너리로 매핑
    mapping_dicts = {
        'activity': {"light": 0, "moderate": 1, "intense": 2},
        'smoke_status': {"non-smoker": 0, "ex-smoker": 1, "current-smoker": 2},
        'edu_level': {'Unknown': 0, 'high school diploma': 1, 'bachelors degree': 2, 'graduate degree': 3},
        'sleep_pattern': {'sleep difficulty': 0, 'normal': 1, 'oversleeping': 2},
        'gender': {"F": 0, "M": 1}
    }
    
    for col, d in mapping_dicts.items():
        if col in df.columns:
            df[col] = df[col].map(d)

    # 순서가 없는 변수 - One-Hot 인코딩
    nominal_cols = ['medical_history', 'family_medical_history']
    existing_nominals = [col for col in nominal_cols if col in df.columns]
    
    if existing_nominals:
        df = pd.get_dummies(df, columns=existing_nominals, prefix=existing_nominals, dtype=int)

    # Test 데이터에 없는 One-Hot 컬럼 방어
    if is_train:
        train_columns = df.columns.tolist()
    else:
        # 실전(Test) 데이터에 병력(예: None)이 없어서 컬럼이 안 생겼다면 0으로 채워넣음
        for col in train_columns:
            if col not in df.columns:
                df[col] = 0
        # 컬럼 순서를 Train과 완벽히 똑같이 맞춤
        df = df[train_columns]

    return df, train_columns

# 전처리 실행 함수
def run_preprocessing(
    df: pd.DataFrame, 
    is_train: bool = True,
    scaler: StandardScaler = None,
    train_columns: list = None
    ) -> Tuple[pd.DataFrame, StandardScaler, list]:

    df = handle_missing_values(df)
    print("결측치 처리 완료")
    
    df = create_features(df)
    print("파생 변수 생성 완료")
    
    df, scaler = scale_numerical(df, is_train, scaler)
    print("수치형 변수 스케일링 완료")

    df, train_columns = encode_categorical(df, is_train, train_columns)
    print("범주형 변수 라벨 인코딩 완료")

    return df, scaler, train_columns