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
        df['mean_working'] = median_val

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
    print(f"num_cols = {num_cols}")
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
def encode_categorical(df: pd.DataFrame, is_train=True, encoders=None) -> Tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    
    if encoders is None:
        encoders = {}

    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            # 1. 문자를 숫자로 변환
            # 2. 정수형(int)으로 변경
            df[col] = le.fit_transform(df[col].astype(str)).astype(int)
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                # 실전(Test) 데이터에 처음 보는 단어가 등장하면 에러가 나므로,
                # 학습 때 보았던 가장 흔한 값(le.classes_[0])으로 덮어씌우는 방어적 코드
                df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col]).astype(int)
        
    return df, encoders

# 전처리 실행 함수
def run_preprocessing(
    df: pd.DataFrame, 
    is_train: bool = True,
    scaler: StandardScaler = None,
    encoders: LabelEncoder = None
    ) -> Tuple[pd.DataFrame, StandardScaler, LabelEncoder]:

    df = handle_missing_values(df)
    print("결측치 처리 완료")
    
    df = create_features(df)
    print("파생 변수(BMI) 생성 완료")
    
    df, scaler = scale_numerical(df, is_train, scaler)
    print("수치형 변수 스케일링 완료")

    df, encoders = encode_categorical(df, is_train, encoders)
    print("범주형 변수 라벨 인코딩 완료")

    return df, scaler, encoders