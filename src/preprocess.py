import pandas as pd
import numpy as np
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
def scale_numerical(df, target_col='stress_score'):
    df = df.copy()
    scaler = StandardScaler()
    
    # 정답지(target_col)와 0/1로 이루어진 이진 변수는 스케일링에서 제외
    num_cols = df.select_dtypes(include=['number']).columns
    cols_to_scale = [c for c in num_cols if c not in [target_col]]
    
    if len(cols_to_scale) > 0:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
    return df

# Label Encoder를 통해 문자열을 숫자로 변경
def encode_categorical(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df

def run_preprocessing(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    전처리 실행
    """

    df = handle_missing_values(df)
    print("결측치 처리 완료")
    
    df = create_features(df)
    print("파생 변수(BMI) 생성 완료")
    
    df = scale_numerical(df)
    print("수치형 변수 스케일링 완료")

    df = encode_categorical(df)
    print("범주형 변수 라벨 인코딩 완료")

    return df