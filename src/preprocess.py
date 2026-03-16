import pandas as pd
import numpy as np

# 결측치 처리
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치(빈 값) 처리
    """
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

def run_preprocessing(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    전처리 실행
    """

    df = handle_missing_values(df)
    print("결측치 처리")

    return df