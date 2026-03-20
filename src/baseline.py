import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 1. 데이터 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# 2. 전처리 함수 정의
def preprocess_data(df):
    # 불필요한 ID 제거
    df = df.drop(columns=['ID'])
    
    # 수치형 결측치는 중앙값으로 대체
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 범주형 결측치는 'Unknown'으로 대체
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].fillna('Unknown')
    
    return df

# 전처리 적용
train_df = preprocess_data(train)
test_df = preprocess_data(test)

# 타겟 변수 분리
X = train_df.drop(columns=['stress_score'])
y = train_df['stress_score']
target_test = test_df.copy()

# 3. 범주형 변수 레이블 인코딩
# Train과 Test에 동일한 기준을 적용하기 위해 합쳐서 인코딩하거나 
# Test의 새로운 카테고리에 대응할 수 있도록 처리합니다.
combined = pd.concat([X, target_test], axis=0)
le = LabelEncoder()

for col in X.select_dtypes(include=['object']).columns:
    combined[col] = le.fit_transform(combined[col].astype(str))

X_encoded = combined[:len(X)]
test_encoded = combined[len(X):]

# 4. 모델 학습 (RandomForest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

# 5. 예측 및 저장
predictions = model.predict(test_encoded)
submission['stress_score'] = predictions
submission.to_csv('submission_baseline.csv', index=False)

print("제출 파일 'submission_baseline.csv'가 생성되었습니다!")