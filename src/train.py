import os
import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 전처리 모듈 불러오기
from preprocess import run_preprocessing

def load_config(config_path: str = '../configs/config.yaml'):
    """
    config.yaml 파일 로드 - 설정 정보 로드
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("train.py 실행")
    # 설정 및 데이터 로드
    config = load_config()

    # 데이터 경로
    raw_dir = config['data']['raw_dir']
    train_file = config['data']['train_file']
    train_path = os.path.join('..', raw_dir, train_file)
    train_df = pd.read_csv(train_path)

    # 샘플별 고유 ID 컬럼 제거
    id_col = config['features']['id_col']
    if id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])

    # 전처리
    processed_df = run_preprocessing(train_df)

    # 학습 데이터 준비 (각 데이터와 결과 분리)
    target_col = config['features']['target']
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['train']['test_size'], 
        random_state=config['model']['params']['random_state']
    )

    # 모델 학습
    print(f"\n모델 학습 시작 (알고리즘: {config['model']['type']})")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # LightGBM 최신 버전에 맞춘 조기 종료(Early Stopping) 콜백 적용
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params=config['model']['params'],
        train_set=train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    # 성능 평가
    y_pred = model.predict(X_val)
    
    # MAE 계산으로 변경
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("\n================================")
    print(f"모델 최종 성능 평가")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - R2 Score: {r2:.4f}")
    print("================================\n")

if __name__ == "__main__":
    main()