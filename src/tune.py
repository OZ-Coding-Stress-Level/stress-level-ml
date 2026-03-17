import os
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error

from utils.config_loader import load_config
from preprocess import run_preprocessing

# Optuna가 1번 실험(Trial)을 할 때마다 실행되는 함수
def objective(trial, X, y, config):
    # Optuna가 이번 실험에서 시도해 볼 파라미터 조합을 출력
    param = {
        'objective': 'mae',
        'metric': 'mae',
        'random_state': 42,
        'verbose': -1,
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500), # 300~1500 사이에서 추천
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True), # 좁은 보폭 탐색
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # 데이터의 몇 %만 쓸지 (과적합 방지)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0) # 피처의 몇 %만 쓸지
    }

    # K-Fold 설정
    n_splits = config['train']['n_splits']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_maes = [] # 5번의 모의고사 점수를 담을 리스트

    # 5번 반복하며 검증
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # 뽑아준 파라미터로 모델을 학습시킵니다.
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 튜닝 중에는 화면이 지저분해지니 verbose=False로 로그를 끕니다.
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        model = lgb.train(
            params=param,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        # MAE를 계산해서 Optuna에게 보고합니다.
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred) # Optuna는 이 값을 최소화(minimize)하기 위해 다음 파라미터를 똑똑하게 조절합니다.
        fold_maes.append(mae)

    # 5번의 '평균 점수'를 Optuna에게 최종적으로 전달
    return np.mean(fold_maes) 

# Optuna를 통해 하이퍼 파라미터 튜닝을 진행.
# 별도로 진행 후 config.yaml에 해당 파라미터 설정
def main():
    config = load_config()

    # 데이터 로드 및 전처리 (train.py와 동일)
    train_path = os.path.join('..', config['data']['raw_dir'], config['data']['train_file'])
    train_df = pd.read_csv(train_path)

    id_col = config['features']['id_col']
    if id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])

    processed_df, _, _ = run_preprocessing(train_df, is_train=True)

    target_col = config['features']['target']
    X = processed_df.drop(columns=[target_col]).reset_index(drop=True)
    y = processed_df[target_col].reset_index(drop=True)

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=config['train']['test_size'], random_state=42
    # )

    # Optuna 스터디(실험) 생성 및 실행
    study = optuna.create_study(direction=config['tune']['direction'])
    
    print(f"총 {config['tune']['n_trials']}번의 파라미터 탐색을 진행")

    # objective 함수에 데이터를 전달하기 위해 람다(lambda) 함수 사용
    study.optimize(
        lambda trial: objective(trial, X, y, config), 
        n_trials=config['tune']['n_trials']
    )

    # 튜닝 완료 후 가장 성적이 좋았던 결과 출력
    print("\n==========================================")
    print(f"최고 성능 (가장 낮은 MAE): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("==========================================\n")

if __name__ == "__main__":
    main()