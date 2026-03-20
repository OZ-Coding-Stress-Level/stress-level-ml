import os
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from utils.config_loader import load_config
from preprocess import run_preprocessing

def objective(trial, X, y, config):
    """
    XGBoost 전용 K-Fold 튜닝 함수입니다.
    """
    # XGBoost에 맞는 파라미터 탐색 범위 설정
    # (XGBoost는 트리가 너무 깊어지면 과적합이 심해지므로 max_depth를 낮게 잡습니다)
    param = {
        'objective': 'reg:absoluteerror', # MAE를 최소화하는 XGBoost 설정
        'eval_metric': 'mae',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10), # LightGBM보다 얕게 설정!
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10) # XGBoost의 핵심 과적합 방지 파라미터
    }

    n_splits = config['train']['n_splits']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_maes = []

    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # XGBoost 모델 생성 및 학습 (조기 종료 포함)
        model = XGBRegressor(
            **param, # 언패킹 활용
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        fold_maes.append(mae)

    # 5번 중 '평균 MAE' 반환
    return np.mean(fold_maes)

def main():
    print("🎯 XGBoost 전용 K-Fold Optuna 튜닝 시작...")
    config = load_config()

    # 데이터 로드 및 전처리
    train_path = os.path.join('..', config['data']['raw_dir'], config['data']['train_file'])
    train_df = pd.read_csv(train_path)

    id_col = config['features']['id_col']
    if id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])

    processed_df, _, _ = run_preprocessing(train_df, is_train=True)

    target_col = config['features']['target']
    X = processed_df.drop(columns=[target_col]).reset_index(drop=True)
    y = processed_df[target_col].reset_index(drop=True)

    study = optuna.create_study(direction=config['tune']['direction'])
    
    print(f"총 {config['tune']['n_trials']}번의 파라미터 탐색을 진행합니다.")
    
    study.optimize(
        lambda trial: objective(trial, X, y, config), 
        n_trials=config['tune']['n_trials']
    )

    print("\n==========================================")
    print(f"XGBoost 5-Fold 최고 성능 (OOF MAE): {study.best_value:.4f}")
    
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("==========================================\n")

if __name__ == "__main__":
    main()