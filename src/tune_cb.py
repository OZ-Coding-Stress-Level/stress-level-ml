import os
import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from utils.config_loader import load_config
from preprocess import run_preprocessing

def objective(trial, X, y, config):
    """
    CatBoost 전용 K-Fold 튜닝 함수입니다.
    """
    # CatBoost에 맞는 파라미터 탐색 범위 설정
    # CatBoost는 트리가 깊어지면 학습 시간이 기하급수적으로 늘어나므로 depth를 4~10 사이로 제한
    param = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': 42,
        'verbose': False, # 화면 출력 끄기
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0), # CatBoost의 핵심 규제 파라미터
        'subsample': trial.suggest_float('subsample', 0.6, 1.0)
    }

    n_splits = config['train']['n_splits']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []

    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # CatBoost 모델 생성 및 학습
        model = CatBoostRegressor(**param)
        
        # early_stopping_rounds를 fit 함수에 직접 삽입
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=False
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        fold_maes.append(mae)

    # 평균 MAE 반환
    return np.mean(fold_maes)

def main():
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
    print(f"CatBoost 최고 성능 (OOF MAE): {study.best_value:.4f}")
    
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("==========================================\n")

if __name__ == "__main__":
    main()