import os
import pandas as pd
import numpy as np
import optuna
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from utils.config_loader import load_config
from preprocess import run_preprocessing

def objective(trial, X, y, config):
    # SVR 핵심 파라미터 탐색 범위 설정
    param = {
        'kernel': 'rbf', # 정형 데이터에서 가장 성능이 좋은 가우시안 커널
        'C': trial.suggest_float('C', 0.1, 100.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
    }

    n_splits = config['train']['n_splits']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []

    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # SVR 모델 생성 및 학습
        model = SVR(**param)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        fold_maes.append(mae)

    return np.mean(fold_maes)

def main():
    config = load_config()

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
    
    # SVR은 트리 모델보다 학습이 오래 걸릴 수 있으므로 n_trials를 조금 줄이는 것도 방법입니다.
    print(f"총 {config['tune']['n_trials']}번의 탐색을 진행합니다.")
    study.optimize(lambda trial: objective(trial, X, y, config), n_trials=config['tune']['n_trials'])

    print("\n==========================================")
    print(f"SVR 5-Fold 최고 성능 (OOF MAE): {study.best_value:.4f}")

    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("==========================================\n")

if __name__ == "__main__":
    main()