import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score

# 전처리 모듈 불러오기
from preprocess import run_preprocessing
# config 호출 함수 불러오기
from utils.config_loader import load_config

def main():
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
    processed_df, scaler, encoders  = run_preprocessing(train_df)

    # 학습 데이터 준비 (각 데이터와 결과 분리)
    target_col = config['features']['target']
    X = processed_df.drop(columns=[target_col]).reset_index(drop=True)
    y = processed_df[target_col].reset_index(drop=True)

    # 기존 K-Fold 교차 검증 5번 분할로 교차 검증을 진행했다.
    # MAE: 0.1777
    # R2 Score: 0.3262
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, 
    #     test_size=config['train']['test_size'], 
    #     random_state=config['model']['params']['random_state']
    # )

    # 5-Fold 교차 검증을 통해 모델을 5개를 얻고 저장한다.
    # K-Fold 설정 (5등분) -> 오히려 점수가 떨어짐.
    # OOF MAE: 0.1906
    # OOF R2 Score: 0.2597
    n_splits = config['train']['n_splits']  # yaml에 설정한 5 가져오기
    random_state = config['model']['lgb_params']['random_state']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 5개의 모델을 담을 빈 리스트와, 검증 점수를 기록할 리스트 준비
    # lgb_models = []
    xgb_models = []
    cb_models = []
    oof_predictions = np.zeros(len(X)) # Out-Of-Fold: 전체 데이터에 대한 예측값 기록장

    print(f"\n총 {n_splits}번을 두 모델이 동시에 학습.")
    print("=" * 50)

    # 5번 반복하며 5개의 모델 훈련하기
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"🔄 Fold {fold + 1} / {n_splits} 학습 중...")
        
        # 5조각 중 4조각은 학습용(train), 1조각은 검증용(val)으로 나눔
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # 모델 학습
        print(f"\n모델 학습 시작 (알고리즘: {config['model']['type']})")

        train_data_lgb = lgb.Dataset(X_train, label=y_train)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
        
        # LightGBM 학습 -> 2026.03.18 LightGBM이 들어가게되면서 데이터의 결과가 좋은 결과를 가져오지 않음.
        # model_lgb = lgb.train(
        #     params=config['model']['lgb_params'], # lgb 전용 파라미터
        #     train_set=train_data_lgb,
        #     valid_sets=[train_data_lgb, val_data_lgb],
        #     valid_names=['train', 'valid'],
        #     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        # )
        # lgb_models.append(model_lgb)
        # val_pred_lgb = model_lgb.predict(X_val)

        # XGBoost
        model_xgb = XGBRegressor(
            **config['model']['xgb_params'], # xgb 전용 파라미터 압축 해제
            early_stopping_rounds=50         # 조기 종료 설정
        )
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        xgb_models.append(model_xgb)
        val_pred_xgb = model_xgb.predict(X_val)

        # CatBoost
        model_cb = CatBoostRegressor(
            **config['model']['cb_params'],
            early_stopping_rounds=50,
            verbose=False, # 훈련 로그 숨기기
            allow_writing_files=False # 기록파일 생성방지
        )
        model_cb.fit(
            X_train, y_train, 
            eval_set=(X_val, y_val), 
            use_best_model=True
        )

        cb_models.append(model_cb)
        val_pred_cb = model_cb.predict(X_val)

        # LightGBM MAE: 0.1911 / R2 Score: 0.2556
        # XGBoost MAE: 0.1752 / R2 Score: 0.3302
        # CatBoost MAE: 0.1731 / R2 Score: 0.3231

        # LightGBM + XGBoost MAE: 0.1811 / R2 Score: 0.3178
        # LightGBM + CatBoost MAE: 0.1790 / R2 Score: 0.3261
        # XGBoost + CatBoost MAE: 0.1710 / R2 Score: 0.3533
        # LightGBM + XGBoost + CatBoost MAE: 0.1762 / R2 Score: 0.3422
        fold_ensemble_pred = (val_pred_xgb + val_pred_cb) / 2

        oof_predictions[val_idx] = fold_ensemble_pred
        fold_mae = mean_absolute_error(y_val, fold_ensemble_pred)
        print(f"   └─ 앙상블 MAE: {fold_mae:.4f}")

    # 전체 모델 통합 성능 평가 (Out-Of-Fold 평가)
    total_mae = mean_absolute_error(y, oof_predictions)
    total_r2 = r2_score(y, oof_predictions)
    
    print("\n" + "=" * 40)
    print(f"5-Fold 앙상블 최종 OOF 성능 평가")
    print(f"   - OOF MAE: {total_mae:.4f}")
    print(f"   - OOF R2 Score: {total_r2:.4f}")
    print("=" * 40 + "\n")

    # 저장 (모델 1개가 아니라, 5개가 담긴 리스트를 통째로 저장합니다!)
    output_dir = os.path.join('..', config['output']['dir'])
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'ensemble_models.pkl')
    joblib.dump({
        # 'lgb_models': lgb_models, # 5개의 LightGBM
        'xgb_models': xgb_models, # 5개의 XGBoost
        'cb_models': cb_models, # 5개의 CatBoost
        'encoders': encoders,
        'scaler': scaler
    }, model_path)

    print(f"앙상블 모델 저장: {model_path}")

if __name__ == "__main__":
    main()