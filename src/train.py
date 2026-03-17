import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
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
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]

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
    random_state = config['model']['params']['random_state']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 5개의 모델을 담을 빈 리스트와, 검증 점수를 기록할 리스트 준비
    models = []
    oof_predictions = np.zeros(len(X)) # Out-Of-Fold: 전체 데이터에 대한 예측값 기록장

    print(f"\n총 {n_splits}개의 쌍둥이 모델을 훈련합니다.")
    print("=" * 40)

    # 5번 반복하며 5개의 모델 훈련하기
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"🔄 Fold {fold + 1} / {n_splits} 학습 중...")
        
        # 5조각 중 4조각은 학습용(train), 1조각은 검증용(val)으로 나눔
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # 모델 학습
        print(f"\n모델 학습 시작 (알고리즘: {config['model']['type']})")

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # LightGBM 최신 버전에 맞춘 조기 종료(Early Stopping) 콜백 적용
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
        
        # 모델 훈련
        model = lgb.train(
            params=config['model']['params'],
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # 훈련된 모델을 리스트에 저장
        models.append(model)
        
        # 이번 Fold의 검증용 데이터 예측값 기록
        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred
        
        # 이번 Fold의 MAE 출력
        fold_mae = mean_absolute_error(y_val, val_pred)
        print(f"   └─ Fold {fold + 1} MAE: {fold_mae:.4f}")

    # 전체 모델 통합 성능 평가 (Out-Of-Fold 평가)
    total_mae = mean_absolute_error(y, oof_predictions)
    total_r2 = r2_score(y, oof_predictions)
    
    print("\n" + "=" * 40)
    print(f"🏆 5-Fold 앙상블 최종 OOF 성능 평가")
    print(f"   - OOF MAE: {total_mae:.4f}")
    print(f"   - OOF R2 Score: {total_r2:.4f}")
    print("=" * 40 + "\n")

    # 저장 (모델 1개가 아니라, 5개가 담긴 리스트를 통째로 저장합니다!)
    output_dir = os.path.join('..', config['output']['dir'])
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'lgb_model.pkl')

    joblib.dump({
        'models': models,  # ⭐️ 이름이 model에서 models(리스트)로 바뀜!
        'encoders': encoders,
        'scaler': scaler
    }, model_path)

    print(f"💾 5쌍둥이 모델과 전처리 도구가 안전하게 저장되었습니다: {model_path}")

if __name__ == "__main__":
    main()