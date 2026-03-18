import os
import pandas as pd
import numpy as np
import joblib

from preprocess import run_preprocessing
from utils.config_loader import load_config
from datetime import datetime # 시간을 가져오기 위한 모듈

def main():
    config = load_config()

    # 경로 설정
    raw_dir = config['data']['raw_dir']
    test_file = config['data']['test_file']
    sub_file = config['data']['submission_file']
    output_dir = config['output']['dir']
    sub_dir = config['output']['submissions']
    
    test_path = os.path.join('..', raw_dir, test_file)
    sub_path = os.path.join('..', raw_dir, sub_file)
    model_path = os.path.join('..', output_dir, 'ensemble_models.pkl')
    save_dir = os.path.join('..', output_dir, sub_dir)

    # 데이터 가져오기
    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(sub_path)

    # ID 제거
    # TODO: - 중복되는 코드를 추후 utils에 별도 파일로 생성
    id_col = config['features']['id_col']
    if id_col in test_df.columns:
        test_df = test_df.drop(columns=[id_col])

    # 모델 및 전처리 도구 가져오기
    saved_tools = joblib.load(model_path)
    # lgb_models = saved_tools['lgb_models']
    xgb_models = saved_tools['xgb_models']
    cb_models = saved_tools['cb_models']
    train_columns = saved_tools['train_columns']
    scaler = saved_tools['scaler']

    processed_test, _, _ = run_preprocessing(
        test_df, 
        is_train=False,
        scaler=scaler,
        train_columns=train_columns
    )

    target_col = config['features']['target']
    if target_col in processed_test.columns:
        processed_test = processed_test.drop(columns=[target_col])

    # lgb_preds = np.zeros(len(processed_test))
    xgb_preds = np.zeros(len(processed_test))
    cb_preds = np.zeros(len(processed_test))
    
    # # LightGBM 5개의 의견 취합
    # for model in lgb_models:
    #     lgb_preds += model.predict(processed_test)
    # lgb_preds /= len(lgb_models)
    
    # XGBoost 5개의 의견 취합
    for model in xgb_models:
        xgb_preds += model.predict(processed_test)
    xgb_preds /= len(xgb_models)

    # CatBoost 5개의 의견 취합
    for model in cb_models:
        cb_preds += model.predict(processed_test)
    cb_preds /= len(cb_models)

    # 제출 양식(sample_submission)의 타겟 컬럼에 예측값 덮어씌우기
    final_predictions = (xgb_preds * 0.0610) + (cb_preds * 0.9390)
    
    target_col = config['features']['target']
    submission_df[target_col] = final_predictions
    
    # 현재 시간으로 다이내믹한 파일명 생성 및 저장
    os.makedirs(save_dir, exist_ok=True) # submissions 폴더가 없으면 알아서 만듦
    
    # datetime.now()를 포맷팅하여 "년월일시분" 문자열 생성 (예: 202603161315)
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    final_filename = f"submission_{current_time}.csv"
    final_filepath = os.path.join(save_dir, final_filename)
    
    # index=False 로 설정해야 파일 맨 왼쪽에 쓸데없는 0, 1, 2, 3 번호가 붙지 않음.
    submission_df.to_csv(final_filepath, index=False)
    
    print("\n========================================")
    print("최종 제출 파일이 생성 완료")
    print("\n========================================")

if __name__ == "__main__":
    main()