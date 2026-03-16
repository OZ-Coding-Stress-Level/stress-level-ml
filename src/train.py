import os
import pandas as pd
import yaml

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

    # 전처리
    processed_df = run_preprocessing(train_df)

if __name__ == "__main__":
    main()