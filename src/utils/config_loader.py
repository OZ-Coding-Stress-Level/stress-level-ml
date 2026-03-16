import yaml

def load_config(config_path: str = '../configs/config.yaml'):
    """
    config.yaml 파일 로드 - 설정 정보 로드
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config