from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 子目录（您已经手动创建好了）
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

# 模型文件完整路径
MODEL_PATH = MODEL_DIR / "params.pth"