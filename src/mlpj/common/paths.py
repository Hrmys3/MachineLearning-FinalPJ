"""源码检出的项目根目录；导入时不改变工作目录。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
