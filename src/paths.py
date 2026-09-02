"""중앙 경로 설정. 하드코딩 절대경로 대신 여기서 DATA_ROOT를 가져다 쓴다."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 원본 데이터/중간 산출물(설문·센서 캐시, npz/npy 데이터셋). 기본은 repo-local data/
# (.gitignore로 제외됨) — 실제 데이터가 다른 디스크에 있으면 환경변수로 오버라이드.
DATA_ROOT = Path(os.environ.get("ROBOTAI_DATA_ROOT", REPO_ROOT / "data"))

# 분석/학습 산출물(그림, 표, 체크포인트 로그 등).
OUTPUTS_ROOT = Path(os.environ.get("ROBOTAI_OUTPUTS_ROOT", REPO_ROOT / "outputs"))

EXTERNAL_COFORMER = REPO_ROOT / "external" / "coformer"
EXTERNAL_MTM = REPO_ROOT / "external" / "mtm"
