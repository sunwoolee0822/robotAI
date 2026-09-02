"""외부 서브모듈(external/coformer, external/mtm) 코드를 import하기 위한 컨텍스트 매니저.

CoFormer의 utils.config.Config는 cfg 파일 glob이 cwd 상대이고, MTM은 벗은 최상위
import(`from mtm.mtm import MTM` 등, 패키지 프리픽스 없이 `tasks.*`/`config.*`를 참조)라
둘 다 sys.path 추가만으로는 부족하고 chdir까지 필요하다.
"""
import os
import sys
from contextlib import contextmanager


@contextmanager
def vendor_ctx(root):
    old = os.getcwd()
    sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old)
        sys.path.remove(str(root))
