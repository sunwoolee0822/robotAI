#!/usr/bin/env bash
# 외부 서브모듈(CoFormer, MTM)을 고정 커밋으로 준비하고 패치를 적용한다.
# 두 번 실행해도 안전 (멱등).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COFORMER_DIR="external/coformer"
COFORMER_COMMIT="69261dbd2578994f758182fdce8ef36dc2205ed6"
MTM_DIR="external/mtm"
MTM_COMMIT="5cc68179b98647a836aaf75c265d36e777ba56ca"
PATCH="patches/coformer-fp16-maskfill.patch"

echo "==> submodule init/update"
git submodule update --init --recursive

checkout_pinned() {
    local dir="$1" commit="$2"
    local current
    current="$(git -C "$dir" rev-parse HEAD)"
    if [ "$current" = "$commit" ]; then
        echo "==> $dir already at $commit"
    else
        echo "==> $dir checkout $commit"
        git -C "$dir" fetch --quiet origin "$commit" || true
        git -C "$dir" checkout --quiet "$commit"
    fi
}

checkout_pinned "$COFORMER_DIR" "$COFORMER_COMMIT"
checkout_pinned "$MTM_DIR" "$MTM_COMMIT"

echo "==> applying $PATCH"
if git -C "$COFORMER_DIR" apply --reverse --check "../../$PATCH" 2>/dev/null; then
    echo "==> patch already applied, skipping"
elif git -C "$COFORMER_DIR" apply --check "../../$PATCH" 2>/dev/null; then
    git -C "$COFORMER_DIR" apply "../../$PATCH"
    echo "==> patch applied"
else
    echo "!! patch does not apply cleanly against $COFORMER_DIR@$COFORMER_COMMIT" >&2
    exit 1
fi

echo "==> setup.sh done"
