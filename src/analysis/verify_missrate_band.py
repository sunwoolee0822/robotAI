"""
verify_missrate_band.py
========================
`missing_rate.subject_missrate_band()`(작업 0821 C — LMM 입력의 high/low 필터에 씀)가
`grouping.split_low_high()`가 학습 데이터 자체를 나눌 때 쓴 low/high 분류
(DATA_ROOT/mtm/{target}/group_membership.json)와 subject 단위로 일치하는지 진단한다
(교수님 확인 요청, 0821).

구현 이력 (0821):
  1차: missing_rate.py 자체(accumulate()/win_miss, raw slot 단위 — 저빈도 feature도 주기
      보정 없이 슬롯 단위로만 셈) -> 2789명 중 148명(5.3%) 불일치.
  2차: grouping.compute_sample_missrate 기반으로 바꾸되 소스를 CoFormer packed baseline
      (packed->dense 재구성 경유)에 둠 -> 162명(5.8%)으로 오히려 더 벌어짐. 즉 결측률
      정의 차이가 아니라 packed<->dense 왕복 자체가 값을 바꾼다는 뜻이었다.
  3차(현재): 재구성을 거치지 않고 group_membership.json을 만든 바로 그 MTM baseline
      dense npz에서 grouping.compute_sample_missrate로 직접 계산 -> grouping.
      split_low_high와 완전히 같은 코드가 완전히 같은 데이터에 돌아 diff=0이 수학적으로
      보장됨. 이 스크립트는 그걸 실제로 재확인하는 회귀 가드다(diff!=0이면 어딘가 코드가
      틀렸다는 뜻).

사용법:
  python -m src.analysis.verify_missrate_band --target phq9
"""
import json
from pathlib import Path

from .. import paths
from .missing_rate import subject_missrate_band


def compare_band_to_membership(target, band_source=None, membership_path=None):
    """반환: dict(n_total, n_matched, n_mismatch, n_not_in_membership, mismatches=[...])."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target!r}")
    band_source = Path(band_source) if band_source else (
        paths.DATA_ROOT / "mtm" / target / "datasets" / "baseline")
    membership_path = Path(membership_path) if membership_path else (
        paths.DATA_ROOT / "mtm" / target / "group_membership.json")

    print(f"[*] band source (missing_rate.subject_missrate_band): {band_source}")
    print(f"[*] membership source (grouping.split_low_high): {membership_path}")
    band_df = subject_missrate_band(band_source, target)
    membership = json.loads(membership_path.read_text())
    low_set = set(map(str, membership["low_subjects"]))
    high_set = set(map(str, membership["high_subjects"]))
    print(f"  membership: low={len(low_set)}  high={len(high_set)}  "
          f"median_score={membership.get('median_score')}")
    print(f"  band: low={(band_df['missrate_band'] == 'low').sum()}  "
          f"high={(band_df['missrate_band'] == 'high').sum()}  n_subjects={len(band_df)}")

    mismatches = []
    for sid, row in band_df.iterrows():
        if sid in low_set:
            expected = "low"
        elif sid in high_set:
            expected = "high"
        else:
            mismatches.append({"subject_id": sid, "band": row["missrate_band"],
                               "membership": None, "kind": "not_in_membership"})
            continue
        if expected != row["missrate_band"]:
            mismatches.append({"subject_id": sid, "band": row["missrate_band"],
                               "membership": expected, "kind": "band_disagrees"})

    n_total = len(band_df)
    n_mismatch = len(mismatches)
    n_not_covered = sum(1 for m in mismatches if m["kind"] == "not_in_membership")
    print(f"\n[*] 비교 결과: {n_total - n_mismatch}/{n_total} 일치, {n_mismatch}개 불일치 "
          f"(그중 membership에 아예 없음: {n_not_covered}개)")
    if mismatches:
        print("  불일치 예시 (최대 20개):")
        for m in mismatches[:20]:
            print(f"    subject_id={m['subject_id']!r}  band={m['band']!r}  "
                  f"membership={m['membership']!r}  ({m['kind']})")
    else:
        print("  diff=0 — 두 소스가 완전히 일치함, band를 그대로 신뢰해도 됨.")

    return {"n_total": n_total, "n_matched": n_total - n_mismatch, "n_mismatch": n_mismatch,
            "n_not_in_membership": n_not_covered, "mismatches": mismatches}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["phq9", "gad7"], required=True)
    ap.add_argument("--band_source", type=str, default=None,
                    help="기본값: DATA_ROOT/mtm/{target}/datasets/baseline")
    ap.add_argument("--membership_path", type=str, default=None,
                    help="기본값: DATA_ROOT/mtm/{target}/group_membership.json")
    args = ap.parse_args()
    compare_band_to_membership(args.target, band_source=args.band_source,
                               membership_path=args.membership_path)
