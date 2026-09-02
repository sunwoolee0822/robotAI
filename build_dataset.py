"""build_dataset.py
===================
결측 증강 데이터셋 조립 진입점. 원본 augment_mtm_group_aware.py + augment_coformer_
group_aware.py의 드라이버를 --format {mtm,coformer}로 통합. 실제 주입/조립 로직은
src/augment/build.py(augment_mtm/augment_coformer/combine_augmented_plus_high)에 있다
— train split에만 결측을 주입하고 val/test는 원본을 그대로 보존한다.

--aug로 configs/aug/*.yaml을 지정한다:
  none.yaml              증강 없음 (아무것도 하지 않음, 원본 low/high_missing을 그대로 사용)
  rate_only.yaml          rate-only 베이스라인 (ablation=uniform과 동일 메커니즘)
  group_aware_full.yaml   기본 group-aware 증강 (cell=Sex x AgeGroup, label=환자/대조)
  ablation_{uniform,age,age_sex}.yaml   W6 ablation 스윕

사용법:
  python build_dataset.py --aug configs/aug/group_aware_full.yaml --format mtm
  python build_dataset.py --aug configs/aug/ablation_uniform.yaml --format coformer
  python build_dataset.py --aug configs/aug/group_aware_full.yaml --format mtm --skip-combine
"""
import argparse

import yaml

from src.augment.build import augment_coformer, augment_mtm, combine_augmented_plus_high


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', required=True, help="configs/aug/*.yaml 경로")
    parser.add_argument("--format", choices=["mtm", "coformer"], required=True)
    parser.add_argument("--target", choices=["phq9", "gad7"], required=True,
                        help="target metric")
    parser.add_argument('--skip-combine', action='store_true',
                        help="low_missing_augmented만 만들고 augmented_plus_high 조립은 생략")
    return parser.parse_args()


def main():
    args = build_args()
    cfg = yaml.safe_load(open(args.aug))
    method = cfg.get('method', 'none')

    if method == "none":
        print(f"[*] method=none — 증강 없음, 원본 low_missing/high_missing을 그대로 사용합니다. "
              f"(아무 파일도 생성하지 않음)")
        return

    if method != 'group_aware':
        raise ValueError(f"Unknown aug method in {args.aug}: '{method}'")

    ablation = cfg['ablation']
    print(f"[*] format={args.format}  target={args.target}  ablation={ablation}  ({args.aug})")

    if args.format == 'mtm':
        out_dir = augment_mtm(ablation=ablation, target=args.target)
        print(f"[*] low_missing_augmented -> {out_dir}")
        if not args.skip_combine:
            combined = combine_augmented_plus_high(format="mtm", ablation=ablation, target=args.target)
            print(f"[*] augmented_plus_high -> {combined}")
    else:
        out_dir = augment_coformer(ablation=ablation, target=args.target)
        print(f"[*] low_missing_augmented -> {out_dir}")
        if not args.skip_combine:
            combined = combine_augmented_plus_high(
                format='coformer', ablation=ablation, target=args.target)
            print(f"[*] augmented_plus_high -> {combined}")

    print("Done.")


if __name__ == '__main__':
    main()
