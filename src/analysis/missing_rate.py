"""
missing_rate.py
===============
논문 figure 생성 (전처리 배열 기준: 72슬롯 창, 1시간 집계, 1일 stride)

  F3  주기성          시간축(A/B1/B2/C) + 요일축(D/E/F), 2행 4열
  F4  feature별 분포  violin + mean±SEM
  F5  누적분포        창 단위 결측률, low/high 컷 근거
  F6  그룹 비교       설문 인스턴스 단위, PHQ-9 / 성별 / 연령
  F8  동시결측        Spearman, grayscale
  A1  subject 분포    24 feature 전체 histogram (appendix)

실행
  python missing_rate.py [--data_root <배열폴더>] [--out_dir <출력>]
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, spearmanr

# =====================================================================
# 상수
# =====================================================================

GROUP_DEF = [
    ("A",  "band, no fill",        ["hr", "rr"]),
    ("B1", "band, no fill",        ["core_temp", "skin_temp"]),
    ("B2", "band PPG, no fill",    ["bp_sys", "bp_dia", "glucose"]),
    ("C",  "phone, no fill",       ["light_sensor"]),
    ("D",  "band+phone, fill 24",  ["spo2", "hrv", "proximity"]),
    ("E",  "band+phone, fill 24",  ["step", "distance", "screen_time", "wake_time",
                                    "sleep_time", "deep_sleep_time", "rem_sleep_time",
                                    "light_sleep_time", "total_sleep_time"]),
    ("F",  "self-report, fill 168", ["EMA_Anxiety", "EMA_Depression",
                                     "EMA_Sleep", "EMA_Stress"]),
]
_FEATURE_TO_GROUP = {f: g for g, _, fs in GROUP_DEF for f in fs}
_GROUP_NAME = {g: name for g, name, _ in GROUP_DEF}
_GROUP_ORDER = [g for g, _, _ in GROUP_DEF]
_GROUP_COLORS = {"A": "#e74c3c", "B1": "#5dade2", "B2": "#154360",
                 "C": "#9b59b6", "D": "#d35400", "E": "#27ae60", "F": "#34495e"}
_FEATURE_STYLE = {
    "hr": ("-", "o"), "rr": ((0, (6, 2)), "s"),
    "core_temp": ("-", "o"), "skin_temp": ((0, (6, 2)), "s"),
    "bp_sys": ("-", "o"), "bp_dia": ((0, (6, 2)), "s"), "glucose": ((0, (1, 2)), "D"),
    "light_sensor": ("-", "o"),
    "spo2": ("-", "o"), "hrv": ((0, (6, 2)), "s"), "proximity": ((0, (1, 2)), "D"),
    "step": ("-", "o"),
    "distance": ((0, (6, 2)), "s"),
    "screen_time": ((0, (1, 2)), "^"),
    "wake_time": ((0, (6, 2, 1, 2)), "D"),
    "sleep_time": ((0, (4, 1, 4, 3)), "v"),
    "deep_sleep_time": ((0, (6, 2, 1, 2, 1, 2)), "P"),
    "rem_sleep_time": ((0, (8, 2, 1, 2)), "X"),
    "light_sleep_time": ((0, (1, 1, 5, 2)), "*"),
    "total_sleep_time": ((0, (7, 2, 3, 2)), "h"),
    "EMA_Anxiety": ("-", "o"), "EMA_Depression": ((0, (6, 2)), "s"),
    "EMA_Sleep": ((0, (1, 2)), "^"), "EMA_Stress": ((0, (6, 2, 1, 2)), "D"),
}

# F3 패널 구성
_HOUR_PANELS = [("A", ["hr", "rr"]),
                ("B1", ["core_temp", "skin_temp"]),
                ("B2", ["bp_sys", "bp_dia", "glucose"]),
                ("C", ["light_sensor"])]
_DOW_PANELS = [("D", GROUP_DEF[4][2]),
               ("E", GROUP_DEF[5][2]),
               ("F", GROUP_DEF[6][2])]

_HOURLY_FEATS = [f for _, fs in _HOUR_PANELS for f in fs]
_WINDOW_DAYS = 3
_SLOTS_PER_HOUR_PER_WINDOW = _WINDOW_DAYS
_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# F6 연령 구간. 20대만 따로 두면 환자 셀이 30명대로 떨어져 결론을 낼 수 없어
# 20-39 로 묶었다.
_AGE_BINS = [19, 39, 49, 59, 69, 120]
_AGE_LABELS = ["20-39", "40-49", "50-59", "60-69", "70+"]


# =====================================================================
# 집계
# =====================================================================

def canonical(col):
    """배열 컬럼명 -> feature 이름 (분당 센서는 _mean 접미사)."""
    return col[:-5] if col.endswith("_mean") else col


def sorted_features(feats):
    return sorted(feats, key=lambda f: (_GROUP_ORDER.index(_FEATURE_TO_GROUP[f]), f))


def _window_dow(root: Path, n):
    """창별 시작 요일 (Mon=0). origins(창 시작일) + w_start_days(창 시작 offset).

    w_start_days 는 요일이 아니라 subject origin 기준 며칠째에 창이 시작하는지다.
    1970-01-01 이 목요일이므로 (일수 + 3) % 7 이 Mon=0 요일이 된다."""
    origins = np.array(json.loads((root / "origins.json").read_text()), dtype="datetime64[D]")
    wstart = np.array(json.loads((root / "w_start_days.json").read_text()), dtype=np.int64)
    assert len(origins) == len(wstart) == n
    start = origins + wstart.astype("timedelta64[D]")
    return ((start.astype(np.int64) + 3) % 7).astype(np.int16)


def accumulate(root: Path, chunk=4000):
    """time.npy(left-packed) 로부터 누적.

    obs_hour  subject x feature x 24   시각별 관측 슬롯 수
    obs_dow   subject x feature x 7    요일별 관측 슬롯 수
    day_dow   subject x 7              요일별 창-일(day) 수  (obs_dow 의 분모용)
    obs_slot  subject x feature        전체 관측 슬롯 수
    obs_day   subject x feature        관측된 날 수
    win_miss  window                   창 단위 전체 결측률   (F5, F6)
    feat_miss window x feature         창 x feature 결측률   (F8)
    """
    root = Path(root)
    cols = [canonical(c) for c in json.loads((root / "feature_columns.json").read_text())]
    subjects = [str(s) for s in json.loads((root / "subject_ids.json").read_text())]
    uniq = sorted(set(subjects))
    s_index = {s: i for i, s in enumerate(uniq)}
    srow = np.array([s_index[s] for s in subjects])

    time = np.load(root / "time.npy", mmap_mode="r")
    n, n_feat, n_slot = time.shape
    assert n_feat == len(cols) and n == len(subjects)
    n_subj = len(uniq)
    win_dow = _window_dow(root, n)

    obs_hour = np.zeros((n_subj, n_feat, 24), dtype=np.int64)
    obs_dow = np.zeros((n_subj, n_feat, 7), dtype=np.int64)
    day_dow = np.zeros((n_subj, 7), dtype=np.int64)
    obs_slot = np.zeros((n_subj, n_feat), dtype=np.int64)
    obs_day = np.zeros((n_subj, n_feat), dtype=np.int64)
    n_win = np.zeros(n_subj, dtype=np.int64)
    win_miss = np.zeros(n, dtype=np.float32)
    feat_miss = np.zeros((n, n_feat), dtype=np.float32)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        T = np.asarray(time[s:e])
        O = T >= 0
        rows = srow[s:e]
        wd0 = win_dow[s:e]

        np.add.at(n_win, rows, 1)
        np.add.at(obs_slot, (rows, slice(None)), O.sum(axis=2))
        win_miss[s:e] = 1.0 - O.sum(axis=(1, 2)) / float(n_feat * n_slot)
        feat_miss[s:e] = 1.0 - O.sum(axis=2) / float(n_slot)

        day = np.where(O, T // 1440, -1)
        changed = (day[:, :, 1:] != day[:, :, :-1]) & O[:, :, 1:]
        np.add.at(obs_day, (rows, slice(None)), O[:, :, 0] + changed.sum(axis=2))

        for d in range(_WINDOW_DAYS):
            np.add.at(day_dow, (rows, (wd0 + d) % 7), 1)

        hour = np.where(O, (T // 60) % 24, -1).astype(np.int16)
        dow = np.where(O, (wd0[:, None, None] + day) % 7, -1).astype(np.int16)
        for f in range(n_feat):
            hf, df_ = hour[:, f, :], dow[:, f, :]
            v = hf >= 0
            if not v.any():
                continue
            r = np.repeat(rows[:, None], n_slot, axis=1)[v]
            np.add.at(obs_hour, (r, f, hf[v].astype(np.int64)), 1)
            np.add.at(obs_dow, (r, f, df_[v].astype(np.int64)), 1)

    return dict(cols=cols, subjects=uniq, subjects_per_window=subjects, obs_hour=obs_hour,
                obs_dow=obs_dow, day_dow=day_dow, obs_slot=obs_slot, obs_day=obs_day,
                n_win=n_win, win_miss=win_miss, feat_miss=feat_miss, n_windows=n, n_slot=n_slot)


def subject_missrate_band(mtm_baseline_root, target):
    """subject별 low/high missrate band — grouping.split_low_high()가 group_membership.json
    을 만들 때 쓴 것과 **동일한 원본 데이터(MTM baseline dense npz)에 동일 계산**(mask=
    np.isnan(X) -> grouping.compute_sample_missrate -> subject로 평균 -> rank-based median
    split: 정렬 후 앞 절반=low)을 그대로 재현한다. 필터용 라벨일 뿐 LMM 요인이 아니다
    (작업 0821 C).

    구현 이력: 처음엔 missing_rate.py 자체(accumulate()/win_miss, raw slot 단위)로 계산했으나
    group_membership.json 대비 148/2789명(5.3%) 불일치(verify_missrate_band.py, 0821).
    grouping.compute_sample_missrate 기반으로 바꿨지만 소스를 CoFormer packed baseline(
    packed->dense 재구성 경유)으로 뒀더니 162/2789명(5.8%)로 오히려 더 벌어졌다 — 포맷
    차이가 원인이 아니라 packed<->dense 왕복 자체가 값을 살짝 바꾼다는 뜻. "학습 조건
    정의와 평가 표본 정의가 갈라지면 안 된다"는 요구를 만족하는 유일한 방법은 재구성을
    거치지 않고 group_membership.json을 만든 바로 그 MTM baseline npz에서 바로 계산하는
    것 — 이러면 grouping.split_low_high와 완전히 같은 코드가 완전히 같은 데이터에 도니
    diff=0이 수학적으로 보장된다(verify_missrate_band.py로 재검증할 것).

    mtm_baseline_root: DATA_ROOT/mtm/{target}/datasets/baseline (dense npz — grouping.
    split_low_high가 읽는 것과 동일 경로/포맷). target: "phq9"|"gad7" — processed_data/
    1_{target}.npz 파일명에 필요. population은 이 데이터셋 전체 window(train+val+test)로,
    특정 split으로 제한하지 않는다(grouping.split_low_high와 동일 원칙 — 부분집합에서
    다시 자르면 median 자체가 어긋난다)."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target!r}")
    from ..augment.grouping import compute_sample_missrate

    mtm_baseline_root = Path(mtm_baseline_root)
    d = np.load(mtm_baseline_root / "processed_data" / f"1_{target}.npz", allow_pickle=True)
    feats = json.loads((mtm_baseline_root / "feature_columns.json").read_text())
    subjects = [str(s) for s in json.loads((mtm_baseline_root / "subject_ids.json").read_text())]
    unit_minutes = json.loads((mtm_baseline_root / "meta.json").read_text())["unit_minutes"]

    X = np.concatenate([d["train_x"], d["val_x"], d["test_x"]], axis=0)
    M = compute_sample_missrate(np.isnan(X), feats, unit_minutes)
    win_score = M.mean(axis=1)
    assert len(subjects) == len(win_score), (
        f"subject_ids.json({len(subjects)}) vs window 수({len(win_score)}) 불일치 — "
        f"mtm_baseline_root가 grouping.split_low_high가 읽는 것과 같은 데이터셋인지 확인")

    df = pd.DataFrame({"subject_id": subjects, "score": win_score})
    subj_missrate = df.groupby("subject_id")["score"].mean().rename("missrate")

    ranked = subj_missrate.sort_values(kind="stable")
    half = len(ranked) // 2
    low_subjects = set(ranked.index[:half])
    band = pd.Series(np.where(subj_missrate.index.isin(low_subjects), "low", "high"),
                     index=subj_missrate.index, name="missrate_band")
    return pd.DataFrame({"missrate": subj_missrate, "missrate_band": band})


def build_tables(acc):
    cols, uniq = acc["cols"], acc["subjects"]
    n_win = acc["n_win"]

    miss_h = 1.0 - acc["obs_slot"] / (n_win[:, None] * acc["n_slot"])
    miss_d = 1.0 - acc["obs_day"] / (n_win[:, None] * _WINDOW_DAYS)

    rows = []
    for j, f in enumerate(cols):
        unit = "1 hour" if f in _HOURLY_FEATS else "1 day"
        mr = miss_h[:, j] if f in _HOURLY_FEATS else miss_d[:, j]
        never = acc["obs_slot"][:, j] == 0
        rows.append(pd.DataFrame({"subject_id": uniq, "feature": f, "unit": unit,
                                  "missing_rate": mr, "missing_rate_hourly": miss_h[:, j],
                                  "missing_rate_daily": miss_d[:, j],
                                  "never_measured": never}))
    subj_tab = pd.concat(rows, ignore_index=True)

    agg = []
    for f in sorted_features(cols):
        sub = subj_tab[subj_tab["feature"] == f]
        n = len(sub)
        agg.append({
            "feature": f, "group": _FEATURE_TO_GROUP[f], "unit": sub["unit"].iloc[0],
            "mean": sub["missing_rate"].mean(), "std": sub["missing_rate"].std(),
            "sem": sub["missing_rate"].std() / np.sqrt(max(n, 1)),
            "n_subject": n, "n_never": int(sub["never_measured"].sum()),
            "pct_never": sub["never_measured"].mean() * 100,
        })
    return subj_tab, pd.DataFrame(agg)


def hourly_frame(acc):
    """subject x feature x hour 결측률 (분모: 창수 x 3일)."""
    cols, uniq, n_win = acc["cols"], acc["subjects"], acc["n_win"]
    exp = (n_win[:, None, None] * _SLOTS_PER_HOUR_PER_WINDOW).astype(float)
    mr = 1.0 - acc["obs_hour"] / np.maximum(exp, 1)
    recs = []
    for j, f in enumerate(cols):
        for h in range(24):
            recs.append(pd.DataFrame({"subject_id": uniq, "feature": f, "hour": h,
                                      "missing_rate": mr[:, j, h]}))
    return pd.concat(recs, ignore_index=True)


def dow_frame(acc):
    """subject x feature x dow 결측률 (분모: 해당 요일의 창-일 수 x 24슬롯).

    fill 적용 전 원 관측 기준이다 (time.npy 가 실관측만 담고 있으므로).
    EMA 처럼 168슬롯을 채우는 feature 도 '실제로 응답한 날' 기준으로 집계된다."""
    cols, uniq = acc["cols"], acc["subjects"]
    exp = (acc["day_dow"][:, None, :] * 24).astype(float)
    mr = 1.0 - acc["obs_dow"] / np.maximum(exp, 1)
    mr[np.broadcast_to(acc["day_dow"][:, None, :] == 0, mr.shape)] = np.nan
    recs = []
    for j, f in enumerate(cols):
        for d in range(7):
            recs.append(pd.DataFrame({"subject_id": uniq, "feature": f, "dow": d,
                                      "missing_rate": mr[:, j, d]}))
    return pd.concat(recs, ignore_index=True)


def load_instance_meta(root: Path):
    """설문 인스턴스 단위 메타 (F6 용).

    sample_id 는 `{subject}_{설문일}_{before|after}_{split}_{k}` 형태다.
    앞 두 토큰이 하나의 설문 인스턴스를 가리키고, 그 안에서 PHQ-9 라벨은
    항상 일관된다. subject 단위로 집계하면 570 명은 설문 회차마다 라벨이
    달라져 하나로 정할 수 없는데, 이 단위에서는 그 모호성이 없다.

    나이/성별은 subject 속성이지만 창 단위 static 에 저장돼 있고 같은 subject
    안에서도 값이 흔들린다 (결측이 0/1 등으로 채워진 창이 섞여 있고,
    조사기간이 1년을 넘어 생일로 나이가 1 늘기도 한다). 그래서
      나이  20 이상인 창들의 중앙값, 그런 창이 없으면 제외
      성별  창별 최빈값, 동률이면 제외
    로 정리한다."""
    root = Path(root)
    st = np.load(root / "static.npy")
    sub = np.array(json.loads((root / "subject_ids.json").read_text()), dtype=object)
    sid = json.loads((root / "sample_ids.json").read_text())
    y = np.load(root / "gt.npy")[:, 0]
    inst = np.array(["_".join(s.split("_")[:2]) for s in sid], dtype=object)

    df = pd.DataFrame({"subject_id": sub, "instance_id": inst,
                       "age": st[:, 1], "sex": st[:, 0], "phq9": y})

    age = df[df["age"] >= 20].groupby("subject_id")["age"].median()
    sex = df.groupby("subject_id")["sex"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) == 1 else np.nan)
    demo = pd.DataFrame({"age": age, "sex": sex}).dropna()

    meta = (df.groupby(["instance_id", "subject_id"], as_index=False)["phq9"].first()
              .join(demo, on="subject_id", how="inner"))
    meta["age_group"] = pd.cut(meta["age"], _AGE_BINS, labels=_AGE_LABELS)
    meta["sex_label"] = meta["sex"].map({0.0: "Female", 1.0: "Male"})
    meta["phq9_label"] = meta["phq9"].map({0.0: "Control", 1.0: "Depressed"})
    return meta


# =====================================================================
# 공통 유틸
# =====================================================================

def _ampm(h):
    hh = h % 24
    return f"{hh % 12 or 12} {'AM' if hh < 12 else 'PM'}"


def _hour_ticks(ax, step=3):
    ax.set_xlim(-0.5, 24)
    ax.set_xticks(range(0, 25, step))
    ax.set_xticklabels([_ampm(h) for h in range(0, 25, step)])

def _declutter(items, gap, lo, hi):
    out = sorted(([n, float(y)] for n, y in items), key=lambda t: t[1])
    for i in range(1, len(out)):
        out[i][1] = max(out[i][1], out[i - 1][1] + gap)
    over = out[-1][1] - hi
    if over > 0:
        for it in out:
            it[1] -= over
    for i in range(len(out)):
        out[i][1] = min(max(out[i][1], lo + i * gap), hi)
    return out


def _sig_mark(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))


def _panel_curves(ax, grp, feats, table, key, n_x):
    """실선=mean, 에러바=SEM.

    subject 간 산포(SD)는 이 그림에서 보여주지 않는다. SD 가 mean 의 절반에
    달해 어떤 형태로 그려도 곡선을 덮어버리고, 사람 축 분포는 F4/A1 이
    담당하기 때문이다. 24시(=다음날 0시) 지점도 그리지 않는다."""
    ends = []
    color = _GROUP_COLORS[grp]
    xs = np.arange(n_x)
    for f in feats:
        g = (table[table["feature"] == f]
             .groupby(key)["missing_rate"].agg(["mean", "std", "count"]).reindex(range(n_x)))
        ls, mk = _FEATURE_STYLE[f]
        y = g["mean"].values
        sem = g["std"].values / np.sqrt(np.maximum(g["count"].values, 1))
        ax.errorbar(xs, y, yerr=sem, color=color, linestyle=ls, marker=mk,
                    markersize=2, linewidth=1.6, capsize=3, elinewidth=1.2, alpha=0.95)
        ends.append([f, float(y[-1])])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title(grp, weight="bold", color=color, fontsize=18)
    for f, ylab in _declutter(ends, gap=0.075, lo=0.03, hi=0.97):
        ax.text(n_x - 0.6, ylab, f, color=color, fontsize=18, weight="bold",
                va="center", ha="left", clip_on=False,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def _group_panel(ax, values, labels, colors, title):
    """violin + mean±SEM + n. 두 그룹이면 Mann-Whitney, 셋 이상이면 Kruskal-Wallis.

    앞선 분석과 같은 비모수 계열로 통일한다."""
    x = np.arange(len(values))
    parts = ax.violinplot(values, positions=x, widths=0.75,
                          showextrema=False, showmedians=False, showmeans=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.7)

    means = [float(np.mean(v)) for v in values]
    sems = [float(np.std(v, ddof=1) / np.sqrt(len(v))) for v in values]
    ax.errorbar(x, means, yerr=sems, fmt="o", color="black", markersize=5,
                capsize=4, linewidth=1.6, zorder=3)

    if len(values) == 2:
        stat, p = mannwhitneyu(values[0], values[1], alternative="two-sided")
    else:
        stat, p = kruskal(*values)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={len(v):,})" for l, v in zip(labels, values)], fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(title, fontsize=13, weight="bold")
    return {"comparison": title, "statistic": float(stat), "p_value": float(p),
            "groups": "|".join(map(str, labels)),
            "n": "|".join(str(len(v)) for v in values),
            "mean": "|".join(f"{m:.4f}" for m in means),
            "sem": "|".join(f"{s:.4f}" for s in sems)}


# =====================================================================
# F3 — 주기성 (시간축 + 요일축)
# =====================================================================

def fig3_periodicity(hour_df, dow_df, n_subjects, n_windows, out_png):
    fig, axes = plt.subplots(2, 4, figsize=(27, 12))

    for c, (grp, feats) in enumerate(_HOUR_PANELS):
        ax = axes[0, c]
        _panel_curves(ax, grp, feats, hour_df, "hour", 24)
        _hour_ticks(ax, step=6)
        ax.set_xlabel("Time of day")
        if c == 0:
            ax.set_ylabel("Missing rate")

    for c, (grp, feats) in enumerate(_DOW_PANELS):
        ax = axes[1, c]
        _panel_curves(ax, grp, feats, dow_df, "dow", 7)
        ax.set_xlim(-0.4, 6.4)
        ax.set_xticks(range(7))
        ax.set_xticklabels(_DOW_LABELS)
        ax.set_xlabel("Day of week")
        if c == 0:
            ax.set_ylabel("Missing rate")
    axes[1, 3].axis("off")

    axes[0, 0].annotate("Hourly features", xy=(-0.30, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center", weight="bold", fontsize=18)
    axes[1, 0].annotate("Daily / weekly features", xy=(-0.30, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center", weight="bold", fontsize=18)
    fig.suptitle("Missing rate across groups", weight="bold", fontsize=18)
    plt.tight_layout(rect=(0.01, 0, 0.99, 0.92), w_pad=4.5)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


# =====================================================================
# F4 — feature별 결측률 분포
# =====================================================================

def fig4_distribution(subj_tab, agg, n_subjects, n_windows, out_png):
    feats = list(agg["feature"])
    data = [subj_tab.loc[subj_tab["feature"] == f, "missing_rate"].dropna().values
            for f in feats]
    x = np.arange(len(feats))

    fig, ax = plt.subplots(figsize=(21, 8))
    parts = ax.violinplot(data, positions=x, widths=0.8, showextrema=False,
                          showmedians=False, showmeans=False)
    for body, f in zip(parts["bodies"], feats):
        body.set_facecolor(_GROUP_COLORS[_FEATURE_TO_GROUP[f]])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.75)

    ax.errorbar(x, agg["mean"], yerr=agg["sem"], fmt="o", color="black",
                markersize=5, capsize=4, linewidth=1.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(agg["feature"], rotation=45, ha="right")
    ax.set_ylabel("Missing rate")
    ax.set_xlabel("Feature (measurement unit)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Missing rate by feature", weight="bold", fontsize=18)
    legend = [plt.Rectangle((0, 0), 1, 1, fc=_GROUP_COLORS[g], edgecolor="black",
                            label=f"{g} ({_GROUP_NAME[g]})") for g in _GROUP_ORDER]
    # 범례는 축 밖으로. 안에 두면 결측률이 높은 feature 의 바이올린 상단을 가림
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              framealpha=0.95, fontsize=18)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


# =====================================================================
# F5 — 창 단위 결측률 누적분포 (low/high 컷 근거)
# =====================================================================

def fig5_cumulative(win_miss, out_png):
    v = np.sort(win_miss.astype(np.float64))
    n = len(v)
    cdf = np.arange(1, n + 1) / n
    med = float(np.median(v))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(v, cdf, color="#2c3e50", linewidth=2.2)
    ax.axhline(0.5, color="#7f8c8d", linestyle=(0, (4, 3)), linewidth=1.4)
    ax.axvline(med, color="#7f8c8d", linestyle=(0, (4, 3)), linewidth=1.4)
    ax.plot([med], [0.5], "o", color="#c0392b", markersize=7, zorder=3)
    ax.annotate(f"median = {med:.3f}", xy=(med, 0.5), xytext=(med + 0.04, 0.38),
                fontsize=18, weight="bold", color="#c0392b")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Window-level missing rate  (per 72h window, all features)")
    ax.set_ylabel("Cumulative proportion of windows")
    ax.grid(alpha=0.3)
    ax.set_title("Cumulative distribution of window-level missing rate",
             weight="bold", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    return med


# =====================================================================
# F6 — 그룹 간 결측률 비교 (PHQ-9 / 성별 / 연령)
# =====================================================================

def fig6_groups(acc, root, meta, out_png):
    """설문 인스턴스 단위 결측률을 PHQ-9 / 성별 / 연령으로 비교.

    한 인스턴스에 평균 15.7 개의 창이 딸려 있으므로 창 결측률의 평균을 낸다.
    subject 단위로 집계하면 라벨이 하나로 정해지지 않고, 창 단위로 두면
    같은 설문에서 나온 창들이 독립이 아니어서 n 이 부풀려진다."""
    sid = json.loads((Path(root) / "sample_ids.json").read_text())
    inst = np.array(["_".join(s.split("_")[:2]) for s in sid], dtype=object)
    per_inst = (pd.DataFrame({"instance_id": inst, "missing_rate": acc["win_miss"]})
                .groupby("instance_id", as_index=False)["missing_rate"].mean())
    d = meta.merge(per_inst, on="instance_id")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2),
                             gridspec_kw={"width_ratios": [1, 1, 2.1]})
    stats = []

    vals = [d.loc[d["phq9_label"] == g, "missing_rate"].values
            for g in ["Control", "Depressed"]]
    stats.append(_group_panel(axes[0], vals, ["Control", "Depressed"],
                              ["#95a5a6", "#c0392b"], "PHQ-9 group"))
    axes[0].set_ylabel("Missing rate  (survey instance, all features)")

    vals = [d.loc[d["sex_label"] == g, "missing_rate"].values for g in ["Female", "Male"]]
    stats.append(_group_panel(axes[1], vals, ["Female", "Male"],
                              ["#8e44ad", "#2980b9"], "Sex"))

    vals = [d.loc[d["age_group"] == g, "missing_rate"].values for g in _AGE_LABELS]
    cmap = plt.get_cmap("YlGnBu")
    cols = [cmap(0.30 + 0.13 * i) for i in range(len(_AGE_LABELS))]
    stats.append(_group_panel(axes[2], vals, _AGE_LABELS, cols, "Age group"))

    fig.suptitle("Missing Rate Across Groups" ,weight="bold", fontsize=18)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    return pd.DataFrame(stats)

# =====================================================================
# F7 — 결측 gap / 관측 duration 길이 분포
# =====================================================================

def collect_runs(root: Path, chunk=2000, max_windows=30000, seed=0):
    """창 안에서 연속 결측(gap)과 연속 관측(duration)의 길이를 모은다.

    time.npy 는 left-packed 라 array[f,j] 가 j 번째 슬롯이 아니라 j 번째
    관측이다. 실제 시각 time[f,j] 를 창 시작 기준 슬롯 번호로 되돌려
    72칸짜리 점유 배열을 복원한 뒤 런 길이를 센다.

    fill 이 적용된 D/E/F 는 제외한다. 하루/일주일치를 24/168 슬롯에 펼쳐
    채우므로 gap 이 원래 의미를 갖지 않는다."""
    root = Path(root)
    cols = [canonical(c) for c in json.loads((root / "feature_columns.json").read_text())]
    idx = [cols.index(f) for f in _HOURLY_FEATS]

    time = np.load(root / "time.npy", mmap_mode="r")
    n, _, n_slot = time.shape
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(n, min(max_windows, n), replace=False))

    recs = []
    for s in range(0, len(pick), chunk):
        sel = pick[s:s + chunk]
        T = np.asarray(time[sel][:, idx, :])          # (b, 8, 72)
        b = T.shape[0]
        # 관측 시각(분) -> 창 시작 기준 시간 슬롯. 창 시작은 그 창 안에서
        # 가장 이른 관측이 아니라 0분이므로 그대로 60 으로 나누면 된다.
        occ = np.zeros((b, len(idx), n_slot), dtype=bool)
        w, f, j = np.nonzero(T >= 0)
        slot = (T[w, f, j] // 60).astype(np.int64)
        ok = (slot >= 0) & (slot < n_slot)
        occ[w[ok], f[ok], slot[ok]] = True

        for fi, fname in enumerate(_HOURLY_FEATS):
            o = occ[:, fi, :]
            # 창 경계에서 잘린 런은 실제 길이를 알 수 없으므로 버린다.
            pad = np.zeros((b, 1), dtype=bool)
            for state, kind in ((True, "Observed"), (False, "Missing")):
                m = (o == state)
                mm = np.concatenate([pad, m, pad], axis=1)
                d = np.diff(mm.astype(np.int8), axis=1)
                starts = np.argwhere(d == 1)
                ends = np.argwhere(d == -1)
                if len(starts) == 0:
                    continue
                lengths = ends[:, 1] - starts[:, 1]
                interior = (starts[:, 1] > 0) & (ends[:, 1] < n_slot)
                lengths = lengths[interior]
                if len(lengths):
                    recs.append(pd.DataFrame({"feature": fname, "kind": kind,
                                              "length": lengths}))
    return pd.concat(recs, ignore_index=True)


def _clip_violin(parts, data, pad=0.5):
    """violin 폴리곤을 실제 데이터 범위로 자른다.

    matplotlib 의 violinplot 은 가우시안 KDE 를 데이터 밖까지 늘려 그린다.
    런 길이처럼 소수의 긴 값이 있는 분포에서는 그 꼬리가 창 길이 끝까지
    실선처럼 뻗어 마치 그 구간에 데이터가 있는 것처럼 보인다."""
    for body, v in zip(parts["bodies"], data):
        if not len(v):
            continue
        lo, hi = float(np.min(v)) - pad, float(np.max(v)) + pad
        xy = body.get_paths()[0].vertices
        xy[:, 1] = np.clip(xy[:, 1], lo, hi)


def fig7_gap_duration(runs, out_png):
    """feature 별로 결측 gap 과 관측 duration 의 길이 분포를 나란히 본다.

    관심사는 세그먼트 개수가 아니라 길이 분포라 histogram 대신 violin 을 쓴다.
    창이 72시간이므로 축도 72까지 열어 둔다."""
    feats = _HOURLY_FEATS
    x = np.arange(len(feats))
    off = 0.19

    fig, ax = plt.subplots(figsize=(15, 7))
    for kind, shift, alpha in [("Missing", -off, 0.85), ("Observed", off, 0.45)]:
        data = [runs.loc[(runs["feature"] == f) & (runs["kind"] == kind),
                         "length"].values for f in feats]
        parts = ax.violinplot(data, positions=x + shift, widths=0.34,
                              showextrema=False, showmedians=False)
        for body, f in zip(parts["bodies"], feats):
            body.set_facecolor(_GROUP_COLORS[_FEATURE_TO_GROUP[f]])
            body.set_edgecolor("black")
            body.set_linewidth(0.6)
            body.set_alpha(alpha)
        _clip_violin(parts, data)

        # 분포가 한쪽으로 크게 치우쳐 평균만으로는 형태가 드러나지 않으므로
        # 사분위와 p90 을 함께 얹는다.
        for xi, v in zip(x + shift, data):
            if not len(v):
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            p90 = np.percentile(v, 90)
            ax.vlines(xi, q1, q3, color="black", linewidth=4, alpha=0.8, zorder=3)
            ax.plot(xi, med, "o", color="white", markersize=3.5,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=4)
            ax.plot(xi, p90, "_", color="black", markersize=9,
                    markeredgewidth=1.4, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=30, ha="right")
    ax.set_ylim(0, 72)
    ax.set_yticks(range(0, 73, 12))
    ax.set_ylabel("Run length (hours)")
    ax.set_xlabel("Feature")
    ax.grid(alpha=0.3, axis="y")

    handles = [plt.Rectangle((0, 0), 1, 1, fc="#7f8c8d", ec="black", alpha=0.85,
                             label="Missing gap"),
               plt.Rectangle((0, 0), 1, 1, fc="#7f8c8d", ec="black", alpha=0.45,
                             label="Observed duration"),
               plt.Line2D([], [], color="black", linewidth=4, alpha=0.8, label="IQR"),
               plt.Line2D([], [], color="white", marker="o", linestyle="none",
                          markeredgecolor="black", markersize=5, label="median"),
               plt.Line2D([], [], color="black", marker="_", linestyle="none",
                          markersize=9, label="p90")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.95, ncol=2, fontsize=18)
    ax.set_title("Co-missingness across features", weight="bold", fontsize=18, pad=22)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

    out = (runs.groupby(["feature", "kind"])["length"]
           .agg(count="count", mean="mean", std="std", median="median",
                p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75),
                p90=lambda s: s.quantile(0.90), max="max")
           .reset_index())
    return out
    
# =====================================================================
# T1a — 설문 전/후 결측률
# =====================================================================

def fig_t1a_before_after(acc, root, meta, out_png):
    """같은 설문 인스턴스의 before 창과 after 창을 짝지어 비교한다.

    sample_id 의 세 번째 토큰이 before/after 다. 한 설문을 기준으로 앞쪽
    72시간과 뒤쪽 72시간이 나뉘므로, 설문에 응답한 시점 전후로 착용 행동이
    달라지는지 볼 수 있다. 같은 인스턴스 안의 쌍이라 대응표본이고,
    Wilcoxon signed-rank 로 검정한다."""
    sid = json.loads((Path(root) / "sample_ids.json").read_text())
    parts = [s.split("_") for s in sid]
    inst = np.array(["_".join(p[:2]) for p in parts], dtype=object)
    side = np.array([p[2] for p in parts], dtype=object)

    df = pd.DataFrame({"instance_id": inst, "side": side,
                       "missing_rate": acc["win_miss"]})
    df = df[df["side"].isin(["before", "after"])]
    piv = (df.groupby(["instance_id", "side"])["missing_rate"].mean()
             .unstack("side").dropna())
    piv = piv.join(meta.set_index("instance_id")[["phq9_label"]], how="inner")

    stat, p = wilcoxon(piv["before"].values, piv["after"].values)
    diff = piv["after"] - piv["before"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    data = [piv["before"].values, piv["after"].values]
    vp = ax.violinplot(data, positions=[0, 1], widths=0.7,
                       showextrema=False, showmedians=False)
    for body, c in zip(vp["bodies"], ["#16a085", "#d35400"]):
        body.set_facecolor(c)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.7)
    for xi, v in zip([0, 1], data):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(xi, q1, q3, color="black", linewidth=4, alpha=0.8, zorder=3)
        ax.plot(xi, med, "o", color="white", markersize=4,
                markeredgecolor="black", markeredgewidth=0.8, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before survey", "After survey"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Missing rate  (72h window)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Paired comparison  (n={len(piv):,} instances)\n"
                 f"Wilcoxon W = {stat:,.0f},  p = {p:.2e}  {_sig_mark(p)}",
                 fontsize=18, weight="bold")

    ax = axes[1]
    ax.hist(diff, bins=np.linspace(-1, 1, 61), color="#7f8c8d",
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.4)
    ax.axvline(float(diff.median()), color="#c0392b", linewidth=1.8,
               linestyle=(0, (4, 3)))
    ax.annotate(f"median Δ = {diff.median():+.3f}",
                xy=(diff.median(), ax.get_ylim()[1] * 0.9),
                xytext=(6, 0), textcoords="offset points",
                color="#c0392b", weight="bold", fontsize=18)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Δ missing rate  (after − before)")
    ax.set_ylabel("Survey instances")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Within-instance change", fontsize=18, weight="bold")

    fig.suptitle("Subject-level missing rate distribution", weight="bold", fontsize=18)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

    rows = [{"group": "All", "n": len(piv),
             "mean_before": float(piv["before"].mean()),
             "mean_after": float(piv["after"].mean()),
             "median_diff": float(diff.median()),
             "wilcoxon_W": float(stat), "p_value": float(p)}]
    for g in ["Control", "Depressed"]:
        sub = piv[piv["phq9_label"] == g]
        if len(sub) < 10:
            continue
        s2, p2 = wilcoxon(sub["before"].values, sub["after"].values)
        rows.append({"group": g, "n": len(sub),
                     "mean_before": float(sub["before"].mean()),
                     "mean_after": float(sub["after"].mean()),
                     "median_diff": float((sub["after"] - sub["before"]).median()),
                     "wilcoxon_W": float(s2), "p_value": float(p2)})
    return pd.DataFrame(rows)


# =====================================================================
# F8 — 동시결측 상관 (grayscale)
# =====================================================================

def spearman_matrix(M: np.ndarray) -> np.ndarray:
    """feature x feature Spearman 상관. 상수 컬럼은 0."""
    C = M.shape[1]
    if M.shape[0] < 3:
        return np.zeros((C, C))
    rho, _ = spearmanr(M, axis=0)
    rho = np.atleast_2d(rho)
    if rho.shape != (C, C):
        rho = np.corrcoef(M.T)
    R = np.nan_to_num(rho, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R


def fig8_comissing(acc, out_png, max_rows=40000, seed=0):
    cols = acc["cols"]
    order = sorted_features(cols)
    idx = [cols.index(f) for f in order]
    M = acc["feat_miss"][:, idx]
    if M.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        M = M[rng.choice(M.shape[0], max_rows, replace=False)]
    R = spearman_matrix(M.astype(np.float64))

    fig, ax = plt.subplots(figsize=(13, 11.5))
    im = ax.imshow(np.abs(R), cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=18)
    ax.set_yticklabels(order, fontsize=18)

    # 블록은 축 옆 별도 색 바로 표시한다. 히트맵 본체에 색을 쓰면 상관 강도와
    # 블록 구분이 같은 채널을 놓고 다투기 때문에 grayscale 을 유지한다.
    for i, f in enumerate(order):
        c = _GROUP_COLORS[_FEATURE_TO_GROUP[f]]
        ax.add_patch(plt.Rectangle((-1.6, i - 0.5), 0.8, 1.0, color=c,
                                   clip_on=False, transform=ax.transData))
        ax.add_patch(plt.Rectangle((i - 0.5, len(order) + 0.8), 1.0, 0.8, color=c,
                                   clip_on=False, transform=ax.transData))

    cb = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.03)
    cb.set_label("|Spearman rho|  of window-level missing rate")
    legend = [plt.Rectangle((0, 0), 1, 1, fc=_GROUP_COLORS[g], edgecolor="black",
                            label=f"{g} ({_GROUP_NAME[g]})") for g in _GROUP_ORDER]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.18, 1.0),
              framealpha=0.95, fontsize=18)
    ax.set_title("Co-missingness Structure Across Features\n"
                 f"(Spearman on window-level missing rates, n={M.shape[0]:,} windows)",
                 weight="bold", pad=22)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    return pd.DataFrame(R, index=order, columns=order)


# =====================================================================
# A1 — subject 단위 결측률 분포 (appendix, 24 feature 전체)
# =====================================================================

def figA1_subject_hist(subj_tab, agg, out_png, ncol=6):
    """y축 스케일은 패널마다 다르다. light_sensor 처럼 한 칸에 2,400 명이
    몰리는 feature 에 맞추면 나머지가 전부 납작해지기 때문."""
    feats = list(agg["feature"])
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.1 * nrow), squeeze=False)
    bins = np.linspace(0, 1, 21)

    for k, f in enumerate(feats):
        ax = axes[k // ncol, k % ncol]
        grp = _FEATURE_TO_GROUP[f]
        v = subj_tab.loc[subj_tab["feature"] == f, "missing_rate"].dropna().values
        ax.hist(v, bins=bins, color=_GROUP_COLORS[grp], alpha=0.8, edgecolor="black",
                linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"{f}  ({grp})", fontsize=18, weight="bold", color=_GROUP_COLORS[grp])
        if k % ncol == 0:
            ax.set_ylabel("Subjects")
        if k // ncol == nrow - 1:
            ax.set_xlabel("Per-subject missing rate")
    for k in range(len(feats), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")

    fig.suptitle("Subject-level Missing Rate Distribution — all 24 features "
                 "(note: y-axis scale differs across panels)",
                 weight="bold", fontsize=18)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


# =====================================================================
# 드라이버
# =====================================================================

def generate_missing_rate_figures(data_root, out_dir):
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    figs = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    print(f"accumulating from {data_root} ...")
    acc = accumulate(data_root)
    subj_tab, agg = build_tables(acc)
    hour_df = hourly_frame(acc)
    dow_df = dow_frame(acc)
    meta = load_instance_meta(data_root)
    n_subjects = len(acc["subjects"])
    n_windows = acc["n_windows"]

    subj_tab.to_csv(out_dir / "missing_rate_by_subject_feature.csv",
                    index=False, encoding="utf-8-sig")
    agg.to_csv(out_dir / "missing_rate_by_feature.csv", index=False, encoding="utf-8-sig")
    hour_df.groupby(["feature", "hour"])["missing_rate"].agg(["mean", "std"]).reset_index() \
        .to_csv(out_dir / "missing_rate_by_hour.csv", index=False, encoding="utf-8-sig")
    dow_df.groupby(["feature", "dow"])["missing_rate"].agg(["mean", "std"]).reset_index() \
        .to_csv(out_dir / "missing_rate_by_dow.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(out_dir / "instance_meta.csv", index=False, encoding="utf-8-sig")

    fig3_periodicity(hour_df, dow_df, n_subjects, n_windows, figs / "F3_periodicity.png")
    fig4_distribution(subj_tab, agg, n_subjects, n_windows, figs / "F4_distribution.png")
    med = fig5_cumulative(acc["win_miss"], figs / "F5_cumulative.png")
    f6 = fig6_groups(acc, data_root, meta, figs / "F6_groups.png")
    f6.to_csv(out_dir / "group_comparison_stats.csv", index=False, encoding="utf-8-sig")
    
    runs = collect_runs(data_root)
    f7 = fig7_gap_duration(runs, figs / "F7_gap_duration.png")
    f7.to_csv(out_dir / "run_length_stats.csv", index=False, encoding="utf-8-sig")
    
    t1a = fig_t1a_before_after(acc, data_root, meta, figs / "T1a_before_after.png")
    t1a.to_csv(out_dir / "before_after_stats.csv", index=False, encoding="utf-8-sig")
    
    R = fig8_comissing(acc, figs / "F8_comissing.png")
    R.to_csv(out_dir / "comissing_spearman.csv", encoding="utf-8-sig")
    figA1_subject_hist(subj_tab, agg, figs / "A1_subject_hist.png")

    print(f"subjects={n_subjects:,}  windows={n_windows:,}  "
          f"instances={len(meta):,}  median window missrate={med:.3f}")
    print(f"-> {out_dir}")
    return subj_tab, agg, hour_df, dow_df, meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str,
                    default="/home/hail/HDD/robot_ai/sw/data/numpy_all_chunk_72_24feat_daily")
    ap.add_argument("--out_dir", type=str,
                    default="/home/hail/Desktop/RobotAI/outputs/analysis/missing_rate")
    args = ap.parse_args()
    generate_missing_rate_figures(args.data_root, args.out_dir)