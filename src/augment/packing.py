"""
packing.py
==========
CoFormer packed format(array/time/mask) <-> dense (N,T,C) NaN-grid 변환 유틸.
원본: MTM/analysis/augment_coformer_group_aware.py의 load_packed/parse_wstart_min/
unpack_dense/repack_from_dense (verbatim numpy, 로직 변경 없음).

CoFormer sample_id의 "_s{N}" 접미사 = 그 샘플의 time 배열과 동일 좌표계의 절대
window 시작(분). 이 값으로 packed<->dense를 왕복 변환할 수 있다
(step2_coformer.py make_sample(): sample_id에 w['w_start_abs_min']을 그대로 기록).
"""
import json

import numpy as np


def load_packed(d):
    """d: packed 데이터셋 디렉터리(Path). array.npy/time.npy/mask.npy 등을 읽어 dict로 반환."""
    array = np.load(d / "array.npy")   # (N,C,T)
    time = np.load(d / "time.npy")     # (N,C,T) absolute minutes, -1=pad
    mask = np.load(d / "mask.npy")     # (N,C) valid-count
    static = np.load(d / "static.npy")
    gt = np.load(d / "gt.npy")
    split = np.load(d / "split.npy", allow_pickle=True)
    feats = json.load(open(d / "feature_columns.json"))
    sids = np.array(json.load(open(d / "subject_ids.json")))
    smids = json.load(open(d / "sample_ids.json"))
    packed = dict(array=array, time=time, mask=mask, static=static, gt=gt,
                  split=split, feats=feats, sids=sids, smids=smids)
    for name in ("phq9_gt", "gad7_gt"):
        path = d / f"{name}.npy"
        if path.exists():
            packed[name] = np.load(path)
    return packed


def parse_wstart_min(sample_ids):
    return np.array([int(s.rsplit("_s", 1)[1]) for s in sample_ids], dtype=np.int64)


def unpack_dense(array, time, mask, wstart_min, T, unit_minutes):
    """packed(N,C,T_cap) -> dense(N,T,C) NaN-grid."""
    N, C, _ = array.shape
    dense = np.full((N, T, C), np.nan, dtype=np.float32)
    for i in range(N):
        w0 = wstart_min[i]
        for c in range(C):
            k = mask[i, c]
            if k == 0:
                continue
            t_vals = time[i, c, :k]
            v_vals = array[i, c, :k]
            slots = np.round((t_vals - w0) / unit_minutes).astype(int)
            valid = (slots >= 0) & (slots < T)
            if valid.any():
                dense[i, slots[valid], c] = v_vals[valid]
    return dense


def repack_from_dense(dense, wstart_min, unit_minutes, T_pack):
    """dense(N,T,C) -> packed(array(N,C,T_pack), time(N,C,T_pack), mask(N,C))."""
    N, T, C = dense.shape
    array = np.zeros((N, C, T_pack), dtype=np.float32)
    time_arr = np.full((N, C, T_pack), -1, dtype=np.float32)
    mask_arr = np.zeros((N, C), dtype=np.int32)
    for i in range(N):
        w0 = wstart_min[i]
        for c in range(C):
            valid_t = np.where(~np.isnan(dense[i, :, c]))[0]
            k = min(len(valid_t), T_pack)
            if k == 0:
                continue
            array[i, c, :k] = dense[i, valid_t[:k], c]
            time_arr[i, c, :k] = w0 + valid_t[:k] * unit_minutes
            mask_arr[i, c] = k
    return array, time_arr, mask_arr
