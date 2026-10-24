"""
Generates data/sample_data.csv — a synthetic CICIDS2017-format dataset.
300 BENIGN rows + 100 DoS Hulk rows + 100 Web Attack rows = 500 rows total.

Design goal: distributions overlap enough that:
  - Baseline RF accuracy ~90-96%  (realistic, not trivially perfect)
  - FGSM at eps=0.1-0.2 causes significant detection rate drop (to ~40-60%)
  - Defense partially recovers (to ~70-85%)

Run once: python _gen_sample_data.py
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_data.csv")


def clip_pos(arr):
    return np.maximum(arr, 0)


def gen_benign(n=300):
    """Typical HTTPS/HTTP browsing — moderate flows, balanced directions."""
    # Deliberately overlapping ranges with attack so the model is challenged
    dur = RNG.gamma(2.0, 300_000, n)               # ~600K us, right-skewed
    fwd_pkts = RNG.poisson(15, n) + 1
    bwd_pkts = RNG.poisson(12, n) + 1
    fwd_len = fwd_pkts * RNG.normal(400, 80, n)
    bwd_len = bwd_pkts * RNG.normal(380, 75, n)
    fpkt_max = RNG.normal(1200, 250, n)
    fpkt_min = RNG.normal(50, 20, n)
    fpkt_mean = RNG.normal(420, 100, n)
    fpkt_std = RNG.normal(250, 60, n)
    bpkt_max = RNG.normal(1150, 240, n)
    bpkt_min = RNG.normal(45, 18, n)
    bpkt_mean = RNG.normal(400, 90, n)
    bpkt_std = RNG.normal(230, 55, n)
    bytes_s = RNG.lognormal(10.5, 1.0, n)          # log-normal, wide range
    pkts_s = RNG.lognormal(3.2, 0.8, n)
    iat_mean = RNG.lognormal(11.0, 1.2, n)
    iat_std = RNG.lognormal(10.8, 1.3, n)
    iat_max = iat_mean * RNG.uniform(2, 8, n)
    iat_min = RNG.uniform(50, 500, n)
    fiat_total = dur * RNG.uniform(0.7, 0.95, n)
    fiat_mean = fiat_total / np.maximum(fwd_pkts - 1, 1)
    fiat_std = fiat_mean * RNG.uniform(0.3, 1.2, n)
    fiat_max = fiat_mean * RNG.uniform(2, 6, n)
    fiat_min = RNG.uniform(40, 400, n)
    biat_total = dur * RNG.uniform(0.6, 0.9, n)
    biat_mean = biat_total / np.maximum(bwd_pkts - 1, 1)
    biat_std = biat_mean * RNG.uniform(0.3, 1.2, n)
    biat_max = biat_mean * RNG.uniform(2, 6, n)
    biat_min = RNG.uniform(40, 400, n)
    fin_flag = RNG.integers(0, 2, n)
    syn_flag = RNG.integers(0, 2, n)
    rst_flag = RNG.integers(0, 2, n)
    psh_flag = RNG.integers(0, 2, n)
    ack_flag = RNG.integers(1, 3, n)
    urg_flag = np.zeros(n, dtype=int)
    down_up = RNG.normal(0.9, 0.35, n)
    avg_pkt = (fwd_len + bwd_len) / (fwd_pkts + bwd_pkts + 1e-6)
    avg_fwd_seg = fpkt_mean
    avg_bwd_seg = bpkt_mean
    init_win_fwd = RNG.integers(4096, 65536, n)
    init_win_bwd = RNG.integers(4096, 65536, n)
    act_data_fwd = fwd_pkts
    min_seg_fwd = np.full(n, 20)
    active_mean = dur * RNG.uniform(0.3, 0.7, n)
    active_std = active_mean * RNG.uniform(0.1, 0.4, n)
    active_max = active_mean + active_std * RNG.uniform(1, 3, n)
    active_min = np.maximum(active_mean - active_std * RNG.uniform(1, 3, n), 0)
    idle_mean = dur * RNG.uniform(0.1, 0.4, n)
    idle_std = idle_mean * RNG.uniform(0.1, 0.4, n)
    idle_max = idle_mean + idle_std * RNG.uniform(1, 3, n)
    idle_min = np.maximum(idle_mean - idle_std * RNG.uniform(1, 3, n), 0)

    return dict(
        FlowDuration=dur, TotalFwdPackets=fwd_pkts, TotalBackwardPackets=bwd_pkts,
        TotalLengthFwdPackets=fwd_len, TotalLengthBwdPackets=bwd_len,
        FwdPacketLengthMax=fpkt_max, FwdPacketLengthMin=fpkt_min,
        FwdPacketLengthMean=fpkt_mean, FwdPacketLengthStd=fpkt_std,
        BwdPacketLengthMax=bpkt_max, BwdPacketLengthMin=bpkt_min,
        BwdPacketLengthMean=bpkt_mean, BwdPacketLengthStd=bpkt_std,
        FlowBytess=bytes_s, FlowPacketss=pkts_s,
        FlowIATMean=iat_mean, FlowIATStd=iat_std,
        FlowIATMax=iat_max, FlowIATMin=iat_min,
        FwdIATTotal=fiat_total, FwdIATMean=fiat_mean, FwdIATStd=fiat_std,
        FwdIATMax=fiat_max, FwdIATMin=fiat_min,
        BwdIATTotal=biat_total, BwdIATMean=biat_mean, BwdIATStd=biat_std,
        BwdIATMax=biat_max, BwdIATMin=biat_min,
        FINFlagCount=fin_flag, SYNFlagCount=syn_flag, RSTFlagCount=rst_flag,
        PSHFlagCount=psh_flag, ACKFlagCount=ack_flag, URGFlagCount=urg_flag,
        DownUpRatio=down_up, AveragePacketSize=avg_pkt,
        AvgFwdSegmentSize=avg_fwd_seg, AvgBwdSegmentSize=avg_bwd_seg,
        InitWinbytesFwd=init_win_fwd, InitWinbytesBwd=init_win_bwd,
        ActDataPktFwd=act_data_fwd, MinSegSizeFwd=min_seg_fwd,
        ActiveMean=active_mean, ActiveStd=active_std,
        ActiveMax=active_max, ActiveMin=active_min,
        IdleMean=idle_mean, IdleStd=idle_std,
        IdleMax=idle_max, IdleMin=idle_min,
        Label=["BENIGN"] * n,
    )


def gen_dos_hulk(n=100):
    """
    DoS Hulk: high rate, short flows — but distributions partially overlap
    with benign to make adversarial attack more effective.
    """
    dur = RNG.gamma(1.5, 50_000, n)                # ~75K us — shorter but overlapping
    fwd_pkts = RNG.poisson(120, n) + 5             # many packets
    bwd_pkts = RNG.poisson(8, n) + 1               # few replies
    fwd_len = fwd_pkts * RNG.normal(180, 40, n)
    bwd_len = bwd_pkts * RNG.normal(90, 30, n)
    fpkt_max = RNG.normal(1460, 50, n)
    fpkt_min = RNG.normal(40, 12, n)
    fpkt_mean = RNG.normal(180, 50, n)
    fpkt_std = RNG.normal(350, 80, n)
    bpkt_max = RNG.normal(180, 60, n)
    bpkt_min = RNG.normal(40, 12, n)
    bpkt_mean = RNG.normal(90, 30, n)
    bpkt_std = RNG.normal(55, 18, n)
    bytes_s = RNG.lognormal(13.0, 0.8, n)          # high bytes/s, overlapping with benign tail
    pkts_s = RNG.lognormal(7.5, 0.9, n)            # high pkt/s
    iat_mean = RNG.lognormal(3.5, 0.9, n)          # tiny IAT (exp ~33us)
    iat_std = RNG.lognormal(3.2, 0.9, n)
    iat_max = iat_mean * RNG.uniform(2, 10, n)
    iat_min = RNG.uniform(1, 20, n)
    fiat_total = dur * RNG.uniform(0.6, 0.9, n)
    fiat_mean = fiat_total / np.maximum(fwd_pkts - 1, 1)
    fiat_std = fiat_mean * RNG.uniform(0.2, 0.8, n)
    fiat_max = fiat_mean * RNG.uniform(3, 8, n)
    fiat_min = RNG.uniform(1, 15, n)
    biat_total = dur * RNG.uniform(0.3, 0.7, n)
    biat_mean = biat_total / np.maximum(bwd_pkts - 1, 1)
    biat_std = biat_mean * RNG.uniform(0.2, 0.8, n)
    biat_max = biat_mean * RNG.uniform(3, 8, n)
    biat_min = RNG.uniform(5, 30, n)
    fin_flag = np.zeros(n, dtype=int)
    syn_flag = np.ones(n, dtype=int)
    rst_flag = RNG.integers(0, 3, n)
    psh_flag = np.ones(n, dtype=int)
    ack_flag = np.ones(n, dtype=int)
    urg_flag = np.zeros(n, dtype=int)
    down_up = RNG.normal(0.07, 0.03, n)
    avg_pkt = (fwd_len + bwd_len) / (fwd_pkts + bwd_pkts + 1e-6)
    avg_fwd_seg = fpkt_mean
    avg_bwd_seg = bpkt_mean
    init_win_fwd = RNG.integers(8192, 65536, n)
    init_win_bwd = RNG.integers(8192, 65536, n)
    act_data_fwd = fwd_pkts
    min_seg_fwd = np.full(n, 20)
    active_mean = dur * RNG.uniform(0.4, 0.8, n)
    active_std = active_mean * RNG.uniform(0.1, 0.3, n)
    active_max = active_mean + active_std * RNG.uniform(1, 3, n)
    active_min = np.maximum(active_mean - active_std * RNG.uniform(1, 3, n), 0)
    idle_mean = dur * RNG.uniform(0.05, 0.2, n)
    idle_std = idle_mean * RNG.uniform(0.1, 0.4, n)
    idle_max = idle_mean + idle_std * RNG.uniform(1, 3, n)
    idle_min = np.maximum(idle_mean - idle_std * RNG.uniform(1, 3, n), 0)

    return dict(
        FlowDuration=dur, TotalFwdPackets=fwd_pkts, TotalBackwardPackets=bwd_pkts,
        TotalLengthFwdPackets=fwd_len, TotalLengthBwdPackets=bwd_len,
        FwdPacketLengthMax=fpkt_max, FwdPacketLengthMin=fpkt_min,
        FwdPacketLengthMean=fpkt_mean, FwdPacketLengthStd=fpkt_std,
        BwdPacketLengthMax=bpkt_max, BwdPacketLengthMin=bpkt_min,
        BwdPacketLengthMean=bpkt_mean, BwdPacketLengthStd=bpkt_std,
        FlowBytess=bytes_s, FlowPacketss=pkts_s,
        FlowIATMean=iat_mean, FlowIATStd=iat_std,
        FlowIATMax=iat_max, FlowIATMin=iat_min,
        FwdIATTotal=fiat_total, FwdIATMean=fiat_mean, FwdIATStd=fiat_std,
        FwdIATMax=fiat_max, FwdIATMin=fiat_min,
        BwdIATTotal=biat_total, BwdIATMean=biat_mean, BwdIATStd=biat_std,
        BwdIATMax=biat_max, BwdIATMin=biat_min,
        FINFlagCount=fin_flag, SYNFlagCount=syn_flag, RSTFlagCount=rst_flag,
        PSHFlagCount=psh_flag, ACKFlagCount=ack_flag, URGFlagCount=urg_flag,
        DownUpRatio=down_up, AveragePacketSize=avg_pkt,
        AvgFwdSegmentSize=avg_fwd_seg, AvgBwdSegmentSize=avg_bwd_seg,
        InitWinbytesFwd=init_win_fwd, InitWinbytesBwd=init_win_bwd,
        ActDataPktFwd=act_data_fwd, MinSegSizeFwd=min_seg_fwd,
        ActiveMean=active_mean, ActiveStd=active_std,
        ActiveMax=active_max, ActiveMin=active_min,
        IdleMean=idle_mean, IdleStd=idle_std,
        IdleMax=idle_max, IdleMin=idle_min,
        Label=["DoS Hulk"] * n,
    )


def gen_web_attack(n=100):
    """
    Web Attack: moderate flow, unusual flags — partially overlaps benign.
    Includes some 'borderline' samples close to benign decision boundary.
    """
    dur = RNG.gamma(2.0, 80_000, n)               # ~160K us
    fwd_pkts = RNG.poisson(10, n) + 2
    bwd_pkts = RNG.poisson(8, n) + 1
    fwd_len = fwd_pkts * RNG.normal(270, 70, n)
    bwd_len = bwd_pkts * RNG.normal(900, 200, n)  # large server response (SQL output)
    fpkt_max = RNG.normal(750, 180, n)
    fpkt_min = RNG.normal(52, 16, n)
    fpkt_mean = RNG.normal(270, 65, n)
    fpkt_std = RNG.normal(190, 50, n)
    bpkt_max = RNG.normal(1100, 200, n)
    bpkt_min = RNG.normal(48, 15, n)
    bpkt_mean = RNG.normal(860, 170, n)
    bpkt_std = RNG.normal(360, 75, n)
    bytes_s = RNG.lognormal(11.3, 0.9, n)         # medium-high, overlaps benign
    pkts_s = RNG.lognormal(5.0, 0.8, n)
    iat_mean = RNG.lognormal(9.5, 1.1, n)
    iat_std = RNG.lognormal(9.8, 1.3, n)
    iat_max = iat_mean * RNG.uniform(3, 10, n)
    iat_min = RNG.uniform(120, 800, n)
    fiat_total = dur * RNG.uniform(0.5, 0.85, n)
    fiat_mean = fiat_total / np.maximum(fwd_pkts - 1, 1)
    fiat_std = fiat_mean * RNG.uniform(0.4, 1.4, n)
    fiat_max = fiat_mean * RNG.uniform(3, 8, n)
    fiat_min = RNG.uniform(100, 600, n)
    biat_total = dur * RNG.uniform(0.4, 0.8, n)
    biat_mean = biat_total / np.maximum(bwd_pkts - 1, 1)
    biat_std = biat_mean * RNG.uniform(0.4, 1.4, n)
    biat_max = biat_mean * RNG.uniform(3, 8, n)
    biat_min = RNG.uniform(100, 600, n)
    fin_flag = RNG.integers(0, 2, n)
    syn_flag = np.ones(n, dtype=int)
    rst_flag = RNG.integers(0, 2, n)
    psh_flag = np.ones(n, dtype=int)
    ack_flag = np.ones(n, dtype=int)
    urg_flag = RNG.integers(0, 2, n)
    down_up = RNG.normal(3.1, 0.8, n)
    avg_pkt = (fwd_len + bwd_len) / (fwd_pkts + bwd_pkts + 1e-6)
    avg_fwd_seg = fpkt_mean
    avg_bwd_seg = bpkt_mean
    init_win_fwd = RNG.integers(8192, 65536, n)
    init_win_bwd = RNG.integers(512, 8192, n)
    act_data_fwd = fwd_pkts
    min_seg_fwd = np.full(n, 20)
    active_mean = dur * RNG.uniform(0.4, 0.75, n)
    active_std = active_mean * RNG.uniform(0.1, 0.35, n)
    active_max = active_mean + active_std * RNG.uniform(1, 3, n)
    active_min = np.maximum(active_mean - active_std * RNG.uniform(1, 3, n), 0)
    idle_mean = dur * RNG.uniform(0.1, 0.35, n)
    idle_std = idle_mean * RNG.uniform(0.1, 0.4, n)
    idle_max = idle_mean + idle_std * RNG.uniform(1, 3, n)
    idle_min = np.maximum(idle_mean - idle_std * RNG.uniform(1, 3, n), 0)

    return dict(
        FlowDuration=dur, TotalFwdPackets=fwd_pkts, TotalBackwardPackets=bwd_pkts,
        TotalLengthFwdPackets=fwd_len, TotalLengthBwdPackets=bwd_len,
        FwdPacketLengthMax=fpkt_max, FwdPacketLengthMin=fpkt_min,
        FwdPacketLengthMean=fpkt_mean, FwdPacketLengthStd=fpkt_std,
        BwdPacketLengthMax=bpkt_max, BwdPacketLengthMin=bpkt_min,
        BwdPacketLengthMean=bpkt_mean, BwdPacketLengthStd=bpkt_std,
        FlowBytess=bytes_s, FlowPacketss=pkts_s,
        FlowIATMean=iat_mean, FlowIATStd=iat_std,
        FlowIATMax=iat_max, FlowIATMin=iat_min,
        FwdIATTotal=fiat_total, FwdIATMean=fiat_mean, FwdIATStd=fiat_std,
        FwdIATMax=fiat_max, FwdIATMin=fiat_min,
        BwdIATTotal=biat_total, BwdIATMean=biat_mean, BwdIATStd=biat_std,
        BwdIATMax=biat_max, BwdIATMin=biat_min,
        FINFlagCount=fin_flag, SYNFlagCount=syn_flag, RSTFlagCount=rst_flag,
        PSHFlagCount=psh_flag, ACKFlagCount=ack_flag, URGFlagCount=urg_flag,
        DownUpRatio=down_up, AveragePacketSize=avg_pkt,
        AvgFwdSegmentSize=avg_fwd_seg, AvgBwdSegmentSize=avg_bwd_seg,
        InitWinbytesFwd=init_win_fwd, InitWinbytesBwd=init_win_bwd,
        ActDataPktFwd=act_data_fwd, MinSegSizeFwd=min_seg_fwd,
        ActiveMean=active_mean, ActiveStd=active_std,
        ActiveMax=active_max, ActiveMin=active_min,
        IdleMean=idle_mean, IdleStd=idle_std,
        IdleMax=idle_max, IdleMin=idle_min,
        Label=["Web Attack"] * n,
    )


def main():
    benign = gen_benign(300)
    dos = gen_dos_hulk(100)
    web = gen_web_attack(100)

    dfs = []
    for data in [benign, dos, web]:
        df = pd.DataFrame(data)
        num_cols = [c for c in df.columns if c != "Label"]
        df[num_cols] = df[num_cols].clip(lower=0)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    combined.to_csv(OUT, index=False)
    print(f"Generated: {OUT}  ({len(combined)} rows)")
    print(combined["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
