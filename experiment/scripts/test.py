#!/usr/bin/env python3
"""在所有或指定数据集上评测所有方法（FP16基线、RQ、QINCo、FDE、PTM）。

每个数据集独占一张 NPU 卡（默认 npu:8~15）。评测顺序：
  1. FP16 exact baseline（计算上界，很快）
  2. RQ M=16（CPU k-means，约5分钟）
  3. FDE two-stage cascade（等 RQ 退出后启动，共用同一卡）
  4. PTM prune-then-merge（CPU-only，并行）

QINCo 只做推理评测（需要先用 train.py 训练好模型），若模型文件不存在则跳过。

用法:
    python experiment/scripts/test.py --emb-dir /path/to/embeddings

    python experiment/scripts/test.py \\
        --emb-dir /path/to/embeddings \\
        --out-dir experiment/results \\
        --datasets physics energy \\
        --npus 8 9 \\
        --mode all \\
        --wait
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

ALL_DATASETS = [
    "physics", "computer_science", "energy",
    "finance_en", "finance_fr", "hr", "industrial", "pharmaceuticals",
]
ALL_NPUS = [8, 9, 10, 11, 12, 13, 14, 15]

RQ_SCRIPT  = ROOT / "experiment/methods/token_quantization/rq/train_rq.py"
FDE_SCRIPT = ROOT / "experiment/methods/muvera/fde/fde_two_stage.py"
PTM_SCRIPT = ROOT / "experiment/methods/token_prune_merge/ptm/ptm_real_attention.py"
COLLECT    = ROOT / "experiment/scripts/summarize_m16_comparison.py"


def launch(cmd, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env or os.environ.copy(),
                            stdout=lf, stderr=subprocess.STDOUT, text=True)
    proc._lf = lf
    return proc


def wait_for(pid):
    """Return a bash snippet that waits for pid to exit."""
    return f"while kill -0 {pid} 2>/dev/null; do sleep 2; done; "


def chain(cmd, log_path, after_pid, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_path, "w")
    bash = wait_for(after_pid) + " ".join(str(c) for c in cmd)
    proc = subprocess.Popen(["bash", "-c", bash], env=env or os.environ.copy(),
                            stdout=lf, stderr=subprocess.STDOUT, text=True)
    proc._lf = lf
    return proc


def npu_env(phys):
    e = os.environ.copy()
    e["ASCEND_RT_VISIBLE_DEVICES"] = str(phys)
    return e


def run_fp16(cache, out_dir, npu, log_dir, ds):
    """FP16 exact baseline — inline, very fast, no subprocess."""
    import torch
    import torch_npu  # noqa: F401
    import json, time

    device = "npu:0"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)

    page  = torch.load(cache / "page_embeddings.pt",  weights_only=True).float()
    query = torch.load(cache / "query_embeddings.pt", weights_only=True).float()
    rel   = torch.load(cache / "relevance.pt",        weights_only=True)
    P, T, d = page.shape
    Q = query.shape[0]
    pmask = page.norm(dim=-1) > 1e-6
    flat  = page.reshape(P * T, d).to(device)
    mf    = pmask.reshape(P * T).to(device)
    flat[~mf] = 0.0
    avg_tok = pmask.sum(1).float().mean().item()

    neg = torch.finfo(torch.float32).min

    @torch.no_grad()
    def maxsim():
        out = torch.zeros(Q, P)
        for qi in range(0, Q, 8):
            qb = query[qi:qi+8].to(device)
            Bq = qb.shape[0]
            s = (qb.reshape(Bq * query.shape[1], d) @ flat.T).reshape(Bq, query.shape[1], P, T)
            s.masked_fill_(~mf.view(1, 1, P, T), neg)
            out[qi:qi+Bq] = s.max(3)[0].sum(1).cpu()
        return out

    def ndcg(sc, k=10):
        k2 = min(k, sc.shape[1])
        idx = sc.argsort(1, descending=True)[:, :k2]
        r   = rel.float().gather(1, idx)
        d_  = 1 / torch.log2(torch.arange(2, k2 + 2, dtype=torch.float32))
        dcg  = (r * d_).sum(1)
        idcg = (rel.float().sort(1, descending=True)[0][:, :k2] * d_).sum(1)
        m = idcg > 0
        return (dcg[m] / idcg[m]).mean().item()

    def sync(): torch.npu.synchronize()

    # warmup + timed
    for _ in range(2): maxsim(); sync()
    ts = []
    for _ in range(3):
        import time as _t; t0 = _t.perf_counter(); maxsim(); sync()
        ts.append(_t.perf_counter() - t0)
    sc  = maxsim()
    val = ndcg(sc)
    kb  = avg_tok * d * 2 / 1024

    result = dict(method="fp16_exact", dataset=ds,
                  ndcg10=round(val, 4),
                  storage_kb_per_page=round(kb, 1),
                  score_time_s=round(min(ts), 3))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fp16_{ds}.json").write_text(__import__("json").dumps(result, indent=2))
    print(f"  [fp16/{ds}]  nDCG@10={val:.4f}  {min(ts):.3f}s/query-set", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--emb-dir", type=Path,
        default=ROOT / "experiment/embeddings/colqwen2.5_v3",
        help="嵌入向量根目录")
    parser.add_argument("--out-dir", type=Path,
        default=ROOT / "experiment/results",
        help="结果输出根目录")
    parser.add_argument("--datasets", nargs="+", default=["all"], metavar="DS",
        help=f"数据集名或 'all'。可选: {', '.join(ALL_DATASETS)}")
    parser.add_argument("--npus", nargs="+", type=int, default=None, metavar="N",
        help="物理 NPU 卡号，每个数据集一个（默认 8~15）")
    parser.add_argument("--mode", choices=["all", "fp16", "rq", "fde", "ptm"],
        default="all", help="运行哪些评测方法（默认 all）")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/test_logs"),
        help="日志目录")
    parser.add_argument("--wait", action="store_true",
        help="等待所有后台任务完成后打印汇总")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets
    for ds in datasets:
        if ds not in ALL_DATASETS:
            parser.error(f"未知数据集: {ds}")

    npus = args.npus if args.npus is not None else ALL_NPUS[:len(datasets)]
    if len(npus) != len(datasets):
        parser.error(f"--npus 数量 ({len(npus)}) 与 --datasets 数量 ({len(datasets)}) 不一致")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print("=== ViDoRe V3 全方法评测 ===")
    print(f"嵌入目录 : {args.emb_dir}")
    print(f"输出目录 : {args.out_dir}")
    print(f"数据集   : {' '.join(datasets)}")
    print(f"NPU 卡号 : {' '.join(str(n) for n in npus)}")
    print(f"模式     : {args.mode}")
    print()

    procs = []

    for ds, npu in zip(datasets, npus):
        cache = args.emb_dir / ds
        if not cache.is_dir():
            print(f"  [跳过] {ds}：未找到嵌入缓存 {cache}")
            continue

        quant_out = args.out_dir / "quant"

        # FP16 baseline（在主进程里跑，极快）
        if args.mode in ("all", "fp16"):
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(npu)
            run_fp16(cache, quant_out, npu, args.log_dir, ds)

        rq_pid = None

        # RQ M=16
        if args.mode in ("all", "rq"):
            log = args.log_dir / f"{ds}_rq.log"
            rq = launch(
                [sys.executable, str(RQ_SCRIPT),
                 "--cache-dir", str(cache),
                 "--out-dir",   str(quant_out)],
                log, npu_env(npu))
            rq_pid = rq.pid
            procs.append((f"{ds}/rq", rq, log))
            print(f"  [{ds}/rq]  NPU={npu}  pid={rq_pid}  log={log}")

        # FDE cascade（等 RQ 退出）
        if args.mode in ("all", "fde"):
            fde_out = args.out_dir / "fde"
            log = args.log_dir / f"{ds}_fde.log"
            cmd = [sys.executable, str(FDE_SCRIPT),
                   "--cache-dir", str(cache),
                   "--out-dir",   str(fde_out)]
            fde = chain(cmd, log, rq_pid, npu_env(npu)) if rq_pid else launch(cmd, log, npu_env(npu))
            procs.append((f"{ds}/fde", fde, log))
            print(f"  [{ds}/fde]  NPU={npu}  pid={fde.pid}  log={log}")

    # PTM（CPU-only，按数据集串行）
    if args.mode in ("all", "ptm"):
        for ds in datasets:
            cache = args.emb_dir / ds
            if not cache.is_dir():
                continue
            ptm_out = args.out_dir / "ptm"
            log = args.log_dir / f"{ds}_ptm.log"
            ptm = launch(
                [sys.executable, str(PTM_SCRIPT),
                 "--cache-dir", str(cache),
                 "--out",       str(ptm_out / f"ptm_{ds}.json")],
                log)
            procs.append((f"{ds}/ptm", ptm, log))
            print(f"  [{ds}/ptm]  cpu  pid={ptm.pid}  log={log}")

    if not procs:
        print("无后台进程，评测完成。")
        return

    print()
    print(f"已启动 {len(procs)} 个评测进程。")
    print(f"监控日志: tail -f {args.log_dir}/*.log")
    print(f"汇总结果: python {COLLECT} --results-dir {args.out_dir}/quant")

    if args.wait:
        print("\n等待所有评测完成...")
        failed = []
        for label, proc, log in procs:
            proc.wait()
            proc._lf.close()
            status = "完成" if proc.returncode == 0 else f"失败(exit={proc.returncode})"
            print(f"  [{label}] {status}")
            if proc.returncode != 0:
                failed.append(label)
        if failed:
            print(f"\n警告: {len(failed)} 个任务失败: {failed}")
        else:
            print(f"\n全部评测完成。结果在: {args.out_dir}")
            subprocess.run([sys.executable, str(COLLECT),
                            "--results-dir", str(args.out_dir / "quant")])


if __name__ == "__main__":
    main()
