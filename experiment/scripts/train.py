#!/usr/bin/env python3
"""训练量化模型（RQ M=16 或 QINCo MSE M=16）在所有或指定数据集上并行运行。

每个数据集独占一张 NPU 卡（默认 npu:8~15）。

用法:
    # 训练所有方法（RQ + QINCo），全数据集
    python experiment/scripts/train.py --emb-dir experiment/embeddings/colqwen2.5_v3

    # 只训练 QINCo
    python experiment/scripts/train.py --emb-dir experiment/embeddings/colqwen2.5_v3 --method qinco

    # 指定数据集和卡号
    python experiment/scripts/train.py \\
        --emb-dir experiment/embeddings/colqwen2.5_v3 \\
        --out-dir  experiment/results/quant \\
        --datasets physics energy \\
        --npus 8 9 \\
        --method all \\
        --epochs 10 \\
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

RQ_SCRIPT    = ROOT / "experiment/methods/token_quantization/rq/train_rq.py"
QINCO_SCRIPT = ROOT / "experiment/methods/token_quantization/qinco2/train_qinco.py"


def launch(cmd, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env or os.environ.copy(),
                            stdout=lf, stderr=subprocess.STDOUT, text=True)
    proc._lf = lf
    return proc


def npu_env(phys):
    e = os.environ.copy()
    e["ASCEND_RT_VISIBLE_DEVICES"] = str(phys)
    return e


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--emb-dir", type=Path,
        default=ROOT / "experiment/embeddings/colqwen2.5_v3",
        help="嵌入向量根目录，每个数据集一个子目录")
    parser.add_argument("--out-dir", type=Path,
        default=ROOT / "experiment/results/quant",
        help="训练结果输出目录")
    parser.add_argument("--datasets", nargs="+", default=["all"], metavar="DS",
        help=f"数据集名或 'all'。可选: {', '.join(ALL_DATASETS)}")
    parser.add_argument("--npus", nargs="+", type=int, default=None, metavar="N",
        help="物理 NPU 卡号，每个数据集一个（默认 8~15）")
    parser.add_argument("--method", choices=["all", "rq", "qinco"], default="all",
        help="训练哪种量化模型（默认 all = RQ + QINCo）")
    parser.add_argument("--epochs", type=int, default=10,
        help="QINCo 训练 epoch 数（默认 10，RQ 无 epoch 概念）")
    parser.add_argument("--m", type=int, default=16, help="码本数量 M（默认 16）")
    parser.add_argument("--k", type=int, default=256, help="码本大小 K（默认 256）")
    parser.add_argument("--qinco-dir", type=Path,
        default=ROOT / "experiment/methods/token_quantization/qinco2",
        help="QINCo 包目录（包含 qinco/ 子包）")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/train_logs"),
        help="日志目录（默认 /tmp/train_logs）")
    parser.add_argument("--wait", action="store_true", help="等待所有任务完成")
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

    print("=== 量化模型训练 ===")
    print(f"方法     : {args.method}  (M={args.m} K={args.k}"
          + (f" epochs={args.epochs}" if args.method in ("all", "qinco") else "") + ")")
    print(f"嵌入目录 : {args.emb_dir}")
    print(f"输出目录 : {args.out_dir}")
    print(f"数据集   : {' '.join(datasets)}")
    print(f"NPU 卡号 : {' '.join(str(n) for n in npus)}")
    print()

    procs = []
    for ds, npu in zip(datasets, npus):
        cache = args.emb_dir / ds
        if not cache.is_dir():
            print(f"  [跳过] {ds}：未找到嵌入缓存 {cache}")
            continue

        # RQ M=16（CPU k-means，同一张卡上先跑）
        if args.method in ("all", "rq"):
            log = args.log_dir / f"{ds}_rq.log"
            proc = launch([
                sys.executable, str(RQ_SCRIPT),
                "--cache-dir", str(cache),
                "--out-dir",   str(args.out_dir),
                "--m",         str(args.m),
                "--k",         str(args.k),
            ], log, npu_env(npu))
            procs.append((f"{ds}/rq", proc, log))
            print(f"  [{ds}/rq]   NPU={npu}  pid={proc.pid}  log={log}")

        # QINCo M=16 MSE
        if args.method in ("all", "qinco"):
            log = args.log_dir / f"{ds}_qinco.log"
            proc = launch([
                sys.executable, str(QINCO_SCRIPT),
                "--cache-dir", str(cache),
                "--out-dir",   str(args.out_dir),
                "--epochs",    str(args.epochs),
                "--m",         str(args.m),
                "--k",         str(args.k),
                "--qinco-dir", str(args.qinco_dir),
            ], log, npu_env(npu))
            procs.append((f"{ds}/qinco", proc, log))
            print(f"  [{ds}/qinco] NPU={npu}  pid={proc.pid}  log={log}")

    if not procs:
        print("没有有效的数据集，退出。")
        return

    print()
    print(f"已启动 {len(procs)} 个训练进程。")
    print(f"监控日志: tail -f {args.log_dir}/*.log")

    if args.wait:
        print("\n等待所有训练完成...")
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
            print(f"\n全部训练完成。结果在: {args.out_dir}")


if __name__ == "__main__":
    main()
