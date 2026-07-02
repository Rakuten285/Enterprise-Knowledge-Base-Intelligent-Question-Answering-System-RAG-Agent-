"""
Ragas 评测运行脚本

用法：
    # 基线评测（默认 chunk_size=500, top_k=5）
    python scripts/run_eval.py

    # 指定参数
    python scripts/run_eval.py --chunk-size 300 --top-k 3

    # 参数扫描（跑全部组合，找最优）
    python scripts/run_eval.py --sweep

    # 只跑前 N 条（快速验证流程通了）
    python scripts/run_eval.py --sample 5

结果保存到 data/eval_results/ 目录，文件名含参数，方便对比。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(result, label: str = "") -> Path:
    from app.eval.evaluator import EvaluatorConfig
    cfg = result.config
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{ts}_chunk{cfg.chunk_size}_top{cfg.top_k}{('_' + label) if label else ''}.json"
    out_path = RESULTS_DIR / filename

    data = {
        "config": {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "top_k": cfg.top_k,
        },
        "metrics": {
            "context_recall": result.context_recall,
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
        },
        "elapsed_seconds": result.elapsed_seconds,
        "per_sample": result.per_sample,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out_path}")
    return out_path


def run_baseline(chunk_size: int = 500, top_k: int = 5, sample: int | None = None):
    from app.eval.evaluator import EvaluatorConfig, evaluate, load_eval_dataset, EVAL_DATASET_PATH

    config = EvaluatorConfig(chunk_size=chunk_size, top_k=top_k)
    print(f"\n=== 基线评测：chunk_size={chunk_size}, top_k={top_k} ===")

    if sample:
        _patch_dataset(sample)

    result = evaluate(config)
    print(f"\n{'='*60}")
    print(f"评测结果：")
    print(f"  Context Recall    : {result.context_recall:.4f}")
    print(f"  Faithfulness      : {result.faithfulness:.4f}")
    print(f"  Answer Relevancy  : {result.answer_relevancy:.4f}")
    print(f"  总耗时            : {result.elapsed_seconds:.1f}s")
    print(f"{'='*60}")
    save_result(result)
    return result


def run_sweep(sample: int | None = None):
    """参数扫描：chunk_size ∈ {300, 500, 800} × top_k ∈ {3, 5, 8}"""
    from app.eval.evaluator import EvaluatorConfig, evaluate

    configs = [
        EvaluatorConfig(chunk_size=cs, top_k=tk)
        for cs in [300, 500, 800]
        for tk in [3, 5, 8]
    ]

    if sample:
        _patch_dataset(sample)

    results = []
    print(f"\n=== 参数扫描：{len(configs)} 个组合 ===\n")
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] chunk_size={cfg.chunk_size}, top_k={cfg.top_k}")
        r = evaluate(cfg)
        results.append(r)
        save_result(r, label="sweep")

    print(f"\n{'='*60}")
    print("扫描结果汇总（按 Context Recall 排序）：")
    print(f"{'chunk_size':>12} {'top_k':>6} {'CR':>8} {'FA':>8} {'AR':>8}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x.context_recall, reverse=True):
        print(
            f"{r.config.chunk_size:>12} {r.config.top_k:>6} "
            f"{r.context_recall:>8.4f} {r.faithfulness:>8.4f} {r.answer_relevancy:>8.4f}"
        )
    print(f"{'='*60}")

    best = max(results, key=lambda x: x.context_recall)
    print(f"\n最优组合（按 Context Recall）：chunk_size={best.config.chunk_size}, top_k={best.config.top_k}")
    return results


def _patch_dataset(n: int):
    """临时替换评测集为前 n 条，用于快速验证流程"""
    import app.eval.evaluator as ev_module
    original_load = ev_module.load_eval_dataset

    def _limited_load(path=None):
        data = original_load(path) if path else original_load()
        return data[:n]

    ev_module.load_eval_dataset = _limited_load
    print(f"[调试模式] 仅使用前 {n} 条评测数据")


def main():
    parser = argparse.ArgumentParser(description="Ragas RAG 评测")
    parser.add_argument("--chunk-size", type=int, default=500, help="分块大小（默认500）")
    parser.add_argument("--top-k", type=int, default=5, help="检索条数（默认5）")
    parser.add_argument("--sweep", action="store_true", help="参数扫描（9个组合）")
    parser.add_argument("--sample", type=int, default=None, help="只跑前N条（调试用）")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(sample=args.sample)
    else:
        run_baseline(chunk_size=args.chunk_size, top_k=args.top_k, sample=args.sample)


if __name__ == "__main__":
    main()
