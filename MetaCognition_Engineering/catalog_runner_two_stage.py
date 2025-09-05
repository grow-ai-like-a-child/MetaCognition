#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从"现有 JSON 题目表（catalog_*.json/.jsonl）"直接跑完整工程：
支持两阶段提问：
  gen     — 逐题调用你的三套生成器生成图片（缓存命中则跳过）
  pack    — 生成 questions_two_stage.jsonl（两阶段问题）
  feed    — 喂给模型（默认 mock，可切 openai）
  collect — 解析模型输出并与题目表/GT 合并，导出 eval.csv

要求：
  generators/
    ├─ grid_levels.py         # 你提供的 Grid（XO 网格）新难度梯度版
    ├─ gabor.py               # 你提供的 Gabor 新难度梯度版
    └─ color_shading.py       # 你提供的 Color（明暗）模块

用法（PowerShell 示例）：
  python .\catalog_runner_two_stage.py gen     --catalog .\catalog_2688_newspec.json --imgroot .\cache\images
  python .\catalog_runner_two_stage.py pack    --catalog .\catalog_2688_newspec.json --imgroot .\cache\images --out .\questions_two_stage.jsonl
  python .\catalog_runner_two_stage.py feed    --pack .\questions_two_stage.jsonl --out .\responses_two_stage.jsonl --engine mock --workers 8
  python .\catalog_runner_two_stage.py collect --resp .\responses_two_stage.jsonl --catalog .\catalog_2688_newspec.json --out .\eval_two_stage.csv

可选：--engine openai 需要 openai>=1.0.0 且设置环境变量 OPENAI_API_KEY。
"""
from __future__ import annotations
import argparse, json, csv, os, re, time, base64, hashlib, random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# -------------------------
# 安全导入你的三套生成器
# -------------------------
try:
    from generators import grid as grid
except Exception:
    import grid as grid  # 若在同级目录
try:
    from generators import gabor as gabor
except Exception:
    import gabor as gabor
try:
    from generators import color_shading as color_mod
except Exception:
    import color_shading as color_mod

GEN_MAP = {
    "Grid": grid,
    "Gabor": gabor,
    "Color": color_mod,
}

# -------------------------
# 工具
# -------------------------

def read_json_any(path: Path) -> List[Dict[str, Any]]:
    """支持 .json（数组）或 .jsonl（每行一条）"""
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    else:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------
# gen — 根据 catalog 生成图片
# -------------------------

def cmd_gen(catalog: Path, imgroot: Path, only_task: str | None = None) -> None:
    rows = read_json_any(catalog)
    n_generated = 0
    n_cached = 0
    n_total = 0
    
    for item in rows:
        task = item["task"]
        if only_task and task != only_task:
            continue
        n_total += 1
        
        params = item.get("params", {})
        mod = GEN_MAP.get(task)
        if mod is None or not hasattr(mod, "generate"):
            raise RuntimeError(f"No generator for task {task}")
        out_dir = imgroot / task
        ensure_dir(out_dir)
        
        qid = item["qid"]
        out_path = out_dir / f"{qid}.png"
        
        if out_path.exists():
            n_cached += 1
            continue
        
        try:
            # 生成图片，获取生成的文件路径
            generated_path = mod.generate(params, str(out_dir))
            # 重命名为qid格式
            if generated_path and Path(generated_path).exists():
                Path(generated_path).rename(out_path)
            n_generated += 1
        except Exception as e:
            print(f"[gen] failed to generate {qid}: {e}")
    
    print(f"[gen] {n_generated} generated, {n_cached} cached, {n_total} total")

# -------------------------
# 两阶段提示生成
# -------------------------

def prompt_for_stage1(item: Dict[str,Any]) -> str:
    """第一阶段：实验任务（不包含置信度要求）"""
    task = item["task"]
    if task == "Grid":
        # 目录里 derived 已给 symA/symB
        d = item.get("derived", {})
        symA = d.get("gt", {}).get("symA") or d.get("symA") or "A"
        symB = d.get("gt", {}).get("symB") or d.get("symB") or "B"
        return (
            f"Which one has more: {symA} or {symB}?\n"
            f"Choose one:\n"
            f"A. {symA}\n"
            f"B. {symB}\n"
            f"Answer with A or B."
        )
    elif task == "Gabor":
        return (
            "Which direction are the stripes closer to: vertical or horizontal?\n"
            "Choose one:\n"
            "A. Vertical\n"
            "B. Horizontal\n"
            "Answer with A or B."
        )
    else:  # Color
        lay = item.get("params", {}).get("layout", "LR")
        hint = {
            "LR": "left/right",
            "UD": "top/bottom",
            "DRUL": "bottom-left/top-right",
            "DLUR": "top-left/bottom-right",
        }.get(lay, "left/right")
        
        # 根据布局生成选项
        if lay == "LR":
            options = "A. Left\nB. Right"
        elif lay == "UD":
            options = "A. Top\nB. Bottom"
        elif lay == "DRUL":
            options = "A. Bottom-left\nB. Top-right"
        elif lay == "DLUR":
            options = "A. Top-left\nB. Bottom-right"
        else:
            options = "A. Left\nB. Right"
            
        return (
            f"Which side is brighter ({hint})?\n"
            f"Choose one:\n"
            f"{options}\n"
            f"Answer with A or B."
        )

def prompt_for_stage2(item: Dict[str,Any]) -> str:
    """第二阶段：置信度评估"""
    task = item["task"]
    if task == "Grid":
        return "Based on your previous answer, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
    elif task == "Gabor":
        return "Based on your previous answer about the stripe orientation, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
    else:  # Color
        return "Based on your previous answer about which side is brighter, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."

def prompt_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# -------------------------
# pack — 生成两阶段问题
# -------------------------

def cmd_pack(catalog: Path, imgroot: Path, out_jsonl: Path) -> None:
    rows = read_json_any(catalog)
    qrows: List[Dict[str,Any]] = []
    
    for it in rows:
        task = it["task"]
        mod = GEN_MAP.get(task)
        params = it.get("params", {})
        
        # 确保图片存在
        out_dir = imgroot / task
        ensure_dir(out_dir)
        ipath = it.get("assets", {}).get("image_path")
        
        # 检查图片是否存在
        if not ipath or not Path(ipath).exists():
            # 尝试从缓存中查找 qid.png
            qid = it["qid"]
            new_filename = f"{qid}.png"
            new_path = out_dir / new_filename
            
            if new_path.exists():
                # 缓存中找到图片，使用缓存
                ipath = str(new_path.as_posix())
            else:
                # 缓存中也没有，需要重新生成
                original_path = mod.generate(params, str(out_dir))
                
                # 重命名文件
                Path(original_path).rename(new_path)
                ipath = str(new_path.as_posix())
        
        # 创建两个阶段的问题
        # 第一阶段：实验任务
        stage1_prompt = prompt_for_stage1(it)
        qrows.append({
            "qid": f"{it['qid']}_stage1",
            "task": task,
            "image_path": str(Path(ipath).as_posix()),
            "prompt": stage1_prompt,
            "prompt_hash": prompt_hash(stage1_prompt),
            "context": {"params": params, "derived": it.get("derived", {})},
            "stage": 1,
            "original_qid": it["qid"]
        })
        
        # 第二阶段：置信度评估
        stage2_prompt = prompt_for_stage2(it)
        qrows.append({
            "qid": f"{it['qid']}_stage2",
            "task": task,
            "image_path": str(Path(ipath).as_posix()),
            "prompt": stage2_prompt,
            "prompt_hash": prompt_hash(stage2_prompt),
            "context": {"params": params, "derived": it.get("derived", {})},
            "stage": 2,
            "original_qid": it["qid"]
        })
    
    write_jsonl(out_jsonl, qrows)
    print(f"[pack] wrote {len(qrows)} two-stage questions -> {out_jsonl}")
    print(f"[pack] {len(qrows)//2} original questions -> {len(qrows)} stage questions")

# -------------------------
# feed — 调模型
# -------------------------

def mock_answer_stage1(item: Dict[str,Any]) -> Tuple[str,int]:
    """第一阶段：离线 mock 生成实验任务答案"""
    rng = random.Random(int(hashlib.md5(item["qid"].encode()).hexdigest(), 16))
    ctx = item.get("context", {})
    d = ctx.get("derived", {})
    gt = d.get("gt")
    
    if isinstance(gt, dict) and "more_symbol" in gt:  # Grid
        a = gt["more_symbol"]
        symA = gt.get("symA", "A")
        symB = gt.get("symB", "B")
        ans = "A" if a == "symA" else "B"
    elif isinstance(gt, str):  # Color
        # 根据布局和gt确定A或B
        layout = ctx.get("params", {}).get("layout", "LR")
        if layout == "LR":
            ans = "A" if gt == "left" else "B"
        elif layout == "UD":
            ans = "A" if gt == "top" else "B"
        elif layout == "DRUL":
            ans = "A" if gt == "bottom-left" else "B"
        elif layout == "DLUR":
            ans = "A" if gt == "top-left" else "B"
        else:
            ans = "A" if gt == "left" else "B"
    else:  # Gabor
        ans = "A" if d.get("gt", "vertical") == "vertical" else "B"
    
    return ans, rng.randint(300, 1200)

def mock_answer_stage2(item: Dict[str,Any]) -> Tuple[str,int]:
    """第二阶段：离线 mock 生成置信度"""
    rng = random.Random(int(hashlib.md5(item["qid"].encode()).hexdigest(), 16))
    conf = rng.randint(1, 5)
    return f"confidence: {conf}", rng.randint(200, 800)

def openai_answer(item: Dict[str,Any]) -> Tuple[str,int]:
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("openai>=1.0.0 未安装或不可用")
    client = OpenAI()
    with open(item["image_path"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = client.chat.completions.create(
        model=os.environ.get("PERCEPT_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": item["prompt"]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        }],
    )
    text = resp.choices[0].message.content.strip()
    return text, 0

def cmd_feed(pack_jsonl: Path, out_jsonl: Path, engine: str = "mock", workers: int = 4) -> None:
    items = read_json_any(pack_jsonl)
    out: List[Dict[str,Any]] = []
    t0 = time.time()
    
    for it in items:
        if engine == "openai":
            text, lat = openai_answer(it)
        else:
            # 根据阶段选择不同的mock答案
            if it.get("stage") == 1:
                text, lat = mock_answer_stage1(it)
            else:
                text, lat = mock_answer_stage2(it)
        
        out.append({
            "qid": it["qid"],
            "task": it["task"],
            "choice": text,
            "confidence": extract_confidence(text),
            "raw_text": text,
            "latency_ms": lat,
            "stage": it.get("stage", 1),
            "original_qid": it.get("original_qid", it["qid"])
        })
    
    write_jsonl(out_jsonl, out)
    print(f"[feed] wrote {len(out)} responses -> {out_jsonl}")
    print(f"[feed] took {time.time() - t0:.1f}s")

def extract_confidence(text: str) -> int:
    """从文本中提取置信度"""
    match = re.search(r'confidence:\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

# -------------------------
# collect — 解析输出
# -------------------------

def cmd_collect(resp_jsonl: Path, catalog: Path, out_csv: Path) -> None:
    responses = read_json_any(resp_jsonl)
    catalog_data = read_json_any(catalog)
    
    # 创建catalog查找字典
    catalog_dict = {item["qid"]: item for item in catalog_data}
    
    # 按original_qid分组响应
    responses_by_original = {}
    for resp in responses:
        original_qid = resp.get("original_qid", resp["qid"])
        if original_qid not in responses_by_original:
            responses_by_original[original_qid] = {}
        stage = resp.get("stage", 1)
        responses_by_original[original_qid][f"stage{stage}"] = resp
    
    # 生成评估结果
    eval_rows = []
    for original_qid, stage_responses in responses_by_original.items():
        if original_qid not in catalog_dict:
            continue
            
        catalog_item = catalog_dict[original_qid]
        task = catalog_item["task"]
        
        # 获取第一阶段的答案和置信度
        stage1_resp = stage_responses.get("stage1", {})
        stage2_resp = stage_responses.get("stage2", {})
        
        # 判断第一阶段是否正确
        is_correct = check_answer_correct(task, stage1_resp.get("choice", ""), catalog_item)
        
        # 构建评估行
        eval_row = {
            "qid": original_qid,
            "task": task,
            "stage1_choice": stage1_resp.get("choice", ""),
            "stage1_confidence": stage1_resp.get("confidence", 0),
            "stage2_confidence": stage2_resp.get("confidence", 0),
            "is_correct": is_correct,
            "stage1_latency_ms": stage1_resp.get("latency_ms", 0),
            "stage2_latency_ms": stage2_resp.get("latency_ms", 0)
        }
        
        # 添加任务特定字段
        if task == "Grid":
            eval_row.update({
                "grid_level": catalog_item.get("params", {}).get("grid_level", ""),
                "shape": catalog_item.get("params", {}).get("shape", ""),
                "symbol_set": catalog_item.get("params", {}).get("symbol_set", ""),
                "color_pair": catalog_item.get("params", {}).get("color_pair", "")
            })
        elif task == "Gabor":
            eval_row.update({
                "gabor_level": catalog_item.get("params", {}).get("gabor_level", ""),
                "freq": catalog_item.get("params", {}).get("freq", ""),
                "position": catalog_item.get("params", {}).get("position", "")
            })
        elif task == "Color":
            eval_row.update({
                "delta_L": catalog_item.get("params", {}).get("delta_L", ""),
                "layout": catalog_item.get("params", {}).get("layout", ""),
                "word_pair_id": catalog_item.get("params", {}).get("word_pair_id", "")
            })
        
        # 为所有任务添加通用字段（避免字段缺失错误）
        eval_row.update({
            "grid_level": eval_row.get("grid_level", ""),
            "shape": eval_row.get("shape", ""),
            "symbol_set": eval_row.get("symbol_set", ""),
            "color_pair": eval_row.get("color_pair", ""),
            "gabor_level": eval_row.get("gabor_level", ""),
            "freq": eval_row.get("freq", ""),
            "position": eval_row.get("position", ""),
            "delta_L": eval_row.get("delta_L", ""),
            "layout": eval_row.get("layout", ""),
            "word_pair_id": eval_row.get("word_pair_id", "")
        })
        
        eval_rows.append(eval_row)
    
    # 写入CSV
    if eval_rows:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
            writer.writeheader()
            writer.writerows(eval_rows)
    
    print(f"[collect] wrote {len(eval_rows)} evaluation rows -> {out_csv}")

def check_answer_correct(task: str, choice: str, catalog_item: Dict[str,Any]) -> bool:
    """检查答案是否正确"""
    if task == "Grid":
        gt = catalog_item.get("derived", {}).get("gt", {})
        if isinstance(gt, dict) and "more_symbol" in gt:
            correct_symbol = gt["more_symbol"]
            if correct_symbol == "symA":
                return "A" in choice.upper()
            elif correct_symbol == "symB":
                return "B" in choice.upper()
    elif task == "Gabor":
        gt = catalog_item.get("derived", {}).get("gt", "")
        if gt == "vertical":
            return "vertical" in choice.lower() or "A" in choice.upper()
        elif gt == "horizontal":
            return "horizontal" in choice.lower() or "B" in choice.upper()
    elif task == "Color":
        gt = catalog_item.get("derived", {}).get("gt", "")
        layout = catalog_item.get("params", {}).get("layout", "LR")
        
        # 根据布局和gt确定正确答案
        if layout == "LR":
            correct_choice = "A" if gt == "left" else "B"
        elif layout == "UD":
            correct_choice = "A" if gt == "top" else "B"
        elif layout == "DRUL":
            correct_choice = "A" if gt == "bottom-left" else "B"
        elif layout == "DLUR":
            correct_choice = "A" if gt == "top-left" else "B"
        else:
            correct_choice = "A" if gt == "left" else "B"
        
        return correct_choice in choice.upper()
    
    return False

# -------------------------
# 主函数
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="两阶段提问的Catalog Runner")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # gen 命令
    gen_parser = subparsers.add_parser("gen", help="生成图片")
    gen_parser.add_argument("--catalog", type=Path, required=True, help="题目表路径")
    gen_parser.add_argument("--imgroot", type=Path, required=True, help="图片输出根目录")
    gen_parser.add_argument("--only-task", type=str, help="只生成特定任务类型")
    
    # pack 命令
    pack_parser = subparsers.add_parser("pack", help="生成两阶段问题")
    pack_parser.add_argument("--catalog", type=Path, required=True, help="题目表路径")
    pack_parser.add_argument("--imgroot", type=Path, required=True, help="图片根目录")
    pack_parser.add_argument("--out", type=Path, required=True, help="输出questions文件路径")
    
    # feed 命令
    feed_parser = subparsers.add_parser("feed", help="调用模型")
    feed_parser.add_argument("--pack", type=Path, required=True, help="questions文件路径")
    feed_parser.add_argument("--out", type=Path, required=True, help="输出responses文件路径")
    feed_parser.add_argument("--engine", default="mock", choices=["mock", "openai"], help="模型引擎")
    feed_parser.add_argument("--workers", type=int, default=4, help="并发数")
    
    # collect 命令
    collect_parser = subparsers.add_parser("collect", help="收集评估结果")
    collect_parser.add_argument("--resp", type=Path, required=True, help="responses文件路径")
    collect_parser.add_argument("--catalog", type=Path, required=True, help="题目表路径")
    collect_parser.add_argument("--out", type=Path, required=True, help="输出eval文件路径")
    
    args = parser.parse_args()
    
    if args.command == "gen":
        cmd_gen(args.catalog, args.imgroot, args.only_task)
    elif args.command == "pack":
        cmd_pack(args.catalog, args.imgroot, args.out)
    elif args.command == "feed":
        cmd_feed(args.pack, args.out, args.engine, args.workers)
    elif args.command == "collect":
        cmd_collect(args.resp, args.catalog, args.out)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
