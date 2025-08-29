#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从“现有 JSON 题目表（catalog_*.json/.jsonl）”直接跑完整工程：
  gen     — 逐题调用你的三套生成器生成图片（缓存命中则跳过）
  pack    — 生成 questions.jsonl（图 + prompt + 上下文）
  feed    — 喂给模型（默认 mock，可切 openai）
  collect — 解析模型输出并与题目表/GT 合并，导出 eval.csv

要求：
  generators/
    ├─ grid_levels.py         # 你提供的 Grid（XO 网格）新难度梯度版
    ├─ gabor.py               # 你提供的 Gabor 新难度梯度版
    └─ color_shading.py       # 你提供的 Color（明暗）模块

用法（PowerShell 示例）：
  python .\catalog_runner.py gen     --catalog .\catalog_2688_newspec.json --imgroot .\cache\images
  python .\catalog_runner.py pack    --catalog .\catalog_2688_newspec.json --imgroot .\cache\images --out .\questions.jsonl
  python .\catalog_runner.py feed    --pack .\questions.jsonl --out .\responses.jsonl --engine mock --workers 8
  python .\catalog_runner.py collect --resp .\responses.jsonl --catalog .\catalog_2688_newspec.json --out .\eval.csv

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
        new_filename = f"{qid}.png"
        new_path = out_dir / new_filename
        
        # 检查缓存中是否已经存在图片
        if new_path.exists():
            # 图片已存在，直接使用缓存
            item.setdefault("assets", {})["image_path"] = str(new_path.as_posix())
            n_cached += 1
            continue
        
        # 图片不存在，需要生成
        original_path = mod.generate(params, str(out_dir))
        
        # 重命名文件
        Path(original_path).rename(new_path)
        
        # 更新 image_path 为新文件名
        item.setdefault("assets", {})["image_path"] = str(new_path.as_posix())
        n_generated += 1
    
    # 可选：把带 image_path 的增强版写出去供后续使用
    out_aug = catalog.parent / (catalog.stem + ".with_images.jsonl")
    write_jsonl(out_aug, rows)
    
    print(f"[gen] processed {n_total} items:")
    print(f"  - {n_cached} images found in cache (skipped)")
    print(f"  - {n_generated} images generated")
    print(f"[gen] wrote augmented catalog: {out_aug}")

# -------------------------
# pack — 生成 questions.jsonl
# -------------------------

def prompt_for(item: Dict[str,Any]) -> str:
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
            f"Answer with A or B, then on a new line write 'confidence: N' (1-5)."
        )
    elif task == "Gabor":
        return (
            "Which direction are the stripes closer to: vertical or horizontal?\n"
            "Choose one:\n"
            "A. Vertical\n"
            "B. Horizontal\n"
            "Answer with A or B, then on a new line write 'confidence: N' (1-5)."
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
            f"Answer with A or B, then on a new line write 'confidence: N' (1-5)."
        )

def prompt_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


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
        # prompt
        ptxt = prompt_for(it)
        qrows.append({
            "qid": it["qid"],
            "task": task,
            "image_path": str(Path(ipath).as_posix()),
            "prompt": ptxt,
            "prompt_hash": prompt_hash(ptxt),
            "context": {"params": params, "derived": it.get("derived", {})},
        })
    write_jsonl(out_jsonl, qrows)
    print(f"[pack] wrote {len(qrows)} questions -> {out_jsonl}")

# -------------------------
# feed — 调模型
# -------------------------

def mock_answer(item: Dict[str,Any]) -> Tuple[str,int]:
    """离线 mock：用 GT 生成一个确定性答案 + 3~5 置信。"""
    rng = random.Random(int(hashlib.md5(item["qid"].encode()).hexdigest(), 16))
    ctx = item.get("context", {})
    d = ctx.get("derived", {})
    gt = d.get("gt")
    if isinstance(gt, dict) and "more_symbol" in gt:  # Grid
        a = gt["more_symbol"]
        symA = gt.get("symA", "A")
        symB = gt.get("symB", "B")
        ans = symA if a == "symA" else (symB if a == "symB" else "equal")
    elif isinstance(gt, str):  # Color 已是 canonical 字符串
        ans = gt
    else:  # Gabor: derived.gt 是 vertical/horizontal
        ans = d.get("gt", "vertical")
    conf = rng.randint(3,5)
    return f"{ans}\nconfidence: {conf}", rng.randint(300, 1200)


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
            text, lat = mock_answer(it)
        out.append({
            "qid": it["qid"],
            "task": it["task"],
            "raw_text": text,
            "latency_ms": lat,
            "prompt_hash": it.get("prompt_hash"),
        })
    write_jsonl(out_jsonl, out)
    print(f"[feed] wrote {len(out)} responses -> {out_jsonl} in {time.time()-t0:.1f}s")

# -------------------------
# collect — 解析 & 合并 GT
# -------------------------
CHOICE_RE_CONF = re.compile(r"confidence\s*[:=]\s*(\d)", re.I)
MAP_GRID = [
    (re.compile(r"\bequal\b", re.I), "equal"),
]
MAP_GABOR = [
    (re.compile(r"vertical|vert\b", re.I), "vertical"),
    (re.compile(r"horizontal|horiz\b", re.I), "horizontal"),
]
MAP_COLOR = [
    (re.compile(r"\bleft\b", re.I), "left"),
    (re.compile(r"\bright\b", re.I), "right"),
    (re.compile(r"top\b", re.I), "top"),
    (re.compile(r"bottom\b", re.I), "bottom"),
    (re.compile(r"top-?left\b", re.I), "top-left"),
    (re.compile(r"top-?right\b", re.I), "top-right"),
    (re.compile(r"bottom-?left\b", re.I), "bottom-left"),
    (re.compile(r"bottom-?right\b", re.I), "bottom-right"),
]

def parse_choice(task: str, text: str, item: Dict[str,Any]) -> str:
    t = text.strip()
    if task == "Grid":
        d = item.get("context", {}).get("derived", {}).get("gt", {})
        symA = d.get("symA", "A")
        symB = d.get("symB", "B")
        w = t.lower()
        
        # 优先解析选择题格式 A/B
        if "a" in w.split() or "A" in t.split():
            return symA
        elif "b" in w.split() or "B" in t.split():
            return symB
        
        # 兼容旧格式：直接包含符号名称
        if symA.lower() in w.split():
            return symA
        if symB.lower() in w.split():
            return symB
        for pat, val in MAP_GRID:
            if pat.search(w):
                return val
        return w.split()[0] if w else ""
        
    elif task == "Gabor":
        w = t.lower()
        # 优先解析选择题格式 A/B
        if "a" in w.split() or "A" in t.split():
            return "vertical"
        elif "b" in w.split() or "B" in t.split():
            return "horizontal"
        
        # 兼容旧格式：直接包含方向名称
        for pat, val in MAP_GABOR:
            if pat.search(t):
                return val
        return t.split()[0] if t else ""
        
    else:  # Color
        w = t.lower()
        # 优先解析选择题格式 A/B
        if "a" in w.split() or "A" in t.split():
            # 根据布局确定 A 对应的选项
            lay = item.get("context", {}).get("params", {}).get("layout", "LR")
            if lay == "LR":
                return "left"
            elif lay == "UD":
                return "top"
            elif lay == "DRUL":
                return "bottom-left"
            elif lay == "DLUR":
                return "top-left"
            else:
                return "left"
        elif "b" in w.split() or "B" in t.split():
            # 根据布局确定 B 对应的选项
            lay = item.get("context", {}).get("params", {}).get("layout", "LR")
            if lay == "LR":
                return "right"
            elif lay == "UD":
                return "bottom"
            elif lay == "DRUL":
                return "top-right"
            elif lay == "DLUR":
                return "bottom-right"
            else:
                return "right"
        
        # 兼容旧格式：直接包含位置名称
        for pat, val in MAP_COLOR:
            if pat.search(t):
                return val
        return t.split()[0] if t else ""

def parse_conf(text: str) -> int:
    m = CHOICE_RE_CONF.search(text)
    if not m:
        return 3
    try:
        v = int(m.group(1))
        return max(1, min(5, v))
    except Exception:
        return 3


def cmd_collect(resp_jsonl: Path, catalog: Path, out_csv: Path) -> None:
    resps = read_json_any(resp_jsonl)
    rmap = {r["qid"]: r for r in resps}
    items = read_json_any(catalog)

    headers = [
        "qid","task","choice","confidence","raw_text","latency_ms","is_correct","err_type"
    ]
    # 扩展一些关键信息列便于分析
    headers += ["grid_level","shape","symbol_set","color_pair","gabor_level","freq","position","delta_L","layout","word_pair_id"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for it in items:
            qid = it["qid"]; task = it["task"]
            rr = rmap.get(qid)
            if not rr:
                continue
            text = rr.get("raw_text", "")
            choice = parse_choice(task, text, it)
            conf = parse_conf(text)
            # GT
            d = it.get("derived", {})
            if task == "Grid":
                gt = d.get("gt", {})
                gt_choice = (gt.get("symA") if gt.get("more_symbol") == "symA" else (gt.get("symB") if gt.get("more_symbol") == "symB" else "equal"))
                ok = (choice.lower() == (gt_choice or "").lower())
                err = "ok" if ok else "wrong"
            elif task == "Gabor":
                gt_choice = d.get("gt")
                ok = (choice == gt_choice)
                err = "ok" if ok else "wrong_axis"
            else:
                gt_choice = d.get("gt")
                ok = (choice == gt_choice)
                err = "ok" if ok else "wrong_side"

            row = {
                "qid": qid,
                "task": task,
                "choice": choice,
                "confidence": conf,
                "raw_text": text,
                "latency_ms": rr.get("latency_ms", 0),
                "is_correct": "1" if ok else "0",
                "err_type": err,
                # 附带参数关键信息
                "grid_level": it.get("params", {}).get("grid_level"),
                "shape": it.get("params", {}).get("shape"),
                "symbol_set": it.get("params", {}).get("symbol_set"),
                "color_pair": it.get("params", {}).get("color_pair"),
                "gabor_level": it.get("params", {}).get("gabor_level"),
                "freq": it.get("params", {}).get("freq"),
                "position": it.get("params", {}).get("position"),
                "delta_L": it.get("params", {}).get("delta_L"),
                "layout": it.get("params", {}).get("layout"),
                "word_pair_id": it.get("params", {}).get("word_pair_id"),
            }
            w.writerow(row)
    print(f"[collect] wrote eval -> {out_csv}")

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Run pipeline from existing JSON catalog")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("gen", help="Generate stimuli for all items in catalog")
    p1.add_argument("--catalog", type=Path, required=True)
    p1.add_argument("--imgroot", type=Path, default=Path("cache/images"))
    p1.add_argument("--only-task", type=str, choices=["Grid","Gabor","Color"], default=None)

    p2 = sub.add_parser("pack", help="Build questions.jsonl from catalog + images")
    p2.add_argument("--catalog", type=Path, required=True)
    p2.add_argument("--imgroot", type=Path, default=Path("cache/images"))
    p2.add_argument("--out", type=Path, default=Path("questions.jsonl"))

    p3 = sub.add_parser("feed", help="Send questions to model (mock/openai)")
    p3.add_argument("--pack", type=Path, required=True)
    p3.add_argument("--out", type=Path, default=Path("responses.jsonl"))
    p3.add_argument("--engine", type=str, default="mock", choices=["mock","openai"])
    p3.add_argument("--workers", type=int, default=4)

    p4 = sub.add_parser("collect", help="Parse responses and export eval.csv")
    p4.add_argument("--resp", type=Path, required=True)
    p4.add_argument("--catalog", type=Path, required=True)
    p4.add_argument("--out", type=Path, default=Path("eval.csv"))

    args = ap.parse_args()

    if args.cmd == "gen":
        cmd_gen(args.catalog, args.imgroot, only_task=args.only_task)
    elif args.cmd == "pack":
        cmd_pack(args.catalog, args.imgroot, args.out)
    elif args.cmd == "feed":
        cmd_feed(args.pack, args.out, engine=args.engine, workers=args.workers)
    elif args.cmd == "collect":
        cmd_collect(args.resp, args.catalog, args.out)

if __name__ == "__main__":
    main()
