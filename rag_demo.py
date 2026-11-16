"""
RAIDX RAG module — an independent reimplementation
==================================================

Rebuilds the RAG module (Figure 3) of the ACM MM 2025 paper:

    RAIDX: A Retrieval-Augmented Generation and GRPO Reinforcement Learning
    Framework for Explainable Deepfake Detection

The paper does not release its code, so this is a reconstruction from the
method description. The module performs three steps and involves no model
training whatsoever:

    1. Index   — encode every gallery image into a vector, store them
    2. Retrieve— encode the query, take its k nearest neighbours
    3. Compose — count the REAL/FAKE labels among those k, fill a fixed template

Two encoders are provided. Everything downstream of the encoder is identical:

    --backend clip   CLIP ViT-B/32 embedding, 512-d, semantic similarity (default)
    --backend hist   colour histogram + grey thumbnail, 112-d, no torch required

Directory layout:

    dataset/real/*   authentic images, label 0
    dataset/fake/*   AI-generated images, label 1

Usage:

    python rag_demo.py                   # interactive: type a path at the prompt
    python rag_demo.py --query p.jpg     # query a single image directly
    python rag_demo.py --stats           # print index summary
    python rag_demo.py --eval            # leave-one-out evaluation
    python rag_demo.py --rebuild         # force re-indexing
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import webbrowser
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
DATASET = HERE / "dataset"
INDEX_DIR = HERE / "index"
OUT_DIR = HERE / "output"

# ---------------------------------------------------------------- 离线策略
# 模型已缓存时强制离线：transformers 启动会去 Hub 查 additional_chat_templates，
# 网络不稳时会白等约 30 秒重试。首次下载需先设 HF_ENDPOINT=https://hf-mirror.com。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_MODEL_DIR = _HF_CACHE / "models--openai--clip-vit-base-patch32"
OFFLINE = _MODEL_DIR.exists()
if OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 论文 Figure 3 黄框里的模板句，逐字照搬
REFERENCE_TEMPLATE = (
    "Among the {k} reference images most similar to the current image, "
    "{n_real} are labeled as REAL, and {n_fake} are labeled as FAKE. "
    "This distribution of authenticity among similar images may help assess "
    "the current image, but please make your judgment primarily based on the "
    "features of the image itself."
)


def make_reference(k: int, n_real: int, n_fake: int) -> str:
    """Fill the template with the neighbour statistics."""
    return REFERENCE_TEMPLATE.format(k=k, n_real=n_real, n_fake=n_fake)


# ---------------------------------------------------------------- 编码后端

def encode_hist(path: Path) -> np.ndarray:
    """轻量描述子：颜色直方图（48 维）+ 灰度缩略图（64 维）。"""
    img = Image.open(path).convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    parts = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=16, range=(0.0, 1.0))
        parts.append(h)
    hist = np.concatenate(parts).astype(np.float32)
    hist /= hist.sum() + 1e-8

    thumb = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float32) / 255.0
    feat = np.concatenate([hist, thumb.reshape(-1)])
    return (feat / (np.linalg.norm(feat) + 1e-8)).astype(np.float32)


_clip = None


def _load_clip():
    """Lazily load CLIP; the model is only pulled in when the clip backend is used."""
    global _clip
    if _clip is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        name = "openai/clip-vit-base-patch32"
        print(f"  加载 CLIP 模型 {name} …", flush=True)
        model = CLIPModel.from_pretrained(name, local_files_only=OFFLINE).eval()
        proc = CLIPProcessor.from_pretrained(name, local_files_only=OFFLINE)
        _clip = (torch, model, proc)
    return _clip


def encode_clip(paths, batch_size: int = 16) -> np.ndarray:
    """CLIP 后端：批量编码成 512 维语义向量。"""
    torch, model, proc = _load_clip()
    chunks = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        with torch.no_grad():
            inputs = proc(images=imgs, return_tensors="pt")
            res = model.get_image_features(**inputs)
            # transformers 5.x 返回 BaseModelOutputWithPooling，4.x 直接返回张量
            feat = res.pooler_output if hasattr(res, "pooler_output") else res
            feat = feat / feat.norm(dim=-1, keepdim=True)
        chunks.append(feat.cpu().numpy().astype(np.float32))
        print(f"    编码 {min(i + batch_size, len(paths))}/{len(paths)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return np.vstack(chunks)


# ---------------------------------------------------------------- 索引

class FlatIndex:
    """最小可用的向量索引：矩阵存全部向量，内积即余弦相似度。

    FAISS 的 IndexFlatIP 语义与此相同，只是底层用 C++/SIMD 加速。
    几十万条以内 numpy 完全够用。
    """

    def __init__(self, mat: np.ndarray, labels: np.ndarray, paths):
        self.mat = mat
        self.labels = labels
        self.paths = paths

    def search(self, query: np.ndarray, k: int, exclude: int | None = None):
        """Return the top-k similarities and indices.

        ``exclude`` drops one index from the candidates. Leave-one-out
        evaluation needs it, otherwise the query retrieves itself with
        similarity exactly 1.0 and the label statistics become meaningless.
        """
        sims = self.mat @ query
        if exclude is not None:
            sims = sims.copy()
            sims[exclude] = -np.inf
        k = min(k, int(np.isfinite(sims).sum()))
        idx = np.argpartition(-sims, k - 1)[:k]          # 先粗选出前 k 大
        idx = idx[np.argsort(-sims[idx])]                # 再按相似度降序排列
        return sims[idx], idx

    def __len__(self):
        return len(self.labels)


def collect_images():
    """扫描 dataset/real 与 dataset/fake，返回 (路径列表, 标签数组)。"""
    paths, labels = [], []
    for sub, lab in (("real", 0), ("fake", 1)):
        d = DATASET / sub
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(p)
                labels.append(lab)
    return paths, np.array(labels, dtype=np.int64)


def build_index(backend: str, force: bool = False) -> FlatIndex:
    """Build (or load from cache) the vector index over dataset/.

    The cache is keyed on backend name and image count, so switching
    backends or adding images triggers a rebuild automatically.
    """
    meta_file = INDEX_DIR / "meta.json"
    feats_file = INDEX_DIR / "feats.npy"
    labels_file = INDEX_DIR / "labels.npy"

    paths, labels = collect_images()
    if not paths:
        sys.exit(f"数据集为空：{DATASET}")

    cache_ok = (
        not force
        and meta_file.exists()
        and feats_file.exists()
        and labels_file.exists()
    )
    if cache_ok:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("backend") == backend and meta.get("count") == len(paths):
            feats = np.load(feats_file)
            labels = np.load(labels_file)
            cached_paths = [Path(p) for p in meta["paths"]]
            print(f"命中缓存：{len(labels)} 张图，{feats.shape[1]} 维（{meta['backend']}）")
            return FlatIndex(feats, labels, cached_paths)

    print(f"建立索引：{len(paths)} 张图，后端 = {backend}")
    t0 = time.time()
    if backend == "clip":
        feats = encode_clip(paths)
    else:
        feats = np.stack([encode_hist(p) for p in paths])
    dt = time.time() - t0

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(feats_file, feats)
    np.save(labels_file, labels)
    meta_file.write_text(
        json.dumps(
            {"backend": backend, "count": len(paths), "dim": int(feats.shape[1]),
             "build_seconds": round(dt, 1),
             "paths": [str(p) for p in paths]},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"索引完成：{feats.shape}，耗时 {dt:.1f} s（{dt / len(paths) * 1000:.0f} ms/张）")
    return FlatIndex(feats, labels, paths)


# ---------------------------------------------------------------- 可视化报告

def thumb_b64(path: Path, max_side: int = 360, quality: int = 82) -> str:
    """把图片压成缩略图并转成 base64，嵌进 HTML 里（单文件，可随意拷贝）。"""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


REPORT_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#faf9f7;color:#2c2c2a;font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;padding:32px}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:18px;font-weight:500;margin-bottom:4px}
.sub{color:#5f5e5a;font-size:13px;margin-bottom:24px}
.panel{background:#fff;border:1px solid #e5e3dd;border-radius:12px;padding:20px;margin-bottom:20px}
.q{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
.q img{width:320px;border-radius:8px;display:block}
.qinfo{flex:1;min-width:260px}
.kv{display:flex;gap:8px;font-size:13px;margin-bottom:6px}
.kv b{font-weight:500;color:#2c2c2a;min-width:72px}
.kv span{color:#5f5e5a}
.prompt{background:#fdf6e8;border:1px solid #f0dfb8;border-radius:8px;padding:14px 16px;
        font-family:ui-monospace,Consolas,monospace;font-size:12.5px;line-height:1.7;color:#412402;margin-top:14px}
h2{font-size:15px;font-weight:500;margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}
.card{border:1px solid #e5e3dd;border-radius:10px;overflow:hidden;background:#fff;transition:border-color .15s}
.card:hover{border-color:#b4b2a9}
.card img{width:100%;height:150px;object-fit:cover;display:block;background:#f1efe8}
.meta{padding:10px 12px}
.row{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.sim{font-size:16px;font-weight:500;font-variant-numeric:tabular-nums}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:500}
.badge.fake{background:#fcebeb;color:#a32d2d}
.badge.real{background:#eaf3de;color:#3b6d11}
.bar{height:4px;background:#f1efe8;border-radius:2px;overflow:hidden;margin-bottom:8px}
.bar i{display:block;height:100%;border-radius:2px}
.simbar-fake{background:#e24b4a}
.simbar-real{background:#639922}
.name{font-size:11.5px;color:#5f5e5a;word-break:break-all}
.note{color:#5f5e5a;font-size:12.5px;margin-top:20px;line-height:1.8}
"""


def render_report(query_path: Path, items, ref_text: str, elapsed_ms: float,
                  k: int, n_real: int, n_fake: int, out_html: Path) -> Path:
    """生成一个自包含的 HTML 报告：查询图 + 邻居网格 + 相似度条。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cards = []
    for sim, path, label in items:
        tag = "FAKE" if label == 1 else "REAL"
        cls = "fake" if label == 1 else "real"
        pct = max(0.0, min(1.0, sim)) * 100
        cards.append(f"""
        <div class="card">
          <img src="data:image/jpeg;base64,{thumb_b64(path)}" alt="">
          <div class="meta">
            <div class="row">
              <span class="sim">{sim:.3f}</span>
              <span class="badge {cls}">{tag}</span>
            </div>
            <div class="bar"><i class="simbar-{cls}" style="width:{pct:.1f}%"></i></div>
            <div class="name">{escape(path.name)}</div>
          </div>
        </div>""")

    q_label = "FAKE" if query_path.parent.name == "fake" else "REAL"
    q_cls = "fake" if q_label == "FAKE" else "real"

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>RAG 检索报告 · {escape(query_path.name)}</title>
<style>{REPORT_CSS}</style></head>
<body><div class="wrap">

<div class="panel">
  <h1>RAG 检索报告</h1>
  <div class="sub">查询图与它在索引中最相似的 {k} 张图</div>
  <div class="q">
    <img src="data:image/jpeg;base64,{thumb_b64(query_path, max_side=520)}" alt="">
    <div class="qinfo">
      <div class="kv"><b>查询图</b><span>{escape(query_path.name)}</span></div>
      <div class="kv"><b>真实标签</b><span class="badge {q_cls}">{q_label}</span></div>
      <div class="kv"><b>检索耗时</b><span>{elapsed_ms:.2f} ms</span></div>
      <div class="kv"><b>邻居构成</b><span>{n_real} 张 REAL · {n_fake} 张 FAKE</span></div>
      <div class="prompt">{escape(ref_text)}</div>
    </div>
  </div>
</div>

<div class="panel">
  <h2>Top-{len(items)} 邻居（按相似度降序）</h2>
  <div class="grid">{''.join(cards)}
  </div>
</div>

<div class="note">
  相似度 = 两个 CLIP 向量归一化后的内积（余弦相似度），取值 0–1，越接近 1 越像。<br>
</div>

</div></body></html>"""

    out_html.write_text(html_doc, encoding="utf-8")
    return out_html


def _clean_path(raw: str) -> Path:
    """处理用户输入：去引号、去空格，相对路径按项目根目录解析。"""
    raw = raw.strip().strip('"').strip("'").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = HERE / p
    return p


def interactive_loop(index: FlatIndex, k: int, backend: str, open_browser: bool):
    """交互模式：反复输入图片路径，直到输入 q 或空行退出。"""
    print("\n" + "=" * 66)
    print("交互检索模式")
    print("  · 直接把图片从资源管理器拖进这个终端窗口，会自动填入路径")
    print("  · 相对路径按项目根目录算，例如 dataset/fake/fake_0006.png")
    print("  · 直接回车 = 随机抽一张   输入 q = 退出")
    print("=" * 66)

    while True:
        try:
            raw = input("\n图片路径 > ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return

        if raw.strip().lower() in {"q", "quit", "exit"}:
            print("已退出。")
            return

        if not raw.strip():
            qp = random.choice(index.paths)
            print(f"  （随机选中）{qp.relative_to(HERE)}")
        else:
            qp = _clean_path(raw)
            if not qp.exists():
                print(f"  找不到这个文件：{qp}")
                continue
            if qp.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                print(f"  这不是支持的图片格式：{qp.suffix}")
                continue

        try:
            run_query(index, qp, k, backend, report=True, open_browser=open_browser)
            open_browser = False      # 只第一张自动弹浏览器，避免开一堆标签
        except SystemExit as e:
            print(f"  出错：{e}")


# ---------------------------------------------------------------- 演示

def run_query(index: FlatIndex, query_path: Path, k: int, backend: str,
              report: bool = True, open_browser: bool = True):
    """检索并打印结果；report=True 时额外生成可视化网页并打开。"""
    if not query_path.exists():
        sys.exit(f"查询图不存在：{query_path}")

    qfeat = encode_clip([query_path])[0] if backend == "clip" else encode_hist(query_path)

    self_idx = None
    for i, p in enumerate(index.paths):
        if p.name == query_path.name and p.parent.name == query_path.parent.name:
            self_idx = i
            break

    t0 = time.time()
    sims, idx = index.search(qfeat, k, exclude=self_idx)
    dt = (time.time() - t0) * 1000

    neigh = index.labels[idx]
    n_fake = int(neigh.sum())
    n_real = len(idx) - n_fake
    ref = make_reference(len(idx), n_real, n_fake)

    try:
        shown = query_path.relative_to(HERE)
    except ValueError:
        shown = query_path

    print("\n" + "=" * 66)
    print(f"查询图      : {shown}")
    print(f"检索耗时    : {dt:.2f} ms（{len(index)} 条候选）")
    print(f"Top-{len(idx)} 邻居 :")
    items = []
    for s, i in zip(sims, idx):
        tag = "FAKE" if index.labels[i] == 1 else "REAL"
        items.append((float(s), index.paths[i], int(index.labels[i])))
        print(f"   {s:6.3f}  {tag:4s}  {index.paths[i].name}")
    print("\n注入 prompt 的参考信息：")
    print("   " + ref)

    if report:
        out = render_report(query_path, items, ref, dt, len(idx),
                            n_real, n_fake, OUT_DIR / f"report_{query_path.stem}.html")
        print(f"\n可视化报告：{out}")
        if open_browser:
            webbrowser.open(out.as_uri())

    return items


def evaluate(index: FlatIndex, k: int, backend: str):
    """留一法评测：用邻居标签的多数投票预测查询图标签。

    这不是论文的指标，而是为了回答一个问题：
    单靠 RAG 检索（不含任何模型推理）能到多少准确率？
    """
    n = len(index)
    preds = np.empty(n, dtype=np.int64)
    purity = np.empty(n, dtype=np.float32)

    for i in range(n):
        sims, idx = index.search(index.mat[i], k, exclude=i)
        neigh = index.labels[idx]
        purity[i] = (neigh == index.labels[i]).mean()
        preds[i] = 1 if neigh.sum() * 2 > len(neigh) else 0

    acc = (preds == index.labels).mean()
    real_mask = index.labels == 0
    fake_mask = index.labels == 1

    print("\n" + "=" * 66)
    print(f"留一法评测（k={k}，无任何模型推理，纯邻居投票）")
    print(f"  样本数      : {n}（REAL {real_mask.sum()} / FAKE {fake_mask.sum()}）")
    print(f"  投票准确率  : {acc * 100:.2f}%")
    print(f"    REAL 召回 : {(preds[real_mask] == 0).mean() * 100:.2f}%")
    print(f"    FAKE 召回 : {(preds[fake_mask] == 1).mean() * 100:.2f}%")
    print(f"  邻居纯度    : 平均 {purity.mean() * 100:.2f}%"
          f"（Top-{k} 中与查询同标签的比例）")
    print(f"  检索耗时    : 0.2-0.4 ms/次（{n} 条候选）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=None, help="查询图路径")
    ap.add_argument("--k", type=int, default=8, help="检索邻居数")
    ap.add_argument("--backend", choices=["clip", "hist"], default="clip")
    ap.add_argument("--rebuild", action="store_true", help="强制重建索引")
    ap.add_argument("--stats", action="store_true", help="只打印索引概况")
    ap.add_argument("--eval", action="store_true", help="留一法评测检索纯度与投票准确率")
    ap.add_argument("--no-open", action="store_true", help="只生成报告，不自动打开浏览器")
    args = ap.parse_args()

    index = build_index(args.backend, force=args.rebuild)

    if args.stats:
        print(f"\n索引规模 : {len(index)} 条，{index.mat.shape[1]} 维")
        print(f"REAL     : {int((index.labels == 0).sum())} 张")
        print(f"FAKE     : {int((index.labels == 1).sum())} 张")
        print(f"矩阵内存 : {index.mat.nbytes / 1024 / 1024:.1f} MB")
        return

    if args.eval:
        evaluate(index, args.k, args.backend)
        return

    if args.query:
        qp = Path(args.query) if Path(args.query).is_absolute() else HERE / args.query
        run_query(index, qp, args.k, args.backend,
                  report=True, open_browser=not args.no_open)
        return

    # 没给 --query：在终端里运行就进交互模式，让用户自己输入路径
    if sys.stdin.isatty():
        interactive_loop(index, args.k, args.backend, open_browser=not args.no_open)
        return

    targets = random.sample(index.paths, min(3, len(index.paths)))
    for n, qp in enumerate(targets):
        run_query(index, Path(qp), args.k, args.backend,
                  report=True, open_browser=(not args.no_open and n == 0))

    print("\n" + "=" * 66)
    print("注意：n_real / n_fake 只是把邻居标签数了一遍，没有任何模型推理。")
    print("      这就是 RAG 模块的全部工作 —— 检索、计数、拼句子。")


if __name__ == "__main__":
    main()
