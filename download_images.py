"""从 hf-mirror 下载真假图各若干张，作为 RAG 检索库。

数据集：Hemg/AI-Generated-vs-Real-Images-Datasets
  - AiArtData/AiArtData/  → AI 生成（标签 FAKE=1）
  - RealArt/RealArt/      → 真实照片（标签 REAL=0）

只取均匀采样的 N 张，避免全量下载。
"""

import io
import json
import random
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO = "Hemg/AI-Generated-vs-Real-Images-Datasets"
API = f"https://hf-mirror.com/api/datasets/{REPO}/tree/main?recursive=true"
RAW = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/"
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ROOT = Path(__file__).parent / "dataset"


def list_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        tree = json.load(r)
    fake, real = [], []
    for e in tree:
        if e.get("type") != "file":
            continue
        p = e["path"]
        if not p.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if p.startswith("AiArtData/AiArtData/"):
            fake.append(p)
        elif p.startswith("RealArt/RealArt/"):
            real.append(p)
    return fake, real


def download_one(path: str, dest: Path) -> bool:
    url = RAW + urllib.parse.quote(path)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data))
        img.verify()                       # 校验不是坏图
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def fetch(paths, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    tried = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {}
        it = iter(paths)
        # 边下边补：失败的跳过继续取下一张，直到凑够 PER_CLASS
        while ok < PER_CLASS and tried < len(paths):
            batch = []
            for _ in range(30):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            for p in batch:
                tried += 1
                name = f"{tag}_{tried:04d}" + Path(p).suffix.lower()
                futures[pool.submit(download_one, p, out_dir / name)] = name
            for f in as_completed(list(futures)):
                if f.result():
                    ok += 1
                else:
                    (out_dir / futures[f]).unlink(missing_ok=True)
                futures.pop(f, None)
            print(f"  {tag}: 成功 {ok}/{PER_CLASS}，尝试 {tried}", flush=True)
    return ok


def main():
    fake, real = list_files()
    print(f"仓库内可用：AI 生成 {len(fake)} 张，真实 {len(real)} 张")
    random.seed(42)
    random.shuffle(fake)
    random.shuffle(real)

    print("下载 AI 生成图 …")
    n_fake = fetch(fake, ROOT / "fake", "fake")
    print("下载真实图 …")
    n_real = fetch(real, ROOT / "real", "real")

    print(f"\n完成：fake {n_fake} 张，real {n_real} 张")
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())
    print(f"总体积：{total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
