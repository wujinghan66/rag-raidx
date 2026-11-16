# RAIDX RAG 模块 · 可运行实现

对 ACM MM 2025 论文 **RAIDX: A Retrieval-Augmented Generation and GRPO Reinforcement Learning Framework for Explainable Deepfake Detection** 中 RAG 模块（论文 Figure 3）的独立复现。

论文未公开源码，本目录是按论文描述重建的等价实现，用于理解与教学。

---

## 它到底做什么

论文的 RAG 模块只有三步，没有任何神经网络训练：

1. **建索引**：训练集每张图经 ViT 编码成向量，存进 FAISS
2. **检索**：查询图编码后，找最相似的 k 张
3. **拼句子**：数这 k 张里有几张真几张假，拼成一句话塞进 prompt

第三步输出的就是下面这句（与论文 Figure 3 黄框逐字一致）：

> Among the k reference images most similar to the current image,
> n_real are labeled as REAL, and n_fake are labeled as FAKE.
> This distribution of authenticity among similar images may help assess
> the current image, but please make your judgment primarily based on the
> features of the image itself.

**关键认识**：RAG 在这里不产生新知识，它只是把"近邻的标签分布"换个形式喂回模型。
本质是 kNN 投票，不是知识注入。

---

## 目录结构

```
rag_raidx/
├── rag_demo.py          # 主程序：建索引 / 检索 / 拼 prompt
├── download_images.py   # 从 hf-mirror 下载演示图片库
├── requirements.txt
├── README.md
├── dataset/
│   ├── real/            # 真图，标签 0（210 张）
│   └── fake/            # AI 生成图，标签 1（214 张）
├── index/               # 索引缓存（自动生成，可删）
│   ├── feats.npy
│   ├── labels.npy
│   └── meta.json
└── .venv/               # 虚拟环境
```

---

## 环境

| 项 | 版本 / 体积 |
|---|---|
| Python | 3.13 |
| torch | 2.15.0+cpu（约 1.3 GB） |
| transformers | 5.23.0 |
| numpy / pillow | 2.5.5 / 13.0.0 |
| 虚拟环境总体积 | 约 1.5 GB |

> **注意**：HuggingFace 直连在本机不通，模型下载与图片下载都走 `hf-mirror.com`。

---

## 使用方法

所有命令都在本目录下执行（Windows）。

```bash
# 交互模式（推荐）：运行后自己输入图片路径，可连续查多张
.venv\Scripts\python.exe rag_demo.py --k 8

# 首次：建索引（会自动下载 CLIP 模型，约 600 MB）
set HF_ENDPOINT=https://hf-mirror.com
.venv\Scripts\python.exe rag_demo.py --stats

# 直接指定查询图
.venv\Scripts\python.exe rag_demo.py --query dataset/fake/fake_0001.png --k 8

# 换个轻量后端（不需要 torch，只看流程）
.venv\Scripts\python.exe rag_demo.py --backend hist

# 强制重建索引
.venv\Scripts\python.exe rag_demo.py --rebuild

# 留一法评测：单靠检索投票能到多少准确率
.venv\Scripts\python.exe rag_demo.py --eval --k 8
```

### 交互模式的三种输入

运行 `rag_demo.py`（不带 `--query`）会停在提示符 `图片路径 >` 等你输入：

| 输入 | 行为 |
|---|---|
| `dataset/fake/fake_0006.png` | 相对路径，按项目根目录解析 |
| 把图片**从资源管理器拖进终端** | 自动填入完整路径（最方便） |
| 直接回车 | 随机抽一张 |
| `q` | 退出 |

查完一张会继续提示下一张，**不用重启程序**（每次重启都要重新加载模型，约 10 秒）。

> 注意：交互模式要求 stdin 是终端。用 VS Code 的「启动调试」运行——配置里已经把
> `console` 设为 `integratedTerminal`，所以能正常输入。若用管道重定向输入，程序会自动
> 退回"随机抽 3 张"的行为。

PowerShell 里设置环境变量用 `$env:HF_ENDPOINT="https://hf-mirror.com"`。

---

## 可视化报告

指定任意一张图，程序会生成一个**自包含的 HTML 报告**（图片以 base64 内嵌，单文件可随意拷贝），
并自动用浏览器打开：

```bash
.venv\Scripts\python.exe rag_demo.py --query dataset/fake/fake_0294.jpg --k 8
```

报告内容：

- 左侧大图 = 查询图，右侧列出它的真实标签、检索耗时、邻居构成
- 下方网格 = Top-k 邻居，每张标注 **相似度数字 + REAL/FAKE 徽章 + 相似度条 + 文件名**
- 中间黄框 = 实际注入 prompt 的那句参考信息

报告存在 `output/report_<图名>.html`，可反复打开，不自动打开时加 `--no-open`。

---

## 实测结果

**留一法评测（k=8，424 张，纯邻居投票，无任何模型推理）**

| 指标 | 数值 |
|---|---|
| 投票准确率 | **82.31%** |
| REAL 召回 | 92.38% |
| FAKE 召回 | 72.43% |
| 邻居纯度 | 77.33%（Top-8 中与查询同标签的比例） |
| 检索耗时 | 0.21–0.36 ms/次 |

这个数字值得注意：**一个不含任何神经网络的 kNN 投票就能到 82%**。

原因不神秘——数据集里真实照片和 AI 绘画在色调、构图上本身差异明显，CLIP 很容易找对邻居。
这也印证了一个判断：**RAIDX 里 RAG 消融只涨 1.45%，是因为"检索近邻"能提供的信息本来就有限**。

同时它暴露了短板：FAKE 召回（72.4%）明显低于 REAL（92.4%），
说明 AI 生成图之间风格差异大，容易被误判成真实照片的近邻。

---

## 两种后端

| | `clip`（默认） | `hist` |
|---|---|---|
| 依赖 | torch + transformers | 仅 numpy + pillow |
| 向量维度 | 512 | 112 |
| 模型 | CLIP ViT-B/32，约 600 MB | 无 |
| 编码速度（CPU） | ~60 ms/张 | ~2 ms/张 |
| "相似"含义 | 语义相似 | 颜色分布 + 构图相似 |

两种后端除编码方式外，检索、计数、拼句子的代码完全一致。

---

## 数据集

来源：[`Hemg/AI-Generated-vs-Real-Images-Datasets`](https://hf-mirror.com/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)

- `dataset/fake/` ← 仓库 `AiArtData/AiArtData/`，AI 生成图（214 张）
- `dataset/real/` ← 仓库 `RealArt/RealArt/`，真实照片（210 张）

重新下载：`.venv\Scripts\python.exe download_images.py 200`（数字为每类目标张数）

---

## 与论文的差异（重要）

本实现是**教学版**，和论文有四处差异，读结论时要注意：

1. **编码器不同**。论文只写 "ViT"，未指明型号；这里用 CLIP ViT-B/32。
2. **k 未定**。论文没给 k 值，这里默认 8。
3. **索引规模小**。论文用约 14 万张训练图，这里只有 424 张，检索统计的稳定性差很多。
4. **数据集不同**。论文用 SID-Set（FLUX 生成的合成图）；这里是网上的 AI 绘画与真实照片，分布不同。

更根本的一点：无论用 CLIP 还是直方图，检索的都是**视觉/语义相似**，而检测伪造真正需要的是**同源生成器痕迹相似**。这个错配是论文本身的问题，不是实现的问题。

---

## 常见报错

### `SSL: UNEXPECTED_EOF_WHILE_READING` / `WinError 10054`

启动时刷出一堆 `Retrying in 1s [Retry 1/5]`，像卡住了。

**这不是代码错误**。模型权重其实已经加载成功（会看到 `Loading weights: 100%`），
卡住的是 transformers 启动时去云端查 `additional_chat_templates` 这个可选文件，
而 hf-mirror 的 SSL 不稳定，于是重试 5 次（1+2+4+8+16 秒）。

**已修复**：`rag_demo.py` 现在检测到模型已在本地缓存，就自动切离线模式
（`HF_HUB_OFFLINE=1` + `local_files_only=True`），完全不再联网。

若哪天要重新下载模型，删掉 `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32`
并去掉 `rag_demo.py` 顶部的离线设置即可。

### `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`

transformers 5.x 把 `get_image_features()` 的返回值从张量改成了对象，取 `.pooler_output`。
代码里已做 4.x / 5.x 双向兼容。

---

## 复现时踩过的三个坑

1. **查询图必须从候选里排除**。否则它会检索到自己（相似度 1.000），标签统计直接失真。
2. **必须先 L2 归一化再用内积**。不归一化时向量模长会主导相似度，结果没有意义。
3. **14 万条以内不必用 FAISS**。14 万 × 512 × 4 B ≈ 287 MB，一次矩阵乘法几十毫秒即可。FAISS 的优势在百万级以上。
