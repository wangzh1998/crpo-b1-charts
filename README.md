# CRPO B1 — Δ̃ 轨迹打分分布分析

交互式可视化：LRAT 训练轨迹在 CRPO 前缀增量对数似然（Δ̃）下的分布分析，按 kept / enabler / dropped_other 三组 × narrow / mid / tail / full 四池展开。

在线查看：打开 `index.html`（或本仓库的 GitHub Pages 链接）。

## 内容

- `index.html` — 交互图表（Chart.js，含池切换器、对数轴切换、8 个分析视图）
- `data/b1_allpools.json` — 四池 × 三组的全部统计量（直方图分箱、分位数、假正例率、正侧分布等）
- `scripts/` — 生成上述数据的分析脚本（在 h20 上对 `b1_deltas_full.jsonl` 运行）

## 核心发现

1. **kept 假正例**：LRAT 判 RELEVANT 的轮次里，强有害（Δ̃<−0.5）占约 8–10%，会给 retriever 注入错误梯度。
2. **正侧同质**：kept 与 enabler 的正贡献同量级、同形状；narrow 池里 enabler 正侧 median 甚至反超 kept。
3. **信号稀释**：narrow→tail 三组 mean 全线衰减；full 池"enabler 弱"是长轨迹稀释的假象。
4. 三组不是"三档质量"，而是二值 relevance 标签的双向噪声；CRPO 连续 Δ̃ 权重是对其的修正。

数据为 Δ̃ 统计量，不含原始检索语料或问答内容。
