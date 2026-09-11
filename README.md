# Tensor Inpainting Agent

一个支持彩图、MSI 与视频多维数据的张量补全研究 Agent。它集成 Matrix、A×₃E、CP、Nonnegative CP、Tucker、BTD、t-SVD、Nonnegative Tucker、Hierarchical Tucker、Tensor Train 和 Tensor Ring，并能自主生成连续 MLP、卷积、高效 Transformer 与混合 PyTorch 候选。

## 目录结构

```text
tensor_inpainting_agent/
├── agents/、core/、tools/、context/  # Agent 框架
├── observability/、skills/           # 追踪与技能支持
└── research_agent/                   # 图像补全研究应用
```

## 快速开始

在本目录中执行：

```bash
python -m pip install -r research_agent/requirements.txt
cp research_agent/.env.example research_agent/.env
```

填写 `research_agent/.env` 后，可以选择三种运行模式：

```bash
# 快速链路检查，最长边 64
./research_agent/scripts/run_ai.sh smoke

# 标准实验，最长边 128
./research_agent/scripts/run_ai.sh full

# 手动指定 TT 分解（也可选其他已支持分解）
./research_agent/scripts/run_ai.sh full --base-model tt

# 手动指定 X = A×₃E 通道模分解
./research_agent/scripts/run_ai.sh full --base-model mode3

# 手动指定 Block-Term Decomposition
./research_agent/scripts/run_ai.sh full --base-model btd

# 标准实验预算，保留原始分辨率
./research_agent/scripts/run_ai.sh original
```

自定义图片：

```bash
./research_agent/scripts/run_ai.sh original \
  --image path/to/image.png
```

MAT 数据可直接输入；MSI/彩图形状为 `[H,W,C]`，视频形状为 `[H,W,T,C]`：

```bash
./research_agent/scripts/run_ai.sh original \
  --image /data/video.mat \
  --mat-key video
```

省略 `--mat-key` 时会优先查找 `data/tensor/image/msi/video/X/x`，否则自动选择最大的合法三维或四维数值变量。空间 mask 为 `[H,W]`，会同时作用于全部波段、帧和通道。

完整参数说明：

```bash
./research_agent/scripts/run_ai.sh --help
```

每次完整运行都会同时保存无损多维 `.npy` 和 `.mat` 补全数据（MAT 变量名为 `data`），并在 `comparison_images/` 中导出 RGB 预览。`report.md` 汇总 PSNR、SSIM、LPIPS、MANIQA、CLIP-IQA 和 MUSIQ；后四项仅适用于 RGB，MSI/视频会明确标记为跳过。

更多设计、实验协议与测试说明见 [research_agent/README.md](research_agent/README.md)。
