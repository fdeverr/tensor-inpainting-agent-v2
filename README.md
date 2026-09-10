# Tensor Inpainting Agent

一个使用张量分解基线，并能自主生成连续 MLP、卷积、高效 Transformer 与混合 PyTorch 候选的图像补全研究 Agent。

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

# 标准实验预算，保留原始分辨率
./research_agent/scripts/run_ai.sh original
```

自定义图片：

```bash
./research_agent/scripts/run_ai.sh original \
  --image path/to/image.png
```

完整参数说明：

```bash
./research_agent/scripts/run_ai.sh --help
```

每次完整运行都会在顶层结果目录的 `comparison_images/` 中导出破损输入、Manhattan 插值、张量基线和已评估候选的效果图，`report.md` 中也会并排显示。

更多设计、实验协议与测试说明见 [research_agent/README.md](research_agent/README.md)。
