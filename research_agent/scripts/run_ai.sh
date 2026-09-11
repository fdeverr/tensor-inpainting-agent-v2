#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
用法：
  ./research_agent/scripts/run_ai.sh smoke [选项]
  ./research_agent/scripts/run_ai.sh full  [选项]
  ./research_agent/scripts/run_ai.sh original [选项]

预设模式：
  smoke  快速检查完整链路：最长边 64，训练 20/20 步，调参 1 次，改进 1 轮。
  full   标准实验预算：最长边 128，方法选择 1000 步，LLM 可在 2000 步上限内调整公平实验。
  original  原图实验：保留原始分辨率，使用 full 的训练预算。

常用选项：
  --image PATH                    输入彩图或 MAT 数据路径
  --mat-key NAME                  MAT 变量名；省略时自动选择
  --image-size N|original         最长边缩放到 N；original 保留原始尺寸
  --mask-type block|random        缺失掩码类型
  --missing-rate FLOAT            缺失比例，范围 (0, 1)
  --seed N                        随机种子
  --base-model METHOD            auto|matrix|mode3|cp|nonnegative_cp|tucker|btd|tsvd|nonnegative_tucker|hierarchical_tucker|tt|tensor_ring
  --method-max-steps N            方法选择阶段最大训练步数
  --fair-max-steps N              LLM 可请求的公平比较训练步数硬上限
  --tuning-trials N               调参次数，范围 1–5
  --max-improvement-rounds N      候选改进轮数，范围 1–2
  --device auto|cpu|cuda          运行设备
  --learned-metrics on|off        是否计算 LPIPS/MANIQA/CLIP-IQA/MUSIQ
  --llm-mode auto|off|required    LLM 使用方式

训练与评估：
  --validation-ratio FLOAT        验证集比例
  --validation-interval N         每 N 步验证一次
  --patience N                    早停耐心值
  --retrieval-top-k N             知识检索数量
  --minimum-psnr-delta FLOAT      候选晋级的最小 PSNR 增益
  --ssim-tolerance FLOAT          候选晋级允许的 SSIM 下降
  --smoke-timeout FLOAT           候选代码 smoke test 超时秒数

输出与高级选项：
  --prompt TEXT
  --output-dir PATH
  --candidate-root PATH
  --approved-root PATH

示例：
  # 使用 smoke 预设
  ./research_agent/scripts/run_ai.sh smoke

  # 保留原始分辨率，使用 full 训练预算
  ./research_agent/scripts/run_ai.sh original --image path/to/image.png

  # 以 full 为基础，但保留原图并手动设定实验预算
  ./research_agent/scripts/run_ai.sh full \
      --image path/to/image.png \
      --image-size original \
      --method-max-steps 1500 \
      --fair-max-steps 3000 \
      --tuning-trials 3

选项同时支持 `--name value` 和 `--name=value` 形式。
EOF
}

die() {
    echo "Error: $*" >&2
    exit 2
}

load_env_file() {
    local env_file="$1"
    local line name value

    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
        [[ "$line" == *=* ]] || die "${env_file} 中存在无效行（应为 KEY=VALUE）。"

        name="${line%%=*}"
        value="${line#*=}"
        name="${name%"${name##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || \
            die "${env_file} 中存在无效变量名: ${name}"

        if [[ ${#value} -ge 2 ]]; then
            if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] || \
               [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
                value="${value:1:${#value}-2}"
            fi
        fi
        export "${name}=${value}"
    done < "$env_file"
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

# 共通默认值。smoke/full 只改动实验预算，后续命令行选项可以逐项覆盖。
IMAGE_PATH="research_agent/assets/example.png"
MAT_KEY=""
MASK_TYPE="block"
MISSING_RATE="0.4"
SEED="42"
VALIDATION_RATIO="0.1"
VALIDATION_INTERVAL="10"
PATIENCE="40"
DEVICE="cuda"
BASE_MODEL="auto"
LEARNED_METRICS="on"
LLM_MODE="required"
RETRIEVAL_TOP_K="8"
MINIMUM_PSNR_DELTA="0.2"
SSIM_TOLERANCE="0.002"
SMOKE_TIMEOUT="10.0"
PROMPT=""
OUTPUT_DIR="research_agent/outputs"
CANDIDATE_ROOT="research_agent/algorithms/candidates"
APPROVED_ROOT="research_agent/algorithms/approved"

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
    smoke)
        MODE="smoke"
        IMAGE_SIZE="64"
        METHOD_MAX_STEPS="20"
        FAIR_MAX_STEPS="20"
        TUNING_TRIALS="1"
        MAX_IMPROVEMENT_ROUNDS="1"
        ;;
    full)
        MODE="full"
        IMAGE_SIZE="128"
        METHOD_MAX_STEPS="1000"
        FAIR_MAX_STEPS="2000"
        TUNING_TRIALS="4"
        MAX_IMPROVEMENT_ROUNDS="2"
        ;;
    original)
        MODE="original"
        IMAGE_SIZE="original"
        METHOD_MAX_STEPS="1000"
        FAIR_MAX_STEPS="2000"
        TUNING_TRIALS="4"
        MAX_IMPROVEMENT_ROUNDS="2"
        ;;
    *)
        die "第一个参数必须是 smoke、full 或 original（使用 --help 查看帮助）。"
        ;;
esac
shift

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi

    if [[ "$1" == --*=* ]]; then
        OPTION="${1%%=*}"
        VALUE="${1#*=}"
        shift
    else
        OPTION="$1"
        [[ "$OPTION" == --* ]] || die "无法识别的参数: ${OPTION}"
        [[ $# -ge 2 && "$2" != --* ]] || die "${OPTION} 后需要一个值。"
        VALUE="$2"
        shift 2
    fi

    [[ -n "$VALUE" ]] || die "${OPTION} 的值不能为空。"
    case "$OPTION" in
        --image) IMAGE_PATH="$VALUE" ;;
        --mat-key) MAT_KEY="$VALUE" ;;
        --image-size) IMAGE_SIZE="$VALUE" ;;
        --mask-type) MASK_TYPE="$VALUE" ;;
        --missing-rate) MISSING_RATE="$VALUE" ;;
        --seed) SEED="$VALUE" ;;
        --base-model) BASE_MODEL="$VALUE" ;;
        --method-max-steps) METHOD_MAX_STEPS="$VALUE" ;;
        --fair-max-steps) FAIR_MAX_STEPS="$VALUE" ;;
        --tuning-trials) TUNING_TRIALS="$VALUE" ;;
        --max-improvement-rounds) MAX_IMPROVEMENT_ROUNDS="$VALUE" ;;
        --validation-ratio) VALIDATION_RATIO="$VALUE" ;;
        --validation-interval) VALIDATION_INTERVAL="$VALUE" ;;
        --patience) PATIENCE="$VALUE" ;;
        --device) DEVICE="$VALUE" ;;
        --learned-metrics) LEARNED_METRICS="$VALUE" ;;
        --llm-mode) LLM_MODE="$VALUE" ;;
        --retrieval-top-k) RETRIEVAL_TOP_K="$VALUE" ;;
        --minimum-psnr-delta) MINIMUM_PSNR_DELTA="$VALUE" ;;
        --ssim-tolerance) SSIM_TOLERANCE="$VALUE" ;;
        --smoke-timeout) SMOKE_TIMEOUT="$VALUE" ;;
        --prompt) PROMPT="$VALUE" ;;
        --output-dir) OUTPUT_DIR="$VALUE" ;;
        --candidate-root) CANDIDATE_ROOT="$VALUE" ;;
        --approved-root) APPROVED_ROOT="$VALUE" ;;
        *) die "未知选项: ${OPTION}（使用 --help 查看支持的选项）。" ;;
    esac
done

if [[ "$LEARNED_METRICS" != "on" && "$LEARNED_METRICS" != "off" ]]; then
    die "--learned-metrics 必须是 on 或 off。"
fi

case "$BASE_MODEL" in
    auto|matrix|mode3|cp|nonnegative_cp|tucker|btd|tsvd|nonnegative_tucker|hierarchical_tucker|tt|tensor_ring) ;;
    *) die "--base-model 不是受支持的张量分解名称。" ;;
esac

if [[ "$IMAGE_SIZE" == "original" ]]; then
    # run.py 将 0 转换为 None，load_rgb_image 因此不会缩放。
    RESOLVED_IMAGE_SIZE="0"
    IMAGE_SIZE_DESCRIPTION="原始分辨率（不缩放）"
elif [[ "$IMAGE_SIZE" =~ ^[0-9]+$ ]] && (( IMAGE_SIZE >= 8 )); then
    RESOLVED_IMAGE_SIZE="$IMAGE_SIZE"
    IMAGE_SIZE_DESCRIPTION="最长边 ${IMAGE_SIZE}px（保留宽高比）"
else
    die "--image-size 必须是不小于 8 的整数，或 original。"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "${PROJECT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"

if [[ "$LLM_MODE" == "required" && ! -f "$ENV_FILE" ]]; then
    echo "Error: environment file not found: ${ENV_FILE}" >&2
    exit 1
fi

if [[ "$LLM_MODE" != "off" && -f "$ENV_FILE" ]]; then
    load_env_file "$ENV_FILE"
fi

if [[ "$LLM_MODE" == "required" ]]; then
    for variable_name in LLM_MODEL_ID LLM_API_KEY LLM_BASE_URL; do
        if [[ -z "${!variable_name:-}" ]]; then
            echo "Error: ${variable_name} is missing or empty in ${ENV_FILE}" >&2
            exit 1
        fi
    done

fi

if [[ -n "${LLM_BASE_URL:-}" ]]; then
    if [[ "$LLM_BASE_URL" != http://* && "$LLM_BASE_URL" != https://* ]]; then
        echo "Error: LLM_BASE_URL must be a plain http(s) URL, not a Markdown link." >&2
        exit 1
    fi
    if [[ "$LLM_BASE_URL" == *"["* || "$LLM_BASE_URL" == *"]"* || \
          "$LLM_BASE_URL" == *"("* || "$LLM_BASE_URL" == *")"* ]]; then
        echo "Error: LLM_BASE_URL contains Markdown link characters: ${LLM_BASE_URL}" >&2
        exit 1
    fi
fi

cd "$REPOSITORY_ROOT"

for framework_file in \
    "core/llm.py" \
    "tools/base.py" \
    "observability/trace_logger.py"; do
    if [[ ! -f "$framework_file" ]]; then
        echo "Error: 本地 Tensor Inpainting Agent Framework 框架不完整，缺少: ${REPOSITORY_ROOT}/${framework_file}" >&2
        echo "请将完整的 tensor_inpainting_agent 仓库（不只是 research_agent 目录）同步到服务器。" >&2
        exit 1
    fi
done

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Error: image file not found: ${IMAGE_PATH}" >&2
    exit 1
fi

RUN_ARGS=(
    --image "$IMAGE_PATH"
    --mask-type "$MASK_TYPE"
    --missing-rate "$MISSING_RATE"
    --seed "$SEED"
    --image-size "$RESOLVED_IMAGE_SIZE"
    --base-model "$BASE_MODEL"
    --method-max-steps "$METHOD_MAX_STEPS"
    --fair-max-steps "$FAIR_MAX_STEPS"
    --tuning-trials "$TUNING_TRIALS"
    --max-improvement-rounds "$MAX_IMPROVEMENT_ROUNDS"
    --validation-ratio "$VALIDATION_RATIO"
    --validation-interval "$VALIDATION_INTERVAL"
    --patience "$PATIENCE"
    --device "$DEVICE"
    --llm-mode "$LLM_MODE"
    --retrieval-top-k "$RETRIEVAL_TOP_K"
    --minimum-psnr-delta "$MINIMUM_PSNR_DELTA"
    --ssim-tolerance "$SSIM_TOLERANCE"
    --smoke-timeout "$SMOKE_TIMEOUT"
    --output-dir "$OUTPUT_DIR"
    --candidate-root "$CANDIDATE_ROOT"
    --approved-root "$APPROVED_ROOT"
)
if [[ -n "$PROMPT" ]]; then
    RUN_ARGS+=(--prompt "$PROMPT")
fi
if [[ -n "$MAT_KEY" ]]; then
    RUN_ARGS+=(--mat-key "$MAT_KEY")
fi
if [[ "$LEARNED_METRICS" == "off" ]]; then
    RUN_ARGS+=(--skip-learned-metrics)
fi

echo "正在运行 Tensor Inpainting Agent"
echo "  模式: ${MODE}（命令行选项已覆盖预设）"
echo "  Python: $(command -v python)"
echo "  输入数据: ${IMAGE_PATH}"
if [[ -n "$MAT_KEY" ]]; then
    echo "  MAT 变量: ${MAT_KEY}"
fi
echo "  图像尺寸: ${IMAGE_SIZE_DESCRIPTION}"
echo "  掩码/缺失率/种子: ${MASK_TYPE}/${MISSING_RATE}/${SEED}"
echo "  基础张量分解: ${BASE_MODEL}"
echo "  方法选择步数 / LLM 公平实验上限: ${METHOD_MAX_STEPS}/${FAIR_MAX_STEPS}"
echo "  调参次数/改进轮数: ${TUNING_TRIALS}/${MAX_IMPROVEMENT_ROUNDS}"
echo "  验证比例/间隔/早停: ${VALIDATION_RATIO}/${VALIDATION_INTERVAL}/${PATIENCE}"
echo "  设备/LLM 模式: ${DEVICE}/${LLM_MODE}"
echo "  感知/IQA 指标: ${LEARNED_METRICS}"
echo "  输出目录: ${OUTPUT_DIR}"
if [[ -n "${LLM_MODEL_ID:-}" ]]; then
    echo "  LLM 模型: ${LLM_MODEL_ID}"
fi
if [[ -n "${LLM_BASE_URL:-}" ]]; then
    echo "  LLM 地址: ${LLM_BASE_URL}"
fi

python -m research_agent.run "${RUN_ARGS[@]}"
