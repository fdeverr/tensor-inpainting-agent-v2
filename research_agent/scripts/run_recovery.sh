#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ==================== 直接修改这里的实验参数 ====================
# 命令行参数优先于这里的默认值。类型名称区分大小写。
DATASET_ROOT="${REPOSITORY_ROOT}/../Multi_dimensional_data"
DATA_TYPES=("Image" "MSI" "Video" "audio")   # 只跑 MSI：DATA_TYPES=("MSI")
REPRESENTATIVES=()                          # 如 ("MSI=xxx.mat" "Image=xxx.mat")
MASK_TYPE="random"                         # random | block | slices（兼容 sildes）
MISSING_RATE="0.4"
SEED="42"
IMAGE_SIZE="128"                           # 0 或 original 保留原尺寸；不影响音频
AUDIO_FRAME_SIZE="256"
BASE_MODEL="auto"
DEVICE="auto"
LLM_MODE="auto"

# 代表样本：候选结构/超参数搜索；每轮晋升仍由同类全部样本决定
EVOLUTION_STEPS="1500"
METHOD_MAX_STEPS=""                        # run_ai 同名参数；空值跟随 EVOLUTION_STEPS
VALIDATION_INTERVAL="10"                   # 每 N 步检查一次 GT
PATIENCE="20"                              # 连续 N 次检查未改善后早停
IMPROVEMENT_ROUNDS="5"
MAX_IMPROVEMENT_ROUNDS=""                  # run_ai 同名参数；空值跟随 IMPROVEMENT_ROUNDS
TUNING_TRIALS="4"
METHOD_MAX_STEPS_CEILING=""                 # 空值 = EVOLUTION_STEPS；更大时允许自动扩展
FAIR_MAX_STEPS=""                          # 空值 = EVOLUTION_STEPS；候选公平比较硬上限
TUNING_NEAR_LIMIT_RATIO="0.9"
TUNING_EXPANSION_FACTOR="2.0"
FAIR_LEARNING_RATES="0.001,0.01,0.1"
FAIR_LEARNING_RATE_REFINEMENT="on"
FAIR_LEARNING_RATE_REFINEMENT_FACTOR="3.0"
ABLATION_SCREEN_TRIALS="1"
ABLATION_SCREEN_MAX_STEPS="300"

# 基础分解方法：LLM/规则组建前 N 个短名单，再用同预算数值预赛决定方法
METHOD_SHORTLIST_SIZE="3"                   # 2–5
SCREENING_TRIALS="2"                        # 1–3
SCREENING_MAX_STEPS=""                      # 空值 = min(200, EVOLUTION_STEPS)
SCREENING_PATIENCE="10"

# SIREN 独立对比
SIREN_COMPARISON="on"
SIREN_MAX_STEPS=""                          # 空值 = EVOLUTION_STEPS
SIREN_TUNING_TRIALS="4"                     # 1–4
SIREN_VALIDATION_INTERVAL="25"
SIREN_PATIENCE="20"

# 整类数据集：进化前对比算法、每轮候选/当前算法及最终版本共用评测参数
EVALUATION_STEPS="1000"
EVALUATION_VALIDATION_INTERVAL="10"
EVALUATION_PATIENCE="0"                     # 0 关闭早停；正整数 = 连续验证未改善次数

# 经验检索、晋级条件、指标开关
RETRIEVAL_TOP_K="8"
MINIMUM_PSNR_DELTA="0.2"
MINIMUM_NMSE_DELTA="0.0"                    # 音频：NMSE 降低量须严格大于此值
SSIM_TOLERANCE="0.002"
SMOKE_TIMEOUT="10.0"
LPIPS="off"                                # 非音频 PSNR/SSIM；音频仅 NMSE；LPIPS 仅 Image
FULL_REFERENCE_METRICS=""                  # run_ai 同名开关；空值跟随 LPIPS（仅 Image）
NO_REFERENCE_METRICS="off"                 # MANIQA/CLIP-IQA/MUSIQ，仅 Image
SELECTION_VISUAL_ASSESSMENT="off"           # 分解选择前视觉观察，仅 Image
MUTATION_VISUAL_ASSESSMENT="off"            # 变异阶段视觉观察，仅 Image
PROMPT=""                                 # 附加研发要求；应用于每个所选类型

# 输出及持久化路径
OUTPUT_DIR="research_agent/outputs/recovery"
HISTORY_ROOT="research_agent/algorithms/history"
CANDIDATE_ROOT="research_agent/algorithms/candidates"
APPROVED_ROOT="research_agent/algorithms/approved"
# ==================== 可调参数结束 ====================

usage() {
    cat <<'EOF'
用法：
  bash research_agent/scripts/run_recovery.sh [选项]

默认数据目录：项目同一级的 Multi_dimensional_data。
默认运行四类；--data-types 可以只选择一类，也可以选择多类。
所有实验选项直接传给 research_agent.run_recovery，使用相同的参数名。
可以直接编辑脚本顶部的参数区；命令行参数会覆盖顶部配置。

常用选项：
  --dataset-root PATH             覆盖默认数据目录
  --data-types Image|MSI|Video|audio [更多类型...]
  --mask-type random|block|slices  缺失类型（兼容 sildes）
  --missing-rate FLOAT            缺失率，默认 0.4
  --representative TYPE=FILE       指定该类研发样本，可重复传入
  --image-size N                  非音频空间最长边，默认 128；0 保留原尺寸
  --audio-frame-size N            音频分帧长度，默认 256
  --evolution-steps N             进化训练预算，默认 1500
  --method-max-steps N            同 evolution-steps，兼容 run_ai 参数名
  --evaluation-steps N            数据集评测预算，默认 1000
  --validation-interval N         研发训练 GT 验证间隔，默认 10 步
  --patience N                    研发训练早停耐心，默认 20 次验证
  --evaluation-validation-interval N  数据集评测 GT 验证间隔，默认 10 步
  --evaluation-patience N         评测早停耐心，默认 0（关闭早停）
  --method-max-steps-ceiling N     方法选择自动扩展硬上限
  --fair-max-steps N              候选公平比较硬上限
  --tuning-near-limit-ratio F      自动扩展触发比例
  --tuning-expansion-factor F      自动扩展倍数
  --fair-learning-rates CSV       学习率搜索，如 0.001,0.01,0.1
  --no-fair-learning-rate-refinement  关闭学习率精搜
  --fair-learning-rate-refinement-factor F  学习率精搜倍数
  --ablation-screen-trials N       消融预赛 trial 数
  --ablation-screen-max-steps N    消融预赛步数
  --method-shortlist-size N        基线预赛后追加预算的家族数，2–5
  --screening-trials N            基线预赛 trial 数，1–3
  --screening-max-steps N          基线预赛步数
  --screening-patience N           基线预赛早停耐心
  --siren-max-steps N             SIREN 训练步数
  --siren-tuning-trials N          SIREN 调参次数，1–4
  --siren-validation-interval N    SIREN GT 验证间隔
  --siren-patience N              SIREN 早停耐心
  --improvement-rounds N          进化轮数，默认 5
  --max-improvement-rounds N      同 improvement-rounds，兼容 run_ai 参数名
  --tuning-trials N               调参次数，默认 4
  --llm-mode auto|off|required    默认 auto；读取 research_agent/.env
  --device auto|cpu|cuda          默认 auto
  --lpips                        开启 Image 的 LPIPS，默认关闭
  --full-reference-metrics on|off 同 LPIPS 开关，兼容 run_ai 参数名
  --no-reference-metrics on|off   MANIQA/CLIP-IQA/MUSIQ，仅 Image
  --selection-visual-assessment on|off  分解选择前视觉观察，仅 Image
  --mutation-visual-assessment on|off   变异阶段视觉观察，仅 Image
  --no-lpips                     关闭 LPIPS（覆盖顶部设置）
  --siren-comparison             开启 SIREN 对比
  --skip-siren-comparison         关闭 SIREN 对比
  --inventory-only               只检查数据，不训练
  --evaluate-only                只评测历史算法，不进化
  --output-dir PATH              结果目录
  --history-root PATH            历史冠军算法库
  --candidate-root PATH          候选算法目录
  --approved-root PATH           晋级算法目录
  --retrieval-top-k N             知识检索数量
  --minimum-psnr-delta FLOAT      候选晋级最小 PSNR 增益
  --minimum-nmse-delta FLOAT      音频晋级最小 NMSE 降低量（默认 0，必须严格改善）
  --ssim-tolerance FLOAT          允许的 SSIM 下降
  --smoke-timeout FLOAT           候选代码检查超时秒数
  --prompt TEXT                  附加算法研发要求

run_ai 中的开关语法（如 --siren-comparison off 和
--fair-learning-rate-refinement off）也可使用，支持 --name=value。
Recovery 从数据集 GT 出发：--image 改用 --representative TYPE=FILE，
MAT 的 GT 固定读取 Ohsi，不使用单样本入口的 --mat-key 自动选变量。

示例：
  bash research_agent/scripts/run_recovery.sh --data-types MSI --llm-mode required
  bash research_agent/scripts/run_recovery.sh --data-types Video --mask-type slices --missing-rate 0.6
  bash research_agent/scripts/run_recovery.sh --data-types Image audio
  bash research_agent/scripts/run_recovery.sh --data-types MSI --evaluate-only

可通过 PYTHON_BIN 指定解释器（默认 python3），例如：
  PYTHON_BIN=/path/to/venv/bin/python bash research_agent/scripts/run_recovery.sh --data-types MSI
相对路径按项目根目录解析；脚本可以从任意工作目录启动。
EOF
}

for argument in "$@"; do
    case "$argument" in
        -h|--help) usage; exit 0 ;;
    esac
done

ENV_FILE="${REPOSITORY_ROOT}/research_agent/.env"

# Parse KEY=VALUE without sourcing arbitrary shell code from the credentials file.
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
        if [[ "$line" != *=* ]]; then
            echo "Error: ${ENV_FILE} 中存在无效行（应为 KEY=VALUE）。" >&2
            exit 2
        fi
        name="${line%%=*}"
        value="${line#*=}"
        name="${name%"${name##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        if [[ ! "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            echo "Error: ${ENV_FILE} 中存在无效变量名。" >&2
            exit 2
        fi
        if [[ ${#value} -ge 2 ]]; then
            if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] || \
               [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
                value="${value:1:${#value}-2}"
            fi
        fi
        export "${name}=${value}"
    done < "$ENV_FILE"
fi

cd -- "$REPOSITORY_ROOT"
[[ -z "$METHOD_MAX_STEPS" ]] || EVOLUTION_STEPS="$METHOD_MAX_STEPS"
[[ -z "$MAX_IMPROVEMENT_ROUNDS" ]] || IMPROVEMENT_ROUNDS="$MAX_IMPROVEMENT_ROUNDS"
[[ -z "$FULL_REFERENCE_METRICS" ]] || LPIPS="$FULL_REFERENCE_METRICS"
RUN_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --data-types "${DATA_TYPES[@]}"
    --mask-type "$MASK_TYPE" --missing-rate "$MISSING_RATE" --seed "$SEED"
    --image-size "$IMAGE_SIZE" --audio-frame-size "$AUDIO_FRAME_SIZE"
    --base-model "$BASE_MODEL" --device "$DEVICE" --llm-mode "$LLM_MODE"
    --evolution-steps "$EVOLUTION_STEPS" --evaluation-steps "$EVALUATION_STEPS"
    --validation-interval "$VALIDATION_INTERVAL" --patience "$PATIENCE"
    --evaluation-validation-interval "$EVALUATION_VALIDATION_INTERVAL"
    --evaluation-patience "$EVALUATION_PATIENCE"
    --improvement-rounds "$IMPROVEMENT_ROUNDS" --tuning-trials "$TUNING_TRIALS"
    --tuning-near-limit-ratio "$TUNING_NEAR_LIMIT_RATIO"
    --tuning-expansion-factor "$TUNING_EXPANSION_FACTOR"
    --fair-learning-rates "$FAIR_LEARNING_RATES"
    --fair-learning-rate-refinement-factor "$FAIR_LEARNING_RATE_REFINEMENT_FACTOR"
    --ablation-screen-trials "$ABLATION_SCREEN_TRIALS"
    --ablation-screen-max-steps "$ABLATION_SCREEN_MAX_STEPS"
    --method-shortlist-size "$METHOD_SHORTLIST_SIZE"
    --screening-trials "$SCREENING_TRIALS" --screening-patience "$SCREENING_PATIENCE"
    --siren-tuning-trials "$SIREN_TUNING_TRIALS"
    --siren-validation-interval "$SIREN_VALIDATION_INTERVAL" --siren-patience "$SIREN_PATIENCE"
    --retrieval-top-k "$RETRIEVAL_TOP_K" --minimum-psnr-delta "$MINIMUM_PSNR_DELTA"
    --minimum-nmse-delta "$MINIMUM_NMSE_DELTA"
    --ssim-tolerance "$SSIM_TOLERANCE" --smoke-timeout "$SMOKE_TIMEOUT"
    --output-dir "$OUTPUT_DIR" --history-root "$HISTORY_ROOT"
    --candidate-root "$CANDIDATE_ROOT" --approved-root "$APPROVED_ROOT"
    --prompt "$PROMPT"
)
if [[ ${#REPRESENTATIVES[@]} -gt 0 ]]; then
    for representative in "${REPRESENTATIVES[@]}"; do
        RUN_ARGS+=(--representative "$representative")
    done
fi
[[ -z "$METHOD_MAX_STEPS_CEILING" ]] || RUN_ARGS+=(--method-max-steps-ceiling "$METHOD_MAX_STEPS_CEILING")
[[ -z "$FAIR_MAX_STEPS" ]] || RUN_ARGS+=(--fair-max-steps "$FAIR_MAX_STEPS")
[[ -z "$SCREENING_MAX_STEPS" ]] || RUN_ARGS+=(--screening-max-steps "$SCREENING_MAX_STEPS")
[[ -z "$SIREN_MAX_STEPS" ]] || RUN_ARGS+=(--siren-max-steps "$SIREN_MAX_STEPS")
for setting in LPIPS SIREN_COMPARISON FAIR_LEARNING_RATE_REFINEMENT NO_REFERENCE_METRICS SELECTION_VISUAL_ASSESSMENT MUTATION_VISUAL_ASSESSMENT; do
    case "${!setting}" in
        on|off) ;;
        *) echo "Error: ${setting} 必须是 on 或 off。" >&2; exit 2 ;;
    esac
done
RUN_ARGS+=(--no-reference-metrics "$NO_REFERENCE_METRICS"
           --selection-visual-assessment "$SELECTION_VISUAL_ASSESSMENT"
           --mutation-visual-assessment "$MUTATION_VISUAL_ASSESSMENT")
if [[ "$LPIPS" == "on" ]]; then RUN_ARGS+=(--lpips); else RUN_ARGS+=(--no-lpips); fi
if [[ "$SIREN_COMPARISON" == "on" ]]; then RUN_ARGS+=(--siren-comparison); else RUN_ARGS+=(--no-siren-comparison); fi
if [[ "$FAIR_LEARNING_RATE_REFINEMENT" == "on" ]]; then
    RUN_ARGS+=(--fair-learning-rate-refinement)
else
    RUN_ARGS+=(--no-fair-learning-rate-refinement)
fi
# Later command-line options override script defaults through argparse.
exec "$PYTHON_BIN" -m research_agent.run_recovery "${RUN_ARGS[@]}" "$@"
