# Qwen3.5-2B SFT Training Runbook

这是当前项目唯一推荐的训练与评测流程。旧 Qwen3-8B 脚本仅用于历史追溯，不要与本流程混用。

## 1. 创建并初始化 4090D 实例

选择带 NVIDIA 驱动、CUDA 和可用 PyTorch GPU 运行时的 4090D 镜像，然后执行：

```bash
git clone <REPOSITORY_URL> Coding-Agent-SFT-Lab
cd Coding-Agent-SFT-Lab

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-qwen3_5-sft.txt
```

如果云端镜像没有 GPU 版 PyTorch，应先按照平台 CUDA 版本安装 PyTorch，再安装上述依赖。最终预检会拒绝
CPU-only PyTorch、版本不匹配、BF16 不可用或显存不足的环境。

## 2. 配置本地模型与运行目录

模型需要预先上传、挂载或由平台缓存准备；训练脚本不会下载模型。

```bash
export MODEL_PATH=/mnt/models/Qwen3.5-2B
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python
export LLAMAFACTORY_CLI=llamafactory-cli

export SMOKE_OUTPUT_DIR=outputs/qwen3_5_2b_smoke_lora
export FORMAL_OUTPUT_DIR=outputs/qwen3_5_2b_lora_sft
export EVAL_OUTPUT_DIR=outputs/qwen3_5_2b_base_vs_sft_eval

mkdir -p logs
```

`MODEL_PATH` 必须是完整的本地 `Qwen/Qwen3.5-2B` 快照，至少包含配置、Tokenizer/Processor 和
Safetensors 权重。

## 3. 最终预检

```bash
MODEL_PATH="${MODEL_PATH}" \
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
FORMAL_OUTPUT_DIR="${FORMAL_OUTPUT_DIR}" \
bash scripts/final_preflight.sh 2>&1 | tee logs/final_preflight.log
```

必须看到最终的 `READY`。预检会验证 CUDA/BF16/显存、依赖版本、全部数据与划分、两套 YAML、Processor、
Chat Template、实际 LoRA 可训练参数及视觉冻结，并运行全部 CPU 测试和语法检查。

## 4. 运行 2-step Smoke

```bash
MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
FORMAL_OUTPUT_DIR="${FORMAL_OUTPUT_DIR}" \
bash scripts/run_qwen3_5_2b_smoke.sh 2>&1 | tee logs/smoke_train.log
```

启动器会在训练前写入 `${SMOKE_OUTPUT_DIR}/experiment_manifest.json`。

## 5. 验证 Smoke Adapter

```bash
python scripts/verify_sft_adapter.py \
  --model-path "${MODEL_PATH}" \
  --adapter-dir "${SMOKE_OUTPUT_DIR}" \
  --expected-steps 2 2>&1 | tee logs/smoke_verify.log
```

训练完成、Adapter 保存、重载、非空生成以及无 NaN/Inf 都是硬门槛；JSON 合法性仅是诊断信息。

## 6. 正式 SFT

只有 Smoke 所有硬门槛通过后才运行：

```bash
MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${FORMAL_OUTPUT_DIR}" \
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
bash scripts/run_qwen3_5_2b_sft.sh 2>&1 | tee logs/formal_train.log
```

正式训练只读取 `coding_agent_train`，`coding_agent_val` 仅用于验证；`coding_agent_test` 不参与训练或调参。
正式输出目录中会生成独立的 `experiment_manifest.json`。

## 7. Base/SFT Test 评测

```bash
MODEL_PATH="${MODEL_PATH}" \
ADAPTER_DIR="${FORMAL_OUTPUT_DIR}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
bash scripts/run_qwen3_5_2b_eval.sh 2>&1 | tee logs/test_eval.log
```

评测只使用 `coding_agent_test`，输出：

- `sample_results.jsonl`：逐样本 Base/SFT 输出、指标和错误类型；
- `summary.json`：机器可读汇总及 Base/SFT 差值；
- `summary.md`：总体和分任务报告。

## 8. 打包并下载结果

```bash
mkdir -p artifacts
tar -czf artifacts/qwen3_5_2b_sft_delivery.tar.gz \
  "${SMOKE_OUTPUT_DIR}" \
  "${FORMAL_OUTPUT_DIR}" \
  "${EVAL_OUTPUT_DIR}" \
  logs
sha256sum artifacts/qwen3_5_2b_sft_delivery.tar.gz \
  | tee artifacts/qwen3_5_2b_sft_delivery.tar.gz.sha256
```

下载 `artifacts/qwen3_5_2b_sft_delivery.tar.gz` 及其 SHA-256 文件。`outputs/`、`logs/` 和 `artifacts/`
都被 Git 忽略，不应提交到仓库。

## 本地 CPU 冻结检查

没有模型或 GPU 时，只运行静态交付检查：

```bash
CPU_ONLY=1 bash scripts/final_preflight.sh
```

该模式通过不代表 GPU 运行时已经验证；完整训练前仍必须在 4090D 上运行默认的完整预检。
