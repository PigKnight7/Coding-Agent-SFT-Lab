# Agentic RL Runbook

本项目实现 **DAPO 增强型 GRPO（DAPO-style Agentic RL）**，不是完整 DAPO 复现。
SFT 流程保持独立，见 [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md)。本文件是 SFT 后继续 RL 的唯一推荐流程。

## 架构与范围

`CodingAgentGraph` 与 RL 共用 `cc_agent.actions` 中的 Actor 提示词和动作执行入口；
RL 环境调用原有 `RepoTools`、路径 Hook 和 JSONL Trace，不另建模型裁判或复制工具实现。
原有规划、review、总结节点仍服务交互式 Agent；训练针对其 Actor 工具决策策略，
通过可注入 Policy 和 Environment 驱动同一个动作协议，规划理由可放在 action 的 reason 中。
RAG 核心不变；RL 工具进程固定使用现有 lexical 模式，无联网 embedding、共享索引或凭据。

模块：`protocol.py` 定义 Task/Action/Observation/Trajectory/Termination/Verification/Reward；
`environment.py` 管理临时副本、工具保护和验证；`rollout.py` 记录多轮 Token 轨迹；
`rewards.py` 计算可验证奖励；`data.py` 加载冻结 split 清单，`hidden.py` 划分原始断言；`training.py` 接 TRL；
`artifacts.py` 管理 manifest/恢复；`evaluation.py` 汇总统一指标。

每条 rollout：创建独立副本 → 仅副本内重置函数体 → 模型 JSON 动作 → 原工具/Hook →
环境观察 → 后续模型动作 → finish/失败 → 环境重新验证 → 奖励 → 销毁副本。
同一任务的 G 条轨迹也不共享仓库。所有模型生成 ID 原样保留，不重新 tokenize 模型输出。
初始 system/user 是 prompt；后续工具观察用明确标注的 user 消息承载以兼容现有 JSON Actor，
其聊天边界和下一轮 assistant 提示头均为环境 Token。只有生成的 assistant/action Token 的 `env_mask=1`。

## 安全和可验证性

- 只允许固定 schema 工具；仅 `solution.py` 可写。测试、断言、配置、新文件、Git 元数据和仓库外路径不可写。
- 来源仓库拒绝符号链接；复制时排除 Git、缓存和 `.env`。每次验证前后对非目标文件集合和 SHA256 做完整比较。
- `run_tests` 只接受任务固定的 `pytest -q`。真正执行的 argv 由环境固定，并禁用 pytest 自动插件和缓存；不执行 shell。
- 启动时探测 bubblewrap namespace 及其内部 Python/pytest 的实际可用性，优先使用独立 PID/network/user namespace、
  只读仓库、清空环境、无 GPU/原始仓库挂载。虚拟环境挂载放在 `/tmp` tmpfs 之后，避免遮蔽位于 `/tmp` 的 Python 环境。
- bubblewrap 缺失或探针失败时使用 **轻量隔离**：独立临时仓库副本、固定 argv、无 shell、严格工具/候选 Python 白名单、
  受保护文件哈希和验证副本执行前后全文件哈希。此回退没有 OS 文件系统/网络隔离，不能当作任意不可信 Python 的安全沙箱。
  报告必须记录实际 backend 与回退原因；pytest 未安装仍是预检失败，不允许 mock 替代。
- 两种后端均禁用 pytest 插件自动加载、用户 site 和缓存，固定 `/dev/null` 配置与测试收集根。
- 另有 CPU、内存、文件大小、FD 限制和外层墙钟超时。超时清理整个验证进程组。
  常规工具在有资源限制的独立 spawn 进程执行，输出最多 4000 字符，超时 30 秒。
- 为阻止候选代码篡改 pytest 进程/报告，第一版只支持保守 Python 函数子集：拒绝动态执行、IO、私有属性、反射、类、
  不受支持的 import/属性以及模块属性写入。此策略是隔离层之外的额外限制，不声称支持任意 Python 或任意 SWE 仓库。
- 不根据输出文本判断通过；环境独立运行测试并解析 JUnit testcase。没有报告属于基础设施错误，会终止运行；不会静默变成任务奖励。
- `hidden.py` 从现有 MBPP/HumanEval 原始断言中交替划分公开/隐藏部分；不添加、改写或推导标准答案。
  相同断言必须留在同侧；只有一个独特断言、循环/共享 setup 等无法安全划分的任务排除。
  Actor 副本仅包含函数接口 stub 和公开测试，原仓库 README、答案和隐藏断言不会复制进去。
- `run_tests` 仅执行公开测试；轨迹结束后 `Environment.verify()` 才在另一私有副本加入隐藏断言并独立执行 pytest。
  隐藏源码、用例和失败输出不进入 prompt、工具返回、observation；最终日志只有独立验证标记和聚合计数。
- 修改/删除测试、弱化断言、变更命令、修改验证器、提供伪造测试结果或执行危险 Python 均为不可恢复的保护终止。
  一旦拦截，不再执行后续动作，success 固定为 −1，integrity 为 −2；辅助奖励无法抵消。

## 奖励公式

`R = success + legality + arguments + format + edit + invalid + dangerous + repeated + empty_edit + untested_finish + timeout + integrity + length`

| 分量 | 默认值/计算 |
| --- | --- |
| success | 无硬失败、正常 finish、有效修改、独立验证标记且隐藏断言数大于 0、公开及隐藏测试全通过：+2；否则 −1 |
| legality | +0.1 × 合法工具数 / 总决策数 |
| arguments | +0.1 × 执行成功工具数 / 总决策数（执行结果代理指标） |
| format | +0.1 × 可解析且进入工具边界的决策数 / 总决策数 |
| edit | 最终目标文件相对初始 stub 有变化 +0.2，否则 −0.2 |
| invalid | −0.2 × min(5, 非法工具数 + JSON 格式错误数) |
| dangerous | −0.5 × min(2, 被保护 Hook 阻止的次数) |
| repeated | −0.1 × min(5, 重复相同工具及参数次数) |
| empty_edit | −0.1 × min(5, 没产生文本变化的编辑次数) |
| untested_finish | finish 前未对当前版本主动测试：−0.5；改动后须重新测试 |
| timeout | 工具或最终验证超时：−0.5 |
| integrity | 保护违规（包括被拦截的尝试）或文件哈希变化：−2 |
| length | `−min(1, max(0, model_tokens−1536)/512)` |

辅助正奖励合计至多 +0.5，因此失败轨迹总奖励仍为负。长度 cap 可配置，校验其不超过成功奖励 2；
只统计模型生成 Token，不统计提示词、工具返回和环境 EOS。

## TRL / DAPO 边界

固定 `trl==1.12.0`，使用公开 `GRPOConfig` 和实验性 `rollout_func` 返回契约。
已核对 [官方 GRPO 文档](https://huggingface.co/docs/trl/v1.12.0/en/grpo_trainer) 和
[固定版本源码](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py)。
环境预检还检查本地版本、配置字段及 `env_mask` 消费能力；接口变化即报错。
模型规格按 [Qwen 官方 2B 配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) 核对。

公开配置：`loss_type=dapo`（按有效模型 Token 聚合）、`epsilon=0.2`、`epsilon_high=0.28`（Clip-Higher）、
`mask_truncated_completions=true`、`beta=0`、`num_iterations=2`。Smoke G=2，正式 G=4。
无 Trainer 子类、内部方法覆盖或自写 GRPO loss。`RepeatSampler` 已重复行，bridge 对每输入行生成一条轨迹，
检查 G 个相邻输入身份一致，防止重复扩增。第一版只支持单进程单 GPU。

采样 temperature=1、top_p=1、top_k=0；回调在线使用当前模型，`logprobs=None`。
实际传参路径为 YAML → `RLConfig` → `grpo_kwargs` → `GRPOConfig` → `GRPOTrainer`。
Trainer 的 `rollout_func=RolloutBridge` 调用 `rollout` → Environment → terminal verify → score，返回
`prompt_ids/completion_ids/env_mask/verified_reward`；TRL 将 `env_mask` 转为 `tool_mask` 并乘入 loss mask，
将额外 reward 字段传入 `verified_reward` 回调。CPU 入口测试使用 Trainer 替身与真实环境/pytest核查此接线；不是实际 TRL 优化验证。

原实现的 μ=1 配合 `gradient_accumulation_steps=steps_per_generation=G` 会令 old logprobs 使用当前 logprobs 的 detach，
每次更新时 ratio 为 1，Clip-Higher 无有效约束机会。现改为 μ=2，`G % (G*2) != 0`，
TRL 在更新前计算 old logprobs，并让同批轨迹跨两次 optimizer update 复用。上界 1.28 因此有机会生效；
不保证实际样本必然触及上界。old logprobs 缓存只是每 Token 的标量，主要增加一次优化计算而非第二个模型。非思考 chat template；正式 SFT LoRA 以 `is_trainable=True` 加载，
不 merge、不新建另一套 LoRA；检查 safetensors 参数名、唯一活动 default Adapter，并在 Trainer 构造后复查仅语言 LoRA 可训练。BF16、gradient checkpointing，无 vLLM、无量化。

TRL 通过最后一个 Token 是否 EOS 判断技术截断。预算耗尽保留非 EOS 末尾；
轮数耗尽、任务失败、超时和保护终止增加 **mask=0 的环境 EOS**，避免被误过滤。
每轮 action Token 上限、累计模型 Token 上限和上下文 Token 上限属于技术预算。
所有轨迹仍保留日志和奖励；只有技术截断的策略损失由 TRL mask。
没有完整 Dynamic Sampling，仅在 JSONL 中报告组内奖励方差为零的比例。不会自动重采样失败组。

## 数据隔离

不改变 `data/sft`、`data/llamafactory` 和原始 `data/repos`。
`python3 scripts/prepare_rl_data.py` 是显式离线数据审计/冻结步骤，不由训练入口调用。
它按 `benchmark:source:id` 对齐 SFT group，生成 `data/rl/{train,validation,test}.jsonl` 及仅含身份/哈希的 manifest。
训练只读取 train 清单和 train 仓库；不会打开 validation/test 清单、仓库或任何 SFT 数据文件。
训练 manifest 也只哈希选定 split，原始测试修改会使冻结清单校验失败。直接向 `train()` 传 held-out Task 会报错。

| Split | 任务 | 公开断言 | 隐藏断言 |
| --- | ---: | ---: | ---: |
| train | 171 | 474 | 336 |
| validation | 7 | 15 | 8 |
| test | 11 | 29 | 20 |
| 合计 | 189 | 518 | 364 |

原 195 个可执行任务额外排除 6 个：5 个不能安全拆分的 HumanEval 和只有重复断言的 mbpp_37；含非执行任务共排除 311 条。
这些是静态断言数，不是 pytest testcase 数，更不是模型成功率。一个 pytest 函数通常包含多条断言。

固定 seed=42、按 group 的约 80/10/10 方案为 151/19/19，已生成审计与确定性测试。
该全量重排会把 36 个现有 SFT train group 移入 held-out，不能提高当前 Adapter 的评估可信度。
因此活动划分保留已有 SFT 留出边界；7/11 的小样本限制仍未解决。
采用新划分必须同步重做 SFT 分组及正式 Adapter，不能单改 RL split 后声称干净评估。
正式 Adapter 的 SFT train 数据哈希必须与冻结的来源记录一致，避免误用其他划分或 Smoke。
即使隐藏断言对 RL Actor 不可见，模型可能在既有 SFT/预训练中见过 benchmark；不声称消除了历史训练污染。
模型选择仅用 validation，最终 test 在冻结后运行，结果不得回流调参。

## 唯一推荐云端顺序

前提：已有完整本地 `Qwen/Qwen3.5-2B` 及按 SFT Runbook 完成的正式 adapter 目录，
其中包含 `experiment_manifest.json`（run_kind=formal）、`trainer_state.json`（epoch≥1）和 adapter safetensors。
使用单张 4090D、支持 BF16 的 CUDA PyTorch。优先安装并允许 bubblewrap；当前 Python 环境必须安装 pytest。
RL 使用独立 Python 环境，安装本项目及 `requirements-agentic-rl.txt`，避免改变可独立运行的 LLaMA-Factory SFT 环境。
CUDA PyTorch/torchvision 按云主机已有 CUDA 环境安装；本项目不自动安装驱动、不下载模型。

在仓库根目录设置已有路径（不要把示例占位符当成真实目录）：

```bash
export MODEL_PATH=/path/to/local/Qwen3.5-2B
export SFT_ADAPTER=/path/to/formal-sft-output
export PYTHON_BIN=python3
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 在已准备好 CUDA PyTorch 的独立 RL Python 环境中
python3 -m pip install -e . -r requirements-agentic-rl.txt
bash scripts/final_rl_preflight.sh
python3 scripts/check_rl_sandbox.py

# 1. 正式 SFT adapter 完整性、真实生成检查（只用训练 Smoke 样本）
python3 scripts/verify_sft_adapter.py --model-path "$MODEL_PATH" \
  --adapter-dir "$SFT_ADAPTER" --expected-steps 1

# 2. RL 环境真实预检：版本、正式 SFT 来源、模型本地文件、CUDA/BF16、隔离正/负控制
bash scripts/check_rl_environment.sh --model-path "$MODEL_PATH" --sft-adapter "$SFT_ADAPTER"

# 3. 两步真实 GRPO Smoke；目录必须不存在或为空
bash scripts/run_agentic_rl_smoke.sh --model-path "$MODEL_PATH" \
  --sft-adapter "$SFT_ADAPTER" --output-dir outputs/agentic_rl_smoke
python3 scripts/verify_rl_run.py --run-dir outputs/agentic_rl_smoke --expected-steps 2

# 4. 正式 RL 从正式 SFT 开始，不能从 Smoke adapter 开始
bash scripts/run_agentic_rl_train.sh --model-path "$MODEL_PATH" \
  --sft-adapter "$SFT_ADAPTER" --smoke-run outputs/agentic_rl_smoke \
  --output-dir outputs/agentic_rl_formal
python3 scripts/verify_rl_run.py --run-dir outputs/agentic_rl_formal --expected-steps 200

# 5. validation：SFT 与 SFT+DAPO-GRPO，同一任务、预算、seed、贪心解码
bash scripts/run_agentic_rl_eval.sh --model-path "$MODEL_PATH" --sft-adapter "$SFT_ADAPTER" \
  --rl-adapter outputs/agentic_rl_formal/final_adapter --split validation \
  --output-dir eval_results/agentic_rl_validation

# 6. 配置及 adapter 冻结后，最终 test；禁止根据这里的结果继续调参
bash scripts/run_agentic_rl_eval.sh --model-path "$MODEL_PATH" --sft-adapter "$SFT_ADAPTER" \
  --rl-adapter outputs/agentic_rl_formal/final_adapter --split test \
  --output-dir eval_results/agentic_rl_final_test
```

第 3 步的证据 gate 要求真实有限 loss、足够训练步、保存 adapter、至少一条合法工具多轮轨迹、
非零有限 LoRA 梯度、训练参数哈希实际变化和 DAPO 配置证据。`model_audit.json` 记录这些运行时事实。
正式启动强制提供 `--smoke-run`，校验 Smoke 与正式的代码、数据、模型、SFT Adapter、依赖及公共配置一致。
恢复只允许本 run 的完整 checkpoint；恢复过程的优化器、调度器、RNG 和下一步梯度仍须 GPU 验证。
如果不通过，先查训练任务上的 Trace；不得声称 Smoke 已成功或把 Mock 结果替代它。
正式默认 200 步是起始实验预算，不是已验证的最优超参数或显存保证。

## 断点续训、日志与评测输出

中断后，从已实际存在的完整 checkpoint 恢复，例如：

```bash
bash scripts/resume_agentic_rl.sh outputs/agentic_rl_formal/checkpoint-25 \
  --model-path "$MODEL_PATH" --sft-adapter "$SFT_ADAPTER" \
  --output-dir outputs/agentic_rl_formal --config configs/agentic_rl_train.yaml
```

Smoke 恢复须显式使用 `--config configs/agentic_rl_smoke.yaml`。
要求同一 output 的 checkpoint，optimizer/scheduler/RNG/Trainer state/adapter 全部存在，
且代码、配置、任务内容、冻结 RL split、模型权重/Tokenizer、adapter 和依赖 manifest 一致。不提供只恢复权重的伪续训。
标准 checkpoint 保存边界恢复，未完成 rollout 会重做；原有 trace 是追加式，可能含重做记录。
manifest 记录 Git commit 及实际源码 SHA256（包括未提交文件），因此不要求提交本轮开发。

每个 run 有 `rl_manifest.json`、`rollouts.jsonl`、`training_log.jsonl`、Trainer checkpoints、`final_adapter/`。
统一评测逐模型顺序加载，输出各自 Trace 和 `metrics.json`：任务成功率、pytest testcase 通过率、工具合法率、
平均轮数、平均模型 Token、技术截断率、所有奖励分量、总奖励、失败类型。无模型时不产生指标文件。
Mock 只存在于 CPU 测试，不进入正式 CLI。

## CPU 验收与待云端确认

`bash scripts/final_rl_preflight.sh` 先真实运行隔离 pytest 正/负控制，再执行原 SFT 只读严格数据检查、全部 CPU 测试、
Python 编译、Shell 语法及所有 tracked/untracked diff 空白检查。pytest 缺失时必须先在自己的 Python 环境安装依赖。
本次使用 `/tmp` 虚拟环境中的 pytest 8.4.2；没有安装 Torch/TRL 或下载模型。
当前宿主工具环境的 bubblewrap 探针报告 `NETLINK_ROUTE ... Operation not permitted`，
因此真实正/负控制和 Canary 使用轻量隔离后端通过。不能把此结果写成 bubblewrap 强隔离通过。

两条 Canary 位于 `tests/test_rl_hardening.py`，用 train 中的 mbpp_27 原始任务和脚本化动作，真实执行 pytest：
合法修复公开/隐藏均通过；先合法通过公开测试、再攻击测试/命令/报告/候选 IO 的轨迹全部保护终止且负奖励。
另有只拟合公开测试的候选，公开通过而隐藏失败，总奖励为负。CPU 数值仅用于 Canary 断言，不输出模型评测指标。

单卡默认禁用 vLLM（包括 server 和 colocate），没有第二张 GPU 服务依赖。微批次 1，G=2/4，BF16，
gradient checkpointing，β=0 无 reference 模型，复用正式 SFT LoRA。2B 权重、视觉塔虽冻结仍占显存，
8192 上下文的词表 logits/反向激活可能使 24GB OOM；当前没有显存测量或吞吐保证。
OOM 回退：新建 Smoke/正式配置，将 max_context_tokens=4096、max_model_tokens=1024、max_action_tokens=256、
safe_length=768、output_limit=1500；先重跑 G=2 Smoke，再尝试正式 G=4。仍不足时正式降为 G=2并在报告标记。
保留 num_iterations=2、Clip-Higher 和隐藏验证；配置变化必须新 run、新 Smoke，不能对旧 checkpoint 改参恢复。
不提供未经实现/验证的 colocate 显存参数。

**4090D 硬门槛（当前 BLOCKED）**：固定依赖与 Qwen3.5 纯文本 forward/generate 兼容性；真实 TRL 回调次数与
RepeatSampler 顺序；多轮 assistant mask 对 observation Token 的零直接 loss（需检查梯度），技术截断与失败负样本区分；
两次优化复用 old logprobs、比率偏离 1、有限 loss/梯度及 LoRA 参数实际更新；视觉参数无梯度；
checkpoint 恢复后 optimizer/scheduler/RNG 一致；实际峰值显存、每轮耗时和任务 reward 方差。
`verify_rl_run.py` 能检查已保存的部分证据，不能代替观察 Token 梯度和恢复一致性实验。
通过这些门槛之前不得将 CPU Canary、源码检查或 Trainer 替身报告为真实 GPU Smoke/训练/评测成功。
