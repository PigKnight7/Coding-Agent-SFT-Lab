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
  一旦拦截，不再执行后续动作，不发放正确性奖励，integrity 为 −3；辅助奖励无法抵消。

## 第一轮结论与 reward v2

按第一轮提供的 validation 结果，200 步只改善工具合法性（0.6607 → 1），
任务成功率和测试通过率均为 0；平均奖励 −1.4982 → −0.6400、轮数 8 → 5，截断率均为 0。
7 条 RL 轨迹均 changed/tested_current_edit=true、0/2、finish，几乎固定为
`replace_in_file → read_file → replace_in_file → run_tests → finish`。
这是 reward shortcut 和策略模式坍缩，不能把奖励提升报告为代码能力提升。
以上为第一轮用户提供的结果，本次没有重新运行或独立复核模型评测。

旧 success 对全部非成功情况统一给 −1，没有部分通过差异；过程正奖励最多 +0.5。
测试过当前修改就能免除 untested_finish，即使测试全失败；max_turns 没有专门惩罚。
格式、非法调用、重复调用等负项减少就能让失败奖励上升。
例如五次调用全部合法、三次 ok、有 edit、一次重复且长度安全时，
旧公式为 −1+0.1+0.06+0.1+0.2−0.1=−0.64。
该例解释数值如何产生，不声称已拿到第一轮逐条分量；精确归因需对应原始 Trace。
完整成功与失败旧上界仍有间隔，但 0/2 和 1/2 没有正确性差异，辅助项可完全决定二者排序。

## 奖励公式与可证明边界

权重为 `rewards.py` 中具名常量，由 CPU 排序回归测试约束，不允许 YAML 任意改大辅助项。
仅终局 independent=true、hidden_total>0、完整性可信且无超时的 passed/total 可用于进度。
公开工具测试仍只反馈公开测试；隐藏断言、失败输出和源码不进入 Actor。

令 f=passed/total，完整成功要求 f=1、有效修改、finish 且无安全失败：

- success：完整成功 +2，其余 −3。
- test_progress：可信 f>0 且非完整成功时 +1.5+f，否则 0。1.5 是跨越终止惩罚带的进度门槛。
- failed_finish：非完整成功却 finish，−0.5；不依赖 tested_current_edit。
- untested_finish：finish 前没有测试当前修改，另扣 −0.5（保留原检查）。
- max_turns：−1.25，比最差失败 finish 的合计 −1 更重，差额大于辅助项的最大摆幅。
- integrity：保护终止、blocked 或完整性破坏，−3；dangerous 每次 −0.5，最多 −1。
- timeout：保留 −0.5，并增加 timeout_severity=−3。安全失败不发放 test_progress 或成功奖励。
- 辅助项包含 legality、arguments、format、edit、invalid、repeated、empty_edit、length。
  先按旧公式计算原始向量 q（length 仍为模型 Token 的柔性惩罚），再令
  `a=0.05/(total+1)`，`aux_i=a*q_i/max(1, sum(abs(q_i)))`。
  因此所有辅助分量绝对值之和 ≤a≤0.05。length_cap 配置仅控制归一化前的长度项。

`R = success + test_progress + failed_finish + untested_finish + max_turns
     + integrity + dangerous + timeout + timeout_severity + sum(aux)`。

下表为 total=2 的保守上下界（a=1/60；包含最差格式/长度等辅助信号）：

| 轨迹 | 奖励区间 |
| --- | --- |
| 2/2 完整成功（含未主动测试的成功） | [1.4833, 2.0167] |
| 1/2 通过，任意普通终止 | [−2.2667, −0.9833] |
| 1/2 通过，已测试后 finish | [−1.5167, −1.4833] |
| 0/2，普通 task_failed/token_limit | [−3.0167, −2.9833] |
| 0/2，已测试后 finish | [−3.5167, −3.4833] |
| 0/2，未测试后 finish | [−4.0167, −3.9833] |
| 0/2，max_turns | [−4.2667, −4.2333] |
| timeout / dangerous / integrity（可组合） | [−11.8, −5.95]（跨 total 的宽界） |

对任意 total，非安全失败的部分通过奖励 >−2.8，全失败普通终止 ≤−2.95；
完整成功 ≥1.45，未完整成功但所有测试通过的轨迹也 ≤−0.45。
同一任务、相同终止/主动测试状态下，相邻通过数带来至少 1/total 的差异，
辅助项最大摆幅为 0.1/(total+1)，无法逆转排序。
终止和主动测试惩罚是显式行为约束，不属于辅助项；不宣称相邻通过数在所有终止方式之间也严格排序。
max_turns 与失败 finish 在同等正确性下仍有严格间隔，不能通过不结束逃避惩罚。
这些常数满足显式不等式：max_turns 惩罚幅度 1.25 > failed_finish+untested_finish 的 1.0 加辅助摆幅 0.1；
进度门槛 1.5 > 最大普通终止惩罚 1.25 加辅助摆幅 0.1。
因此先选择有间隔的奖励带，再用测试锁定上下界，并非只把几个辅助权重凭直觉调小。
技术截断的负奖励仍记账，但依照已有 TRL 配置会过滤其策略 loss。

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
reward v2 修正为 **基于 correctness/safety outcome 的同任务有界 Dynamic Sampling**。
此前只检查 total_reward 方差，即使辅助预算很小，两条均 0/N 的轨迹也可能因过程奖励不同而被接受。
GRPO 的组内标准化会放大这些微小差异；缩小绝对权重不能消除“规范地失败”的 reward shortcut。

每条轨迹的 primary outcome 包含 full_success、可信 test_pass_fraction、protected_integrity_failure、timeout。
可信比例沿用独立终局验证、隐藏测试、完整性和超时检查；不可信计数为 0。
四项全部相同才是 correctness-zero-variance；辅助项、普通终止原因不能使其成为有效组。
存在通过比例、完整成功或安全状态差异就接受，包括普通失败与 integrity failure 的组。
每个相邻 G 行 group 整组生成，默认 enabled=true、max_retries=3，每组最多 4G 条。
耗尽保留最后组，但 optimization_reward 全设为精确 0.0，确保 TRL float32 归一化优势为零。
禁用采样或 max_retries=0 也执行相同的优化奖励保护；disabled 不计 exhausted。

有 primary variance 时，若 diagnostic total 已严格保持 primary 排序则直接使用；否则使用组内
primary 等级序号 + 0.05*tanh(diagnostic_total_reward)，相邻等级间隔至少 0.9。
排序依次优先避免 integrity failure、避免 timeout、完整成功、通过比例，防止跨终止类别的惩罚
或辅助项逆转正确性排序。仅安全差异仍形成梯度；同 outcome 的纯辅助差异不能单独开启训练。
`rl_trajectory` 保留 total_reward 兼容字段、diagnostic_total_reward 和原始 reward.components；
`rl_group_attempt` 按 rollout_ids 顺序记录 primary_outcomes、diagnostic_total_reward、reward_components、
optimization_reward（丢弃组为 null）、selection、correctness_zero_variance、total_reward_zero_variance、
optimization_reward_zeroed、exhausted。最终 optimization_reward 经原 verified_reward 字段交给 TRL。

max_retries 严格校验为整数 [0,16]；zero_variance_epsilon 为有限数 [0,1e−4]，现仅用于总奖励
float32 方差诊断，不参与 primary outcome 判定；enabled 必须是真布尔值。
Smoke/formal 这些参数完全一致，manifest gate 保持仅允许原有规模差异。

实现仅使用公共 rollout_func，最终仍按输入顺序每行返回一条 prompt/completion/mask/reward，
不更换任务、不跨进程拼组、不补新的 dataset 行、不改 Trainer 或 loss。
依据固定版源码的 `_generate`、`_generate_and_score_completions` 和 RepeatSampler，
额外 reward 字段绑定原 inputs，后续按 G reshape 计算优势。因此保留原任务和基数；
这不是论文中筛选有效 prompt 并不断补充新 prompt 直到填满批次的完整 Dynamic Sampling。
丢弃样本在 old logprobs 快照和优化前被移除；返回的原始 token/mask 不变，logprobs=None 和 μ=2 不变。
近零判定是数值容差，不声称 epsilon 内的差异在数学上严格零优势。
同正确性但辅助项不同的组仍可能有优势；组内归一化会放大小信号。
有界重采样也无法保证发现正确解，必须观察成功率、进度、耗尽比例，不能用总奖励宣布修复有效。

Trace 用唯一 batch_id/rollout_id 关联 rl_tool_call、rl_trajectory（candidate）与
rl_group_attempt（accepted/discarded/exhausted）。只有带 rl_batch_selected 的完成批次中的 accepted 轨迹
才是返回给 TRL 的样本；中断留下的 candidate 不得算作训练证据。
这表示进入 Trainer 的候选，不保证未被技术截断 mask，也不等于 optimizer 已完成更新。
Smoke gate 仅使用这些采用轨迹；丢弃、未完成回调的轨迹不能充当证据。

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
  --sft-adapter "$SFT_ADAPTER" --output-dir outputs/agentic_rl_smoke_reward_v2
python3 scripts/verify_rl_run.py --run-dir outputs/agentic_rl_smoke_reward_v2 --expected-steps 2

# 4. 正式 RL 从正式 SFT 开始，不能从 Smoke adapter 开始
bash scripts/run_agentic_rl_train.sh --model-path "$MODEL_PATH" \
  --sft-adapter "$SFT_ADAPTER" --smoke-run outputs/agentic_rl_smoke_reward_v2 \
  --output-dir outputs/agentic_rl_formal_reward_v2
python3 scripts/verify_rl_run.py --run-dir outputs/agentic_rl_formal_reward_v2 --expected-steps 200

# 5. validation：SFT 与 SFT+DAPO-GRPO，同一任务、预算、seed、贪心解码
bash scripts/run_agentic_rl_eval.sh --model-path "$MODEL_PATH" --sft-adapter "$SFT_ADAPTER" \
  --rl-adapter outputs/agentic_rl_formal_reward_v2/final_adapter --split validation \
  --output-dir eval_results/agentic_rl_validation_reward_v2

# 6. 配置及 adapter 冻结后，最终 test；禁止根据这里的结果继续调参
bash scripts/run_agentic_rl_eval.sh --model-path "$MODEL_PATH" --sft-adapter "$SFT_ADAPTER" \
  --rl-adapter outputs/agentic_rl_formal_reward_v2/final_adapter --split test \
  --output-dir eval_results/agentic_rl_final_test_reward_v2
```

第二轮必须使用上述新目录，保留第一轮全部产物。代码/配置变化使旧 Smoke 证据失效，必须重跑 GPU Smoke；
仍从第一轮相同正式 SFT Adapter 启动正式 RL，不能从第一轮 RL 或新 Smoke 权重启动。
validation 用于选择方案，test 仅在方案冻结后运行一次，不根据 test 调参。

第 3 步的证据 gate 要求真实有限 loss、足够训练步、保存 adapter、至少一条合法工具多轮轨迹、
非零有限 LoRA 梯度、训练参数哈希实际变化和 DAPO 配置证据。`model_audit.json` 记录这些运行时事实。
正式启动强制提供 `--smoke-run`，校验 Smoke 与正式的代码、数据、模型、SFT Adapter、依赖及公共配置一致。
恢复只允许本 run 的完整 checkpoint；恢复过程的优化器、调度器、RNG 和下一步梯度仍须 GPU 验证。
如果不通过，先查训练任务上的 Trace；不得声称 Smoke 已成功或把 Mock 结果替代它。
正式默认 200 步是起始实验预算，不是已验证的最优超参数或显存保证。

## 断点续训、日志与评测输出

中断后，从已实际存在的完整 checkpoint 恢复，例如：

```bash
bash scripts/resume_agentic_rl.sh outputs/agentic_rl_formal_reward_v2/checkpoint-25 \
  --model-path "$MODEL_PATH" --sft-adapter "$SFT_ADAPTER" \
  --output-dir outputs/agentic_rl_formal_reward_v2 --config configs/agentic_rl_train.yaml
```

Smoke 恢复须显式使用 `--config configs/agentic_rl_smoke.yaml`。
要求同一 output 的 checkpoint，optimizer/scheduler/RNG/Trainer state/adapter 全部存在，
且代码、配置、任务内容、冻结 RL split、模型权重/Tokenizer、adapter 和依赖 manifest 一致。不提供只恢复权重的伪续训。
标准 checkpoint 保存边界恢复，未完成 rollout 会重做；原有 trace 是追加式，可能含重做记录。
manifest 记录 Git commit 及实际源码 SHA256（包括未提交文件），因此不要求提交本轮开发。

每个 run 有 `rl_manifest.json`、`rollouts.jsonl`、`training_log.jsonl`、Trainer checkpoints、`final_adapter/`。
统一评测逐模型顺序加载，输出各自 Trace 和 `metrics.json`：任务成功率、pytest testcase 通过率、工具合法率、
平均轮数、平均模型 Token、技术截断率、所有奖励分量、总奖励、失败类型。
新增 full_success_rate、partial_test_pass_rate、mean_test_pass_fraction、all_tests_failed_rate、
failed_finish_rate、max_turns_rate、timeout_rate、protected_integrity_failure_rate、empty_edit_rate。
empty_edit_rate 表示最终没有有效修改的轨迹比例，empty_edit_attempt_rate 表示出现空编辑尝试的轨迹比例。
outcome_counts 为互斥的正确性/安全类别（含 unverified、all_tests_passed_incomplete），
termination_counts 单独统计终止原因，outcome_termination_counts 提供二者交叉计数；
因此部分通过后 finish 和全部失败后 finish 可分别审计。
failure_types 将 failed_finish、max_turns、timeout、protected_integrity_failure 明确分开。
mean_test_pass_fraction 是可信逐任务比例的宏平均；原 test_pass_rate 保留原始总 passed/总 total 的微平均。

训练在 training_log.jsonl 的 rl_sampling_metrics 事件中逐已采用批次记录以上指标，以及
correctness_zero_variance_group_rate（首轮 primary 全同组/输入组）、
accepted_correctness_zero_variance_group_rate（最终 primary 全同组/输入组）、
total_reward_zero_variance_group_rate（首轮总奖励 float32 近等值组/输入组）、
optimization_reward_zeroed_group_rate（最终优化奖励主动置零组/输入组）、
dynamic_sampling_retry_count（本批额外整组尝试数）、dynamic_sampling_exhausted_rate（耗尽组/输入组）
和 dynamic_sampling_exhausted_count。这里 correctness-zero-variance 包含安全状态一致的要求。
每行包含 step、batch_id、tasks 和 group_count；跨批轨迹率按 tasks 加权，组率按 group_count 加权，retry_count 求和。
这些日志按生成批次计一次，不按 μ=2 重复计数；原 trainer_log 和模型更新审计继续保留。
无模型时不产生评测指标文件。
Mock 只存在于 CPU 测试，不进入正式 CLI。

## CPU 验收与待云端确认

`bash scripts/final_rl_preflight.sh` 先真实运行隔离 pytest 正/负控制，再执行原 SFT 只读严格数据检查、全部 CPU 测试、
Python 编译、Shell 语法及所有 tracked/untracked diff 空白检查。pytest 缺失时必须先在自己的 Python 环境安装依赖。
本轮使用 `/tmp/agentic-rl-v2-venv`，Python 3.12.3、pytest 8.4.2、CPU-only torch 2.6.0+cpu、
transformers 5.5.0（原有 Tensor/BatchEncoding 回归测试需要真实依赖），未安装 TRL、未下载模型。
本次 correctness/safety 采样修复后，全部 **121 项 CPU 测试**通过，`final_rl_preflight.sh` 完整通过。
日志为 `/tmp/rl-primary-preflight.log`；修复前 114 项历史及本次验收命令见 Review。
用户提供的修复前 GPU Smoke 已通过基本训练验证；本次未运行 GPU，修复后的 GPU 行为仍待验证。
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
