# agentic-rl 独立训练前审查（2026-09-05）

结论：**BLOCKED（正式训练）**。CPU 可验证的接线、隔离与奖励修复已完成并通过预检；真实 4090D/TRL 的训练与恢复证据仍缺失。未下载模型、未运行 GPU 训练、未提交或推送代码。

## 审查依据与发现

审查对象为当前 `agentic-rl` 的所有未提交文件：读取 tracked diff、全部新增 RL 模块、配置、脚本与测试，并追踪 Shell → CLI → Trainer。没有采用上一轮报告作为验收证据。修改前原有 80 项 CPU 测试通过，但存在以下实际缺口：

- `grpo_kwargs()` 硬编码 μ=1，使当前步频下 Clip-Higher 没有有效约束机会。
- 数据加载没有构造隐藏测试；`run_tests` 调用最终验证路径，若指定隐藏目录则会泄露隐藏验证反馈。
- 保护规则拦截没有立即终止，危险操作的惩罚可能被 success 抵消。
- 必需 bubblewrap 且不探测能力；Python 环境未装 pytest，未完成过真实隔离测试。
- 训练启动及 manifest 哈希会读取所有 SFT split；没有语言 LoRA 的视觉参数/多 Adapter 复查。

## 已落实的训练路径

`run_agentic_rl_{smoke,train}.sh` → `scripts/agentic_rl.py` → `load_tasks(ROOT, "train")` → `training.train()`。
YAML → `RLConfig.validate()` → `grpo_kwargs()` → `GRPOConfig(**kwargs)` → `GRPOTrainer`。

| Trainer 参数 | Smoke | 正式 |
| --- | --- | --- |
| loss_type | dapo | dapo |
| epsilon / epsilon_high | 0.2 / 0.28 | 0.2 / 0.28 |
| mask_truncated_completions | true | true |
| num_iterations | 2 | 2 |
| num_generations | 2 | 4 |
| microbatch / accumulation / steps_per_generation | 1 / 2 / 2 | 1 / 4 / 4 |
| beta / use_vllm | 0 / false | 0 / false |

`rollout_func=RolloutBridge` → 多轮 `rollout()` → `Environment.run()` → 终止后的 `Environment.verify()` → `score()`。
Bridge 返回原始 `prompt_ids/completion_ids`、assistant-only `env_mask` 和独立验证奖励 `verified_reward`。
TRL 将 mask 作为 `tool_mask` 乘入 loss mask，并将 reward 额外字段传给 `reward_funcs=verified_reward`。
该契约和 old-logprob 复用条件已对照 [TRL 固定版本源码](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py)。

旧 μ=1 时 `G % (G*1) == 0`，old logprobs 使用当前值 detach，ratio=1；现 μ=2 时 `G % (G*2) != 0`，TRL 在更新前计算 old logprobs并复用同批轨迹。Clip-Higher 有机会作用于第二次更新，不保证必然发生 clipping。CPU 测试检查实际构造参数、模数条件和入口回调；没有声称运行了真实 TRL loss。

## 隐藏验证和数据边界

| Split | 任务数 | 公开原始断言 | 隐藏原始断言 |
| --- | ---: | ---: | ---: |
| train | 171 | 474 | 336 |
| validation | 7 | 15 | 8 |
| test | 11 | 29 | 20 |
| 总计 | 189 | 518 | 364 |

上述是断言数，不能当作模型通过数或 pytest testcase 数。`hidden.py` 对原始断言按顺序交替划分，同一断言不得跨公开/隐藏两侧。测试验证拆分前后 AST 断言多重集合完全一致。5 个不能安全拆分的 HumanEval 和仅有重复断言的 mbpp_37 被排除；连同非执行任务总计排除 311 条。

Actor 工作副本仅有 stub 和公开测试；不复制原始解答、README、隐藏测试、验证器或报告。隐藏测试仅在终止后进入独立私有验证副本。工具只能运行公开 pytest，返回公开聚合结果。Prompt、read/grep/retrieval/list 输出和工作副本的隐藏断言缺席均有测试。

训练仅加载 `data/rl/train.jsonl` 及所选 train 仓库；全局 manifest 只有身份、计数、来源哈希，不含 held-out 的任务内容。文件访问拦截测试证明训练 loader 不打开 SFT 数据、validation/test 清单或对应仓库。训练 manifest 不再遍历 SFT/所有 split。

固定 seed=42、按 group 的 80/10/10 候选为 **151/19/19**，但会把 **36 个现有 SFT train group** 移入新评估集。因此保留现有干净留出边界，未将候选划分启用。7/11 的样本量限制仍未解决；换新划分必须同步重做 SFT 数据与正式 Adapter。历史 SFT/预训练见过 benchmark 的可能性也不能用隐藏执行消除。

## 硬失败、奖励和隔离

隐藏验证成功是主要奖励 +2；要求 independent=true、hidden_total>0、完整性、正常结束和有效修改。辅助正奖励最多 +0.5；普通失败 success=−1，最终仍为负。保护违规固定 success=−1、integrity=−2，且立即终止；即使先正确修复并通过公开测试，也不能抵消。

测试修改/删除/断言弱化、配置或验证器写入、测试命令变更（含空字符串）、伪造测试输出字段、候选代码 print/IO/反射等均被保护终止。长度惩罚仅统计模型 Token。超时、轮数耗尽、任务失败和保护终止保留 mask=0 的环境 EOS，避免 TRL 技术截断过滤；真正 Token 预算耗尽才使用非 EOS 末尾。

bubblewrap 探针包括 namespace 与其内部 Python/pytest。当前结果为 `bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted`，实际后端是 **lightweight**：临时副本、固定 argv/无 shell、严格工具和 Python 白名单、受保护文件及验证副本哈希、资源/超时限制。它没有 OS 文件系统/网络隔离，不能称为强沙箱。虚拟环境挂载顺序已修复，但本机没有实际通过 bubblewrap pytest。

## CPU 验收证据

执行环境：Python 3.12.3，`/tmp/agentic-rl-review-venv` 中 pytest 8.4.2；安装的只有 CPU 测试依赖，没有 Torch、TRL、模型权重。

命令：`PYTHON_BIN=/tmp/agentic-rl-review-venv/bin/python bash scripts/final_rl_preflight.sh`。

- **92 项测试全部通过**（原 80 项 + 12 项独立加固测试；旧 mock fixture 增加第二条断言以支持真实公开/隐藏分区）。
- **合法 Canary 通过**：train 的 mbpp_27，脚本动作修复源码，真实公开与隐藏 pytest 均通过，成功奖励为正。
- **攻击 Canary 通过**：9 种攻击子用例，先正确修复、公开测试通过，再攻击；全部立即保护终止、总奖励为负。
- 只拟合公开测试的候选公开通过、隐藏失败、负奖励；未独立验证不可能获得 success。
- **真实隔离 pytest 正/负控制通过**，实际为轻量后端。
- Trainer 入口测试使用构造器替身、真实 Bridge/Environment/pytest，核查参数、reward、mask 接线；它不是实际 TRL 优化测试。
- 原 SFT 严格数据检查、Python 编译、全部 Shell 语法、`git diff --check` 及所有新增文件 whitespace 检查通过。

## 4090D 硬门槛和剩余限制

正式 SFT Adapter 必须证明 run_kind=formal、训练已完成、冻结视觉塔/projector、训练数据哈希与本次留出边界一致。CPU 可读 safetensors 头拒绝非语言 LoRA；加载只调用一次 `PeftModel.from_pretrained(..., is_trainable=True)`，不 merge/叠加新 LoRA；Trainer 构造后再次检查唯一 default Adapter 和可训练参数。

实际本地 Adapter/模型未提供，以下仍须在单张 4090D 完成：

1. 固定依赖与 Qwen3.5 纯文本 forward/generate、非思考 Token 边界兼容。
2. 真实 TRL 的重复采样次序、两次更新 old logprobs 复用与 ratio、观察 Token 的零直接 loss/梯度，失败样本不被技术截断过滤。
3. 有限 loss/LoRA 梯度、至少一组非零梯度、参数真实更新、视觉参数无梯度。
4. checkpoint 恢复的 optimizer/scheduler/RNG 与下一步行为一致。
5. 峰值显存、每轮耗时、实际 reward 方差；没有任何已测训练/评测指标。

正式默认 G=4、Smoke G=2、微批次 1、BF16、gradient checkpointing、β=0、禁用 vLLM。冻结视觉塔仍占内存，8192 上下文及大词表 logits 存在 24GB OOM 风险。Runbook 给出缩短上下文/Token/output 的回退配置；保留 μ=2，并以新 run 重跑 Smoke。无 server/colocate 第二模型依赖。

正式启动现在要求 `--smoke-run`，检查已保存的非零梯度、参数变化、有限 loss、足够步数、轨迹及 DAPO 参数证据，并与正式运行的代码/数据/依赖/模型/Adapter 对齐。该 gate 不能替代第 2、4 项的 GPU 梯度与恢复实验。全部 GPU 门槛未通过前，结论保持 **BLOCKED**。
