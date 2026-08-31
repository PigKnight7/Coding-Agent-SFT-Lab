# 简历项目写法参考

## 项目名称

**面向代码仓库任务的 Coding Agent 与 Qwen3-8B 后训练实验**

## 一句话介绍

构建一个可运行的轻量级 Code Agent，并将其工具调用轨迹与代码任务数据整理为 SFT 数据，基于 Qwen3-8B 做 LoRA 微调，验证后训练对 Agent 输出协议和工具调用格式的对齐效果。

## 简历 Bullet 版本

- 参考 Claude Code / Aider / OpenHands 的公开架构，设计并实现轻量级 Coding Agent，支持仓库检索、任务规划、工具调用、测试执行、Review Subagent 和 JSONL trace 记录。
- 构建面向代码仓库的 Hybrid RAG：采用 AST/tree-sitter 结构化切分，融合 Dense Embedding 与 BM25 召回，通过 RRF 和代码感知 reranker 排序，并实现增量向量索引、Git/依赖图加权及 Recall@K、MRR 评估。
- 构建 Agent 后训练数据流水线，将 Agent traces、MBPP/HumanEval 代码任务、SWE-bench Lite 修复计划统一转换为 LLaMA-Factory alpaca 格式，形成 687 条训练样本和 37 条验证样本。
- 基于 Qwen3-8B 使用 LoRA 进行 1 epoch SFT，仅训练 21.8M 参数（0.27%），跑通从模型下载、训练、checkpoint 保存到推理评估的完整闭环。
- 设计 base/SFT 对比评估脚本，从 JSON 合法率、字段命中率、ROUGE-L、工具选择准确率、文件命中率等维度量化微调效果；实验中 JSON 合法率从 0.0% 提升到 94.6%，工具选择准确率从 0.0% 提升到 83.3%。
- 在报告中区分“输出协议对齐”和“真实修代码能力”，并提出 held-out 测试集、patch 执行评估、端到端 Agent loop、RLVR/GRPO 后训练等后续路线。

## STAR 版本

**Situation**：通用大模型在代码仓库任务中经常输出长篇自然语言解释，难以稳定遵循 Agent 工具调用 JSON 协议。

**Task**：构建一个可运行的 Code Agent，并验证小规模 SFT 是否能提升模型对工具调用格式和修复计划格式的遵循能力。

**Action**：

1. 实现 Hybrid RAG 仓库检索、工具调用、Hook 安全检查、Review Subagent 和 trace 记录。
2. 将 Agent traces、MBPP/HumanEval 和 SWE-bench Lite plan 数据整理成 SFT 样本。
3. 使用 LLaMA-Factory 基于 Qwen3-8B 进行 LoRA SFT。
4. 编写评估脚本，对比 base model 与 SFT model 的 JSON 合法性、字段命中、工具选择准确率和 ROUGE-L。

**Result**：SFT 后 JSON 合法率从 0.0% 提升到 94.6%，字段命中率从 1.4% 提升到 94.6%，工具选择准确率从 0.0% 提升到 83.3%，验证了小规模 Agent SFT 对输出协议对齐的有效性。

## 面试讲解思路

1. **为什么做这个项目**：Code Agent 的核心不只是代码生成，而是可控工具调用、可验证执行和轨迹数据闭环。
2. **系统怎么设计**：LangGraph 编排多阶段节点，Hybrid RAG 提供代码上下文，Tool Registry 管理工具，Hook System 限制风险，Trace 用于后训练数据。
3. **数据怎么来**：MBPP/HumanEval 提供函数级任务，SWE-bench Lite 提供真实 issue，Agent traces 提供工具调用行为。
4. **训练怎么做**：LLaMA-Factory + Qwen3-8B + LoRA，先做 1 epoch smoke test，验证格式学习能力。
5. **结果怎么看**：指标提升很大，但主要反映格式对齐；真实修复能力还需要 patch 执行和端到端评估。

## 不建议夸大的说法

- 不要说“训练出了强大的代码修复模型”。
- 不要说“真实 SWE-bench 通过率显著提升”，除非已经跑过 SWE-bench harness。
- 不要把 JSON 合法率提升等同于真实工程能力提升。

## 更稳妥的说法

- “完成了 Code Agent 运行轨迹到 SFT 数据的闭环。”
- “验证了 SFT 对 Agent 输出协议、工具调用 schema 的对齐效果。”
- “搭建了后续做端到端 Agent 评估和 RLVR/GRPO 的基础设施。”
