# Sovereign AI — Route B Pure AI Brain & Deterministic Agent Harness (v10.32)

本仓库包含从第一性原理出发的自研主权 AI 全栈架构研究与可执行代码，严格执行**「纯 AI 大脑本体（`src/brain/`）」与「确定性外骨骼 / Agent Harness（`src/harness/`）」双轨解耦**，并由 `SYSTEM_CONSTITUTION.md`（七章 28 条最高元约束，含第 27 条全仓四层活性与反腐烂强制审计律）与 `docs/whitepapers/brain_paper_organ_registry.md`（2236 条零重复文献统一字典，98.4% 为 2025–2026 前沿文献）全链路兜底。

## 全工程单源真相（SSOT）信息地图（严格锁定 8 份 `.md`，零重复职责）

| 层级 | 文件路径 | 唯一职责（想查什么就只看这份） |
| :--- | :--- | :--- |
| **L1 总入口与规范** | `README.md` | 项目总览、硬件基线（RTX 3050 4GB）、16 大领域 77 轮子活性与冲突裁决表及全仓导航入口 |
| **L1 总入口与规范** | [`SYSTEM_CONSTITUTION.md`](SYSTEM_CONSTITUTION.md) | 七章 28 条最高工程与科研元规范（含第 27 条轮子/语料/论文四层活性反腐烂铁律） |
| **L2 架构裁决与拓扑** | [`docs/architecture/ARCHITECTURE_DECISIONS.md`](docs/architecture/ARCHITECTURE_DECISIONS.md) | **唯一架构决策与全链路追踪中枢（`ADR-001`–`ADR-013`：`✅既定主干 / ⚔️同位待PK / ❌淘汰` × `Phase 1/2/3/Scale-Up`）** |
| **L2 架构裁决与拓扑** | [`docs/architecture/ARCHITECTURE_DIAGRAMS.md`](docs/architecture/ARCHITECTURE_DIAGRAMS.md) | 唯一纯文本 Mermaid 系统拓扑图与活体模块映射（v10.32，`repo_doctor.py` 硬门禁强制同步） |
| **L3 深度设计母本** | [`docs/whitepapers/sovereign_ai_brain_whitepaper.md`](docs/whitepapers/sovereign_ai_brain_whitepaper.md) | 纯 AI 大脑第一性原理、正文演进 `0.0D`、硬件屋顶线与 2236 条权威参考文献母本 |
| **L3 深度设计母本** | [`docs/whitepapers/deterministic_agent_harness_whitepaper.md`](docs/whitepapers/deterministic_agent_harness_whitepaper.md) | 确定性外骨骼 / Agent Harness 工程协议与安全红线母本 |
| **L4 数据字典** | [`docs/whitepapers/brain_paper_organ_registry.md`](docs/whitepapers/brain_paper_organ_registry.md) | 2236 条文献统一 3 大裁决 × 6 细分字典（`✅既定主干` 175 | `⚔️待PK/扩展` 234 | `❌淘汰/禁令` 493） |
| **L4 数据字典** | [`docs/whitepapers/brain_dataset_curriculum_registry.md`](docs/whitepapers/brain_dataset_curriculum_registry.md) | 预训练语料、ADR-013 反投毒清洗、HF 存活/Gated 门禁与四阶段数据课程字典 |
| **L2 机器图谱 SSOT** | [`docs/architecture/ARCHITECTURE_GRAPH.json`](docs/architecture/ARCHITECTURE_GRAPH.json) | **11 维全串联机器可读图谱（ADR×论文×裁决×阶段×机制×AST代码×单测×硬件×成熟度×跨器官边×回滚）** |
| **L4 基准总览** | [`benchmarks/reports/BENCHMARK_SUMMARY.json`](benchmarks/reports/BENCHMARK_SUMMARY.json) | 全仓 16 份历史基准 JSON 报告统一汇总索引 |

## 工程三层物理隔离（v10.32 全栈 CTO 洁净室架构）

- **Tier 1 — 活体热路径核心（`src/brain/`，AST 强制零死类 + 零死函数）**：`sovereign_brain_pytorch.py`（含活体 `GateSightKVPruner`、`BOS Summary`、`SGA-Sparse` 与 `bitnest_spec_decode`）、`training_engine.py`、`wake_sleep_runtime.py`、`exoskeleton_bridge.py`、`ecosystem_backends.py`（16 大领域 77 轮子 + GitHub 活性门禁）、`prepare_seed_corpus.py`（含 `Mixture-then-Select`、RCE 反投毒、`--use-ube` 与 HF 存活/Gated 前置巡检）、`config_profiles.py`、`ablation_harness.py`。
- **Tier 2 — 阶段隔离与同位待 PK 池（`src/experimental/`）**：按领域拆分为 4 个隔离模块（`kernel_and_serving_planners.py`、`training_and_rl_planners.py`、`memory_and_sleep_planners.py`、`harness_and_safety_policies.py`）+ `legacy_numpy_backbone_poc.py`，并在 `src/experimental/__init__.py` 的 `PHASE_ROUTING_REGISTRY` 与 `ADR-001`–`ADR-013` 一一同构映射。
- **Tier 3 — 唯一 ADR 中枢 + 3 大裁决 × 6 细分统一字典**：[`docs/architecture/ARCHITECTURE_DECISIONS.md`](docs/architecture/ARCHITECTURE_DECISIONS.md)（`ADR-001`–`ADR-013`）统一裁决与追踪；[`brain_paper_organ_registry.md`](docs/whitepapers/brain_paper_organ_registry.md) 以脚本自动闭合核算全库 `2236` 条文献。

## 2026 顶级开源轮子全栈映射与冲突裁决（16 大领域 × 77 轮子 · `2026-11-21` GitHub API 实时实测）

**架构红线**：只自研全网独有的核心大脑拓扑（256 字节无词表 `DynamicBytePatcher` + `UBE` 中文 `-33.3%` 无损压缩 + 3:1 共享循环核 `OscillatoryDualDeltaNet`/`HoPE` + `0-MB-VRAM` CPU 锁页 Engram + 昼夜睡眠巩固 + 外骨骼）；一切通用基础设施按 **16 大领域** 严格执行 **`✅ 主干首选 / ⚔️ 同位待PK / 🚀 后端与Scale-Up / ❌ 淘汰禁用`** 冲突与主权协议裁决，并由 `python3 scripts/repo_doctor.py --audit-freshness` 对**轮子（GitHub API）+ 语料（HF Hub API）+ 论文（arXiv 年份与查重）**三线自动防腐烂巡检（代码入口：[`src/brain/ecosystem_backends.py`](src/brain/ecosystem_backends.py) 与 [`requirements-rtx3050.txt`](requirements-rtx3050.txt)）：

| 领域 | `✅ 主干首选` | `⚔️ 同位待PK / 备选` | `🚀 后端 / Scale-Up` | `❌ 淘汰禁用（冲突/闭源排雷）` |
| :--- | :--- | :--- | :--- | :--- |
| **1. GPU 算子与内核编译** | [fla](https://github.com/fla-org/flash-linear-attention)、[Liger-Kernel](https://github.com/linkedin/Liger-Kernel)（均已接热路径）、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)（推理核底座） | [Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy)（已接 CE 回退链，大词表降 37× 显存） | [FlashAttention-4](https://github.com/dao-ailab/flash-attention)（仅 Hopper/Blackwell）、[FlashQLA](https://github.com/QwenLM/FlashQLA)、[TileLang](https://github.com/tile-ai/tilelang)、[PyTorch Helion](https://github.com/pytorch/helion) | — |
| **2. 量化、QAT 与权重存储** | [safetensors](https://github.com/huggingface/safetensors)（已接热路径）、[TorchAO](https://github.com/pytorch/ao)（含 PARQ 3-bit QAT） | [Intel AutoRound](https://github.com/intel/auto-round)（SignRoundV2 PTQ，与分块 Hadamard Q4 PK） | — | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)：被 `PolarExpressMuon`+`TorchAO`+`AutoRound` 全面超越且 `torch.compile` 兼容弱 |
| **3. 微秒级受限解码** | [XGrammar-2](https://github.com/mlc-ai/xgrammar)（`<40 µs`，已接 `<tool_call>` 语法编译，CoT 零约束） | [llguidance](https://github.com/guidance-ai/llguidance)（`~50 µs` Rust 惰性掩码引擎） | — | — |
| **4. 服务引擎与端侧推理** | GPU：[SGLang](https://github.com/sgl-project/sglang)（混合 GDN 状态前缀缓存）；端侧：[llama.cpp](https://github.com/ggml-org/llama.cpp) + [bitnet.cpp](https://github.com/microsoft/BitNet) | GPU：[vLLM](https://github.com/vllm-project/vllm)（统一混合块池）；移动/跨端：[ExecuTorch](https://github.com/pytorch/executorch) / [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | — | [Ollama](https://winder.ai/vllm-vs-ollama-vs-sglang-llm-inference-comparison/)：对 11 类混合线性注意力强制单并发槽（50 并发慢 52×）且篡改 Prompt |
| **5. 分布式训练、优化器与 RLVR** | 预训练：[TorchTitan](https://github.com/pytorch/torchtitan)（FlexShard Muon 降 79% 耗时）；RLVR：[verl](https://github.com/verl-project/verl) + [AReaL 2.0](https://github.com/areal-project/areal)（全异步快 2.77×） | [slime](https://github.com/THUDM/slime)、[SkyRL](https://github.com/skyrl-project/SkyRL)、[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)、[OpenRLHF](https://github.com/openrlhf/openrlhf)、[FLA Flame](https://github.com/fla-org/flame)（停更观察） | [Microsoft Dion/Dion3](https://github.com/microsoft/dion)（分布式正交 Muon 提速 6×） | — |
| **6. 语料清洗与去重** | [DataTrove](https://github.com/huggingface/datatrove)（CPU 优先 FineWeb-Edu 流水线，配 30GB 本地中文 60:40） | — | [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)（GPU MinHash LSH `4 TB/hr`，仅万亿级启用） | — |
| **7. 基准评测与 Agent 评测** | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)、[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)（含 [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) 沙盒 Agent/SWE-bench Pro） | [EvalPlus](https://github.com/evalplus/evalplus)（停更观察 `2025-10-02`）、[OpenHands](https://github.com/OpenHands/openhands)（ACP 控制面参考） | — | — |
| **8. 外骨骼沙盒与可观测** | [Firecracker](https://github.com/firecracker-microvm/firecracker) + [E2B](https://github.com/e2b-dev/e2b)（Apache-2.0 硬件 microVM）、[Langfuse](https://github.com/langfuse/langfuse)（MIT 自托管追踪） | — | — | [Daytona](https://github.com/daytonaio/daytona)（API 实测 `archived=True` 已闭源）；[Arize Phoenix](https://github.com/Arize-ai/phoenix)（Elastic-2.0 非 OSI 且被 Dynatrace 收购） |
| **9. 流式加载与高速数据工具** | [LitData](https://github.com/Lightning-AI/litdata)（零拷贝断点续训）、[Polars](https://github.com/pola-rs/polars)、[DuckDB](https://github.com/duckdb/duckdb)、[uv](https://github.com/astral-sh/uv)、[Distilabel](https://github.com/argilla-io/distilabel) | [MosaicML Streaming](https://github.com/mosaicml/streaming)、[Bespoke Curator](https://github.com/bespokelabsai/curator)、[WebDataset](https://github.com/webdataset/webdataset)（停更观察） | — | — |
| **10. KV 分级卸载、看板与护栏** | [LMCache](https://github.com/LMCache/LMCache)（CPU/SSD 多级 KV 复用，3–10× TTFT）、[SwanLab](https://github.com/SwanHubX/SwanLab)（中文主权本地训练看板） | [Trackio](https://github.com/gradio-app/trackio)、[MLflow](https://github.com/mlflow/mlflow)、[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)、[PurpleLlama](https://github.com/meta-llama/PurpleLlama) | [Mooncake](https://github.com/kvcache-ai/Mooncake)（Kimi KVCache 分离架构与 RDMA 池） | — |
| **11. 形式化裁判与 STEM 解析** | [Lean 4](https://github.com/leanprover/lean4) + [Pantograph](https://github.com/lenianiva/Pantograph)、[Z3](https://github.com/Z3Prover/z3)、[SandboxFusion](https://github.com/bytedance/SandboxFusion)（RLVR 零幻觉裁判）、[MinerU](https://github.com/opendatalab/MinerU)（中英 STEM/LaTeX 解析） | [Docling](https://github.com/docling-project/docling)（MIT 商用友好解析）、[olmOCR](https://github.com/allenai/olmocr)（停更观察） | — | — |
| **12. 本地向量检索与 MCP 协议** | [USearch](https://github.com/unum-cloud/usearch)（零守护进程 SIMD 索引，0 MB 显存承载前沿定理/Mathlib 外置检索）、[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | [Qdrant](https://github.com/qdrant/qdrant)（Scale-Up 服务版混合检索） | — | — |
| **13. 代码 AST 与静态分析** | [Tree-sitter](https://github.com/tree-sitter/tree-sitter)（增量多语言 AST）、[Ruff](https://github.com/astral-sh/ruff)（100× Rust Python Linter）、[ripgrep](https://github.com/BurntSushi/ripgrep)（极速代码库检索） | — | — | — |
| **14. 网页抓取与浏览器外骨骼** | [Crawl4AI](https://github.com/unclecode/crawl4ai)（异步转洁净 Markdown）、[Playwright](https://github.com/microsoft/playwright-python)（确定性无头浏览器）、[Trafilatura](https://github.com/adbar/trafilatura)（Ultra-FineWeb 同款正文提取） | — | — | — |
| **15. 白盒机械可解释性** | [nnsight](https://github.com/ndif-team/nnsight)（原生支持自定义 DeltaNet+HoPE 干预）、[SAELens](https://github.com/decoderesearch/SAELens)（稀疏自编码器电路探针） | — | — | [TransformerLens](https://nnsight.net/blog/2026/02/26/nnterp-integration/)：仅重实现标准 Transformer，不兼容本仓 DeltaNet+HoPE |
| **16. 端侧离线语音外设** | [whisper.cpp](https://github.com/ggml-org/whisper.cpp)（C/C++ 量化本地 STT）、[Kokoro-82M](https://github.com/hexgrad/kokoro)（Apache-2.0 82M CPU 中英实时 TTS，0 MB 显存） | — | — | — |

## 核心架构主线（Route B v10.32：时空解耦与递归涌现）

- **L0 无损字节感知与并行解码**：`CanonCausalConv1D` + `DynamicBytePatcher`（Step-0 锁定 LZ77 表 + `UBE` 中文 `-33.3%` 无损压缩 + `ATDC` + `LCA` `4×` 并行解码）。
- **L1 3:1 双工共享循环核（时间换空间）**：三明治拓扑 `Prelude (2层) -> L2/L3 调质与 CPU 锁页 Engram -> 共享 3:1 循环核 (4层 × 8轮 = 32层) -> Coda (2层)`，有效深度 **36 层**；75% `OscillatoryDualDeltaNet` + 25% `HoPEHLAMELTAttention`（集成 `GateSightKVPruner` 释放 90% KV + `BOS Summary` 零参削弱 Sink）+ `SGA-Sparse` 原生 SwiGLU 75–85% 稀疏 + `bitnest_spec_decode()` 零显存推测。
- **L2–L4 调质、冷记忆、睡眠固化与外骨骼安全**：`TernaryNeuromodulator` + `CPUPinnedEngramMemory`（`0.0 MB` GPU 显存）+ `WakeSleepBrainRuntime` + `Soteria`/`TenonAgent`/`QHijackCrossPrecisionAuditGate`/`PrefixCacheMetastabilityBudgeter`。
- **RTX 3050 4GB 训推闭环**：训练期原生 `BF16` AMP + `PolarExpressMuonOptimizer`（全链路 0 `.item()` GPU-CPU 同步）；推理导出 `export_q4_inference_state(block_size=32)` 分块 Hadamard + Q4 打包。

## 仓库目录结构

```text
ai/
├── README.md
├── SYSTEM_CONSTITUTION.md                        # 七章 28 条最高元约束宪章（v10.32）
├── pyproject.toml                                # PEP 621 标准打包入口
├── requirements.txt
├── requirements-rtx3050.txt                      # RTX 3050 4GB 16大领域77轮子依赖清单
├── docs/
│   ├── architecture/
│   │   ├── ARCHITECTURE_DECISIONS.md             # 唯一两维正交 CTO 架构决策与追踪中枢（ADR-001~013）
│   │   └── ARCHITECTURE_DIAGRAMS.md              # 6 张纯文本 Mermaid 活体工程架构图（v10.32，CI 强校验）
│   └── whitepapers/
│       ├── sovereign_ai_brain_whitepaper.md      # 纯 AI 大脑唯一主白皮书（2236 条零重复文献）
│       ├── brain_paper_organ_registry.md         # 12 器官 × 3 大裁决 × 6 细分统一字典（2236 条）
│       ├── brain_dataset_curriculum_registry.md  # 2026 四阶段数据集课程与 ADR-013 反投毒字典（v10.32）
│       └── deterministic_agent_harness_whitepaper.md
├── src/
│   ├── brain/                                    # Tier 1 活体主干（AST 强制零死类、零死函数）
│   ├── experimental/                             # Tier 2 阶段隔离与同位待 PK 池（含 legacy_numpy_backbone_poc.py）
│   └── harness/                                  # 确定性外骨骼运行时
├── benchmarks/reports/BENCHMARK_SUMMARY.json     # 16 份历史基准 JSON 统一索引
├── tests/
│   ├── test_01_live_brain_core.py                # Tier 1 活体核心单测
│   ├── test_02_experimental_prototypes.py        # Tier 2 阶段原型单测
│   └── test_03_governance_and_ssot.py            # Tier 3 治理/SSOT/架构图同步单测
└── scripts/
    └── repo_doctor.py                            # 一键 CTO 架构体检官 CLI（v10.32，含 --audit-freshness 三线活性巡检）
```

## 本地验证与原型晋级速查

- **一键 CTO 体检与全量单测**：`python3 scripts/repo_doctor.py --run-tests`（强制校验 8 份 `.md` 上限、AST 零死类/零死函数、Mermaid 架构图同步、ADR-001~013、2236 条文献闭环与离线三线活性门禁）
- **在线三线活性防腐烂巡检（GitHub API + HF Hub API + arXiv 年份审计）**：`python3 scripts/repo_doctor.py --audit-freshness`
- **RTX 3050 4GB 生产环境一键装轮子**：`pip install -r requirements-rtx3050.txt`
- **RTX 3050 4GB 烟雾验证**（待用户解除 `GAP-10` 后）：`pip install -e ".[cuda,dev]" && python3 src/brain/sovereign_brain_pytorch.py --config rtx3050_4gb_65m --steps 20`
- **单条新突破 6 步强制闭环**：去重裁决 → 真实代码落点 → 合入 `ARCHITECTURE_DECISIONS.md` → 同步 `ARCHITECTURE_DIAGRAMS.md` → 更新白皮书 `0.0D` 与文献字典自动重算 → 跑通 `repo_doctor.py --run-tests`

## v11.0 工程融合升级（2026-11 原生 Windows + 悟性/刹车闭环 + 2B 强底座双路径）

- **原生 Windows 2026 执行栈（取代 WSL2 默认方案）**：官方 [`triton-lang/triton-windows` 3.9](https://github.com/triton-lang/triton-windows) 已原生支持 PyTorch 2.15 与 Ampere (`sm_86`, RTX 3050)，内置 `ptxas 12.9` 与 TinyCC；配合 `WindowsNativeRuntimeGuard` 将全量缓存锁死在 `D:\Environment\.cache`，消除 WSL2 3%–15% GPU 虚拟化税与 15%–30% `/mnt/d/` 跨盘 I/O 损耗；`CPUPinnedEngramMemory.lookup` 已加装 `@_compiler_disable` 隔离 `torch.compile` CPU 指针崩溃。
- **数据流两大真实 Bug 根治 + 5 维硬特征掐尖 + 5 原语因果 IR 生成器**：
  - `prepare_seed_corpus.py` 流式拉取强制开启 `.shuffle(seed=42, buffer_size=10000)` 与加权交错混写（`weighted_interleave=True`），彻底消除 Parquet 顺序分片截断偏科与逐源串行写入遗忘；
  - 新增零 gated 高纯浓缩配置 `rtx3050_4gb_super_dense_v11`（含 [`nvidia/SWE-Hero-openhands-trajectories`](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) 与 [`allenai/Dolci-Think-SFT-7B`](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B)）及 `score_super_density_sample` 5 维前置掐尖筛子；
  - 新增 `CausalIRConflictGenerator`（`State / Invariant / CausalEdge / Conflict / Intervention` + 显式 `UnresolvedHole` 节点 + 跨外壳换皮 + Context Distillation 规则内化）。
- **元认知刹车 + 亚里士多德邻域探针 + 外部 2B 强底座双接法**：
  - `GracefulRLRewardEngine`（[arXiv:2610.20401](https://arxiv.org/pdf/2610.20401)）：实现严格偏序奖励 `做对 (+1.0) > 诚实拒答/报洞 (+0.20) > 过度拒答 (-0.25) > 瞎编 (-1.0)`；
  - `AristotelianNeighborhoodProbe`（[arXiv:2602.14486](https://arxiv.org/html/2602.14486v2) / [arXiv:2609.27252](https://arxiv.org/abs/2609.27252)）：基于 `mKNN` 局部邻域一致度测量极小模型表征收敛；
  - `ZeroInitGatedEngramAdapter` + `EDGE_BACKBONES_2026_11`（[`OVHaiLLM/Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B)、[`NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB)、[`Gemma 4 E2B`](https://ai.google.dev/gemma/docs/core/model_card_4)、[`EmbeddingGemma 2 (740M)`](https://blog.google/innovation-and-ai/technology/ai/google-ai-updates-october-2026/)）：支持 Route B 原创从零预训练与成熟 2B 强底座双路径无缝挂载。

## v12.2 原生 Windows 一键开干（2026-11 最新依赖全栈 + 统一数据闭环）

- **最新原生 Windows 依赖栈（零 C 盘写入，全锁 `D:\Environment`）**：推荐 **Python `3.15.2` / `3.14`**（[升级说明](https://blog.codercops.com/blog/python-3-10-end-of-life-upgrade-to-3-14-2026)）+ **[PyTorch `2.15.0`](https://pytorch.org/blog/pytorch-2-15-release-blog/)（`cu134` 最新 / `cu132` 备选）** + 官方 **[`triton-windows` `3.9.0.post30`](https://github.com/triton-lang/triton-windows)** + **[`flash-linear-attention` `0.6.0` + `liger-kernel` `0.8.5`](https://pypi.org/project/fla-core/0.5.2/)** + **[`transformers` `5.24.0` + `datasets` `5.2.0` + `torchao` `0.18.0` + `safetensors` `0.8.0` + `hf-xet` `1.8.0`](https://pypi.org/project/transformers/)** / `huggingface-hub` / `datasets` / `safetensors` / `torchao` / `liger-kernel`（严禁安装 `bitsandbytes`）。
- **一键装机与自检**：`powershell -ExecutionPolicy Bypass -File scripts\setup_windows_d_drive.ps1`（自动锁定 `D:\Environment\.cache`、按 `nvidia-smi` 自动选 CUDA wheel、跑 GPU/Triton 与全仓 `repo_doctor.py --run-tests`）。
- **一键 200MB 掐尖冒烟 + 3050 4GB 实测**：`powershell -ExecutionPolicy Bypass -File scripts\run_smoke_windows.ps1`（默认走零门控 `rtx3050_4gb_super_dense_v11`：80% 开源 Tier-S 实时 5 维掐尖 + 20% 内生 5 原语因果冲突/拒答燃料，随后以 `np.memmap` 零内存膨胀直接喂入 `sovereign_brain_pytorch.py --data-bin` 跑 20 步实测并打印峰值显存与吞吐）。
