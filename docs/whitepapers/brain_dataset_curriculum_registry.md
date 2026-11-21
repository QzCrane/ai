# Sovereign AI Brain 2026 全球顶级数据集分类与四阶段课程总控表（v10.32）

> **设计原则（彻底告别散乱堆砌）**：所有开源数据集严格按 **“四阶段培养管线（Stage 0–4）× 目标喂入器官（GPU 逻辑核 / CPU Engram 嫁接 / 睡眠共现 / Harness 外置知识库）× 硬件档位预算配方 × 负面偏差红线”** 四维正交组织，与一键流式构建脚本 `src/brain/prepare_seed_corpus.py` 100% 绑定。

---

## 一、总控决策图：不同类型的数据应该喂给系统的哪个器官？

| 数据性质 | 绝对不能放哪里 | 唯一正确归属器官 | 物理原因（防遗忘红线） |
| :--- | :--- | :--- | :--- |
| **语法、算法、代码逻辑、数理推演规律** | 外部 RAG 检索库（太慢、无泛化） | **L0/L1 GPU 紧致循环核权重**（Stage 1/3/4） | `~0.008 bits/param` 推理电路极珍贵，一旦学会递归算法可泛化至无限输入 |
| **高频固定模式、常见 API 签名、专有名词** | GPU 神经核权重（`2.0 bits/param` 瞬间撑爆） | **L3 CPU 锁页 `CPUPinnedEngramMemory`**（Stage 2 离线 Grafting，`0.0 MB` 显存） | 剥离死记硬背，微秒级 $O(1)$ 最长后缀精确命中 |
| **跨领域已验真历史片段** | 单遍刷过即丢弃 | **L4 CPU `CoObservationSleepBuffer`**（空闲期睡眠共现固化） | 打破顺序学习无共现表征天花板（`arXiv:2608.18803v1`） |
| **海量长尾百科、个人文档、实时变动资料** | 任何神经权重（训完即过时且易幻觉） | **Agent Harness 外置本地知识库 / SQLite / MCP**（非大脑本体） | 要求 100% 精确溯源、可随时增删审计，必须留在离散外骨骼层 |

---

## 二、按五阶段培养管线精确分类的 2025–2026 全球顶级开源数据集全谱

### Stage 0：Step-0 LZ77 Novelty 表拟合（仅需 `2–10 MB`，训前永久锁定）
* **目标器官**：`L0 DynamicBytePatcher._lz77_novelty_cpu`（`0.25 MB` CPU RAM / `0.0 MB` 显存）。
* **数据来源**：从 Stage 1 最终混排好的 `seed_corpus.bin` 头部截取 `2–10 MB` 代表性字节流（调用 `init_step0_novelty_from_file()`）。
* **硬红线**：必须在 Step 0 之前锁定，训后严禁更换分块规则（`arXiv:2610.22100v1` 实证训后换规则导致 12M 精度从 `73.6%` 坍塌至 `0.0–0.5%`）。

### Stage 1：真实锚点高密度预训练底座（真实数据占比硬红线 $\ge 30\%$，推荐 `75–80%`）

| 子类 | 数据集名称与 HuggingFace / 官方直链 | 规模与特点 | 喂入目标与适用档位 | 负面边界与避坑警示 |
| :--- | :--- | :--- | :--- | :--- |
| **1A. 真实代码（核心逻辑锚点）** | [NVIDIA Nemotron-Pretraining-Code-v3 & Nemotron-CC-Math-v1 (133.3B 4plus)](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) | `427.9B` tokens 工业级清洗代码（2025-12 发布，持续维护） | `L0/L1` 主力代码底座（占比 `35–40%`） | 需在 HuggingFace 点击同意公开协议方可流式拉取 |
| **1A. 真实代码（小模型精选）** | [SmolLM3 Pretraining Datasets](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets)（含 Stack-Edu / FineMath） | 专为小参数高密度模型调优的配比合集 | `micro_debug` & `rtx3050_4gb` 极佳替代/补充 | 需注意与 Nemotron-Code 交叉 SHA-256 去重 |
| **1B. 英文教育与学术 PDF** | [FineWeb-Edu (`sample-10BT`)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 教育价值分类器精选网页，自带 `10BT` 小分片 | `L0/L1` 英文科学/教科书骨架（占比 `20–25%`） | 偏重科普/教科书，缺乏真实代码与对话格式 |
| **1B. 英文高密度混合包** | [FinePDFs-Edu 50BT + DCLM 30BT + FineWeb-Edu 20BT](https://huggingface.co/datasets/HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled) | `100B` tokens 预混排学术 PDF + DCLM + 教育网页 | `cluster_1b` 放量档英文主底盘 | 单文档偏长，流式读取须配合 `seq_len` 切片 |
| **1B. 英文超大规模清洗** | [NVIDIA Nemotron-CC-v2.2](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) | `6.6T` tokens 高质量清洗与合成重述（含 `sample` 小包） | `cluster_1b` 放量档扩展 | 含部分 LLM 重述文本，不可单独替代真实代码锚点 |
| **1C. 中文最强开源底盘** | [OpenBMB Ultra-FineWeb (`en 2.2T + zh 120B`)](https://openbmb.cn/en/ultradata/datasets/Ultra-FineWeb/) | MiniCPM4 同源，验证式质量分类器过滤（融合 CCI3/IndustryCorpus2/WanJuan 等） | `L0/L1` 中文主底盘（占比 `15%`） | 中文占比过高会挤占 65M 代码预算，建议锁在 `15–20%` |
| **1C. 多语言广度底座** | [FineWeb2 (`HuggingFaceFW/fineweb-2`)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) / [FineWeb2-HQ](https://github.com/epfml/fineweb2-hq) | `20 TB`、`1,000+` 语言 | 仅在需要多语言扩展时按需截取 | 65M 本地专精档不建议大比例混入小语种 |
| **1D. 端侧小模型全栈混合** | [Domyn-Edge-Pretraining-Mix-8.4T](https://huggingface.co/datasets/domyn/Domyn-Edge-Pretraining-Mix-8.4T/blob/main/README.md) | 2026-10 最新开源端侧 8.4T 预训练混合包 | `cluster_1b` 端侧放量参考 | 2026-10 首批发布暂扣留部分 stack-edu 文件待补署名 |

### Stage 2：离线 Memory Grafting 静态知识嫁接源（仅喂 CPU RAM，`0.0 MB` GPU 显存）

| 数据集 / 来源 | 获取方式 | 喂入接口与作用 |
| :--- | :--- | :--- |
| **FineWiki / 维基百科精华** | [HuggingFaceFW Collections (`finewiki`)](https://huggingface.co/HuggingFaceFW/collections) | 经 `training_engine.extract_and_graft_ngram_facts()` 离线抽取高频实体/术语隐状态写入 CPU 锁页精确后缀表 |
| **高频编程语言标准库与 API 文档** | 本地 Python/Rust/C++ 标准库文档 + `Nemotron-Code-v2` 高频头部 | 卸载死记硬背的 API 签名，避免占用 GPU 逻辑核容量 |

### Stage 3：高纯度思维链与逻辑直觉开蒙（必须用 DLE `0x1C`–`0x1F` 帧隔离思维与结论）

| 子类 | 数据集名称与直链 | 规模与特点 | 核心作用与适用阶段 |
| :--- | :--- | :--- | :--- |
| **3A. 通用多域最强开源思维链** | [OpenThoughts3-1.2M (ICLR 2026 Oral)](https://openreview.net/forum?id=7xjoTuaNmN) / [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) / [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M) | 85 万数学 + 25 万代码 + 10 万科学（1,000+ 次受控消融调优） | Stage 3 逻辑开蒙主力（65M 先用 `114k` 轻量版） |
| **3B. 数学专项深度推演** | [HuggingFaceTB/FineMath](https://huggingface.co/collections/joezgao/datasets-and-evals-worth-knowing)、[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)、[OpenR1-Math-220k](https://huggingface.co/collections/open-r1/reasoning-datasets) | 覆盖预微积分至竞赛数学的分步推导 | 激活共享循环核多步推演深度 |
| **3C. 极小高纯度激活种子** | `GAIR/LIMO`（817 题）与 `simplescaling/s1K`（1,000 题，均在 [OpenR1 合集](https://huggingface.co/collections/open-r1/reasoning-datasets)） | 极高纯度手工精选难题 | 预训练末期退火/退火后轻量 SFT 激活深思习惯 |
| **3D. 软件工程 Agent 轨迹** | [SWE-rebench OpenHands 轨迹](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | 真实 GitHub 仓库定位、修改与测试多轮轨迹 | 训练大脑通过 `exoskeleton_bridge.py` 协同外骨骼工具 |

### Stage 4：断奶后 RLVR / GRPO 自博弈客观验真题库（100% 外部裁判打分，零老师依赖）

| 子类 | 数据集名称与直链 | 客观验证器类型 | 喂入接口 |
| :--- | :--- | :--- | :--- |
| **4A. 可执行代码单测题库** | `KodCode/KodCode-V1`（[OpenR1 合集](https://huggingface.co/collections/open-r1/reasoning-datasets)）、[SWE-smith (`50K` 仓库实例)](https://openreview.net/forum?id=63iVrXc8cC&noteId=Rpf3zZjbdn) | Python/Pytest 沙盒单元测试（Pass/Fail） | `training_engine.rlvr_grpo_self_play_step()` |
| **4B. 严格可验真数学题库** | `SynthLabsAI/Big-Math-RL-Verified`、`BytedTsinghua-SIA/DAPO-Math-17k`、`virtuoussy/Multi-subject-RLVR`（[OpenR1 合集](https://huggingface.co/collections/open-r1/reasoning-datasets)） | 符号数学等价验真（SymPy / 精确答案匹配） | `training_engine.rlvr_grpo_self_play_step()` |
| **4C. Lean 4 形式化定理证明与纠错** | [Goedel-Prover-V2](https://github.com/Goedel-LM/Goedel-Prover-V2)、[APRIL (`uw-math-ai/APRIL`, `260K` 报错-修复对, `arXiv:2602.02990v1`)](https://arxiv.org/html/2602.02990v1) | Lean 4 形式化内核编译器（0 幻觉绝对客观裁判） | `rlvr_grpo_self_play_step()` + 编译器报错自纠错训练 |

---

## 三、三档硬件一键配比配方（直接对应 `prepare_seed_corpus.py --profile`）

| 配方名（`--profile`） | 总字节规模 | 真实代码 | 英文教育/PDF | 中文 Ultra-FineWeb | 验真推理（DLE 帧） | 真实锚点总占比 | 适用场景 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`micro_debug_200mb`** | `200 MB` | `45%` | `25%` | `10%` | `20%` | **`80%`** | 本地 5 分钟冒烟测试与 7 组控制变量消融 |
| **`rtx3050_4gb_5gb`** | `5 GB` (`~1.25B` tok) | `40%` | `25%` | `15%` | `20%` | **`80%`** | RTX 3050 4GB 35M–65M 专精大脑黄金冷启动 |
| **`rtx3050_4gb_prod_30gb`** | `30 GB` (`~7.5B–10B` tok) | `35%` | `31%` (FineWeb-Edu 18% + MegaMath 13%) | `14%` | `20%` (CoT 12% + SWE轨迹 5% + 安全/拒答免疫 3%) | **`80%`** | **v10.32 新增**：65M 真实生产级 `115×–150×` Chinchilla 过训闭环档位 |
| **`cluster_1b_80gb`** | `80 GB` (`~20B` tok) | `35%` | `30%` | `15%` | `20%` | **`80%`** | 1B 完全体 Chinchilla 底线预训练 |

## 五、2026 数据工程硬核配方与反投毒红线（ADR-013，v10.32 升级）

| 环节 | 采用机制 / 公开基准 | 核心收益与物理边界 | 一手来源 |
| :--- | :--- | :--- | :--- |
| **高纯度公开底座** | `FineWeb-Edu` / `DCLM` / `Nemotron-CC-Math` / `Limite-1B` 数学课程 | 1B 参数 237B 精选数学语料 + RLVR 达 AIME 2026 `94.01%` | [Nemotron-CC](https://developer.nvidia.com/blog/announcing-nemotron-cc-a-trillion-token-english-language-dataset-for-llm-pretraining/), [arXiv:2611.21646](https://arxiv.org/abs/2611.21646) |
| **跨域配比选样** | `Mixture-then-Select` 两阶段组合选样（已接入 `prepare_seed_corpus.py`） | 先定跨域配额再域内排序，捕捉非加性互补性，领先单样本选样 `+20–54%` | [arXiv:2610.14028](https://arxiv.org/abs/2610.14028) |
| **在线批次去噪** | `LOOM` 留一法批次梯度 SNR 选样（`src/experimental/`） | 免外部验证集、半批次误差低于全批次、剔除 `>80%` 注入标签噪声 | [arXiv:2610.00436](https://arxiv.org/abs/2610.00436) |
| **合成数据治理** | 群体级（Group-Level）合成数据筛选 + 多语种 MLP 质量迁移（100+ 语种） | 证伪纯单样本打分在高合成占比下的盲区；英文分类器低成本迁至中文 | [arXiv:2610.00779](https://arxiv.org/abs/2610.00779), [arXiv:2610.11585](https://arxiv.org/abs/2610.11585) |
| **反坏样本隐蔽化** | `GRAFT` 牺牲适配器梯度路由（135M–32B 实测） | 将可疑/Reward-Hacking 样本梯度导入牺牲参数并在导出时切除，不教模型隐藏坏行为 | [arXiv:2610.17062](https://arxiv.org/abs/2610.17062) |
| **⛔ 反投毒绝对红线** | `sanitize_pretraining_rce_triggers()`（已接入 `prepare_seed_corpus.py`） | 仅 `0.0005%–0.1%` 投毒网页即可在 0.6B–4B CLI Agent 植入 `~92%` 成功率 RCE 后门且穿透干净后训练 | [arXiv:2610.37741](https://arxiv.org/abs/2610.37741) |
| **PT/MT × LR 协同与域名抓取** | `compute_coupled_pt_mt_lr_schedule()` + `score_and_prioritize_hosts()`（已接入活体） | 必须等高纯度 MT 数据进入后再启动 LR 退火；域名先抽样打分再深度抓取获数倍高质量页面 | [arXiv:2610.29348](https://arxiv.org/abs/2610.29348), [arXiv:2610.13321](https://arxiv.org/abs/2610.13321) |
| **2026 顶级直下数学与课程材料** | [openbmb/UltraData-Math (290B+ L1/L2/L3)](https://huggingface.co/datasets/openbmb/UltraData-Math/blob/31d9ec48d0e239e53376c816b2e18eddf52ce8fd/README.md) · [zgcagi/ZGCM-1-Data (4.2T 两阶段课程)](https://huggingface.co/datasets/zgcagi/ZGCM-1-Data) · [nvidia/Nemotron-Math-Proofs-v3-SFT (414K IMO 金牌证明轨迹)](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v3-SFT) | MiniCPM 同源三级渐进数学底座 + 实测 AIME 2026 75% 开源课程 + CC-BY-4.0 证明/验证/元验证长轨迹 | 2026 HuggingFace 实查直链 |

### Stage 5：独立评测基准与多轮工具沙盒补齐（v10.32 补齐最后 5% 材料清单，严禁混入训练集）
| 维度 | 2026 标准评测集 / 工具题库 | 对应脚本 / 验收指标 |
| :--- | :--- | :--- |
| **可执行代码** | `HumanEval+` / `MBPP+` / `LiveCodeBench`（防污染） | `prepare_seed_corpus.py::get_eval_suite_manifest()` · `pass@1` |
| **可验证数学推理** | `GSM8K` / `MATH-500` / `AIME` | 外部确定性验算器精确匹配 |
| **超长上下文精确召回** | `RULER` / `Multi-Query NIAH`（4K–128K） | 验证 25% 注意力层 + CPU `Engram` 联动召回 |
| **多轮工具与自博弈** | `Nemotron-SFT-Math-v3` / `Tool-R0` (`arXiv:2602.21320v1`) / `R-Zero` (`arXiv:2508.05004`) | 端到端沙盒执行成功率 + `0/63` 物理缓存残影泄漏 |

---

## 六、v10.32 真实生产语料储量审计与 65M / 1B 过训缺口闭环（硬核量化裁决）

1. **上游开源原始储量（绝对充足，>20T+ Tokens）**：当前已登记直链开源底盘超过 `20 Trillion+` Tokens（含 `Nemotron-CC-v2.2` `6.6T`、`Domyn-Edge` `8.4T`、`Ultra-FineWeb` `2.32T`、`FineWeb-2` `20TB`、`Nemotron-Code-v3` `427.9B`、`MegaMath` `371B`、`MixtureVitae` `422B`、`UltraData-Math` `290B+`、`SWE-Universe` `807,693` 真实可验真仓库任务），**原材料储量不存在短缺**。
2. **本地 `prepare_seed_corpus.py` 此前存在的 3 个真实生产缺口（已在 v10.32 补齐）**：
   - **过训深度缺口**：旧版 `rtx3050_4gb_5gb`（`5 GB ≈ 1.25B tok`）仅达到 65M 参数的 Chinchilla 1× 冷启动下限（`20 tok/param ≈ 1.3B`），远未达到现代端侧强推理 SLM 所需的 `100×–300×` 过训区间（`SmolLM2` [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)）。v10.32 已新增 **`rtx3050_4gb_prod_30gb`**（`30 GB ≈ 7.5B–10B tok`，`115×–150×` Chinchilla，兼顾 [arXiv:2605.26097](https://arxiv.org/abs/2605.26097) 警示的过训容量遗忘上限）。
   - **多轮工具与 SWE 轨迹缺口**：旧版流式脚本未接通任何多轮代码仓库交互轨迹；v10.32 已接入 `nebius/SWE-rebench-openhands-trajectories`（8% 权重），遵循 [arXiv:2610.22019](https://arxiv.org/abs/2610.22019) 中训预热原则。
   - **拒答与提示注入免疫缺口**：旧版流式脚本缺乏拒答与防间接注入免疫切片（`AbstentionBench` [arXiv:2506.09038](https://arxiv.org/abs/2506.09038) 实证纯推理微调会恶化拒答率）；v10.32 已接入 `LLMail-Inject` / 拒答免疫切片（4% 权重）。
3. **物理落盘前提**：当前沙盒未预下载数十 GB 原始二进制语料；在真实 RTX 3050 4GB 宿主机启动生产训练前，需准备 `50–150 GB` 本地 NVMe SSD 与 HuggingFace 令牌，执行 `python3 src/brain/prepare_seed_corpus.py --profile rtx3050_4gb_prod_30gb` 流式去重落盘。

---

## 七、v10.32「20T+ 语料池新旧质量分级审计」与老旧语料淘汰令

> **铁律**：对 65M 参数紧致大脑而言，`1:7` 脏数据会直接摧毁 `20×` 有效容量（[arXiv:2609.37891](https://arxiv.org/abs/2609.37891)）；2026 中文 Common Crawl 原始快照含 `68.72%` 赌博/成人污染（[arXiv:2608.10678](https://arxiv.org/abs/2608.10678)）；老旧快照存在不可逆时序泛化衰减（[arXiv:2610.17018](https://arxiv.org/abs/2610.17018)）。因此 65M 主配方严禁混入 2024 及更早泛爬网页。

| 质量梯队 | 年份 | 代表数据集与规模 | 合计可用规模 | 65M 生产处置策略 |
| :--- | :---: | :--- | :---: | :--- |
| **Tier-S（2025–2026 最高密度黄金池）** | `2025–2026` | `Domyn-Edge` (7.59T) · `Nemotron-CC-v2.1 High` (2.5T) · `Ultra-FineWeb` (2.32T, 含 2025-30 快照) · `Nemotron-Code-v3` (2026-05) / `CC-Code-v1` (428B) · `ClimbMix` (400B) · `MegaMath` (371B) · `UltraData-Math` (290B+) · `FinePDFs-DCLM` (100B) · `Dolci-Think-SFT-Olmo-Hybrid` (2026-03) · `OpenCodeReasoning-II` (2.5M) · `KernelMosaic` (14.5K CUDA 轨迹) · `SWE-Universe` (807K) | **`~13.5T+` Tokens** | **唯一允许进入 `rtx3050_4gb_prod_30gb` 主配方**（65M 仅需截取其中最顶尖的 `0.06% ≈ 7.5B–10B` tok，且重复多轮高质数据远胜掺低质新数据 [arXiv:2610.05126](https://arxiv.org/abs/2610.05126)） |
| **Tier-A（2025–2026 合规/专项扩展池）** | `2025–2026` | `MixtureVitae` (422B) · `MultiSynt/MT` (4.8T) · `Essential-Web` 顶级子集 · `ZGCM-1-Data` (4.2T) · `MegaScience` (1.25M) · `OmniThought` (2M RV/CD 标注) | **`~9T+` Tokens** | 供 `1B` 放量档与 Stage 3/4 专项定向退火按需截取 |
| **Tier-B（2024 及更早泛爬/中低质档）** | `<=2024` | `FineWeb-Edu` (2024) · `FineWeb-2` 原始多语 (20TB) · `Zyda-2` (5T) · `HPLT` 原始泛爬 · `Nemotron-CC Medium` | **`~8T–10T+` Tokens** | **已从 `rtx3050_4gb_prod_30gb` 彻底淘汰**（仅保留在 `micro_debug_200mb` 历史兼容对照中） |

---

## 八、v10.32 顶级实验室独特资源对照 + S+ 增补池 + 65M 七维开蒙精确蓝图

1. **顶级闭源/半闭源实验室的 5 大独特壁垒（开源池之外的真实差距）**：
   - 多智能体种子合成教科书与边界反例（`Phi-4` [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)）与分块保真改写榨干高质 token（`Kimi K2` [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)）；
   - 自建上千个交互式可执行沙盒环境做大规模 Agent RL（`DeepSeek-V3.2` 1,800+ 环境 / 85K 题 [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)）；
   - 私有 VLM 精解出版图书/论文 PDF 与 150B+ GitHub Issue-PR/Diff/CI 构建日志中训（`Qwen3` [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)、`GLM-5/Kimi-Dev` [arXiv:2610.22019](https://arxiv.org/abs/2610.22019)）；
   - 真实线上用户多轮纠偏日志飞轮；
   - 数据-模型协同演化动态配比（`MobileLLM-R1` 仅用 11.7% token 追平 36T 竞品 [arXiv:2509.24945](https://arxiv.org/abs/2509.24945)）。
2. **本轮新增开源 S+ 顶尖平替源**：`allenai/dolma3.5_pool` (~10T) 与 `allenai/dolma3_dolmino_mix-100B-1125`、`opencsg/Fineweb-Edu-Chinese-V3`（100,442 册真出版中文图书教材提取）、`HuggingFaceTB/smoltalk2`（自然说话/多轮对话）、`AgentPack`（60GB 代码 Diff/Commit）、`SYNTHETIC-2` + `SynLogic/Enigmata`（零污染纯逻辑谜题）与 `NCA` 预-预训练序列（164M 超越 1.6B C4，[arXiv:2603.10055](https://arxiv.org/abs/2603.10055)）。
3. **65M 精确投喂总量与五阶段节奏（已固化进 `compute_exact_65m_curriculum_blueprint()`）**：
   - **Stage 0.5 悟性预热（`0.18B` tok / `~0.7 GB`）**：NCA + SynLogic 纯规律电路；
   - **Stage 1 主预训练（`8.5B` tok / `~34 GB`，`130×` TPP 甜点区）**：80% 真实 S+ 锚点 + 20% 推理/对话/安全，停在 `>500×` 遗忘饱和悬崖之前；
   - **Stage 2 中训退火与扩窗（`1.0B` tok / `~4 GB`）**：出版教材 + GitHub Diff/CI + 10% Alignment Pretraining（[arXiv:2601.10160](https://arxiv.org/abs/2601.10160)），上下文扩至 16K；
   - **Stage 3 克制 SFT 学会说话与思维链（`0.10B` tok / `~0.4 GB`）**：`SmolTalk2` + `Dolci-Hybrid` + 中文 R1 CoT（不长训，留足 RL 空间）；
   - **Stage 4 断奶 RLVR 自博弈（`~120K` 次可验真 Rollout）**：Pytest / SymPy / Lean 4 / SWE-Universe 客观打分。


## 九、v10.32 中文交互 60:40 生产配比、双正交分类（4大内容域 × 7+1能力 × 6阶段）与顶级数据寻源实战手册

1. **中文交互优先 `60:40` 生产配比（`rtx3050_4gb_prod_30gb` 实装）**：
   - **英码+数学+STEM（`59%`）**：`Nemotron-Pretraining-Code-v3` (`24%`) + `Dolmino-3` (`11%`) + `MegaMath` (`11%`) + `FinePDFs-Edu` (`3%`) + `SmolTalk2` (`3%`) + `Dolci-Think-Olmo-Hybrid` (`3%`) + `SWE-rebench` (`2%`) + `LLMail-Inject` (`2%`)。
   - **高纯中文底座与对话思维链（`41%`）**：`Ultra-FineWeb zh` (`18%`) + `Fineweb-Edu-Chinese-V3` 10 万本出版图书 (`13%`) + `Chinese-DeepSeek-R1-Distill-110k` (`6%`) + `BAAI/Infinity-Instruct` 中文对话指令 (`4%`)；总真实锚点 **`80%`（严守 `>=79%` 防坍缩红线）**，全源 `2025–2026 Tier-S`。
2. **双正交分类矩阵（消除「4 领域 vs 7+1 能力」口径歧义，已固化于 `get_unified_domain_capability_stage_matrix()`）**：
   - **轴 A：4 大内容题材域（学什么题材）**：① 软件工程与代码、② 数学与形式逻辑、③ STEM 出版教材与因果科学、④ 中英双语自然对话。
   - **轴 B：7+1 认知工程能力（练什么本领）× 6 生命周期阶段**：
     - `Stage 0`（词表/字节流）→ `Stage 0.5`（活体 Rule 110 / 2D NCA / Dyck / 长程检索 PPT，练能力 1）→ `Stage 1`（`60:40` 预训练，练能力 1/2/3）→ `Stage 2`（16K 退火+GitHub Diff+10% Alignment Pretraining，练能力 2/4/6/7）→ `Stage 3`（克制 SFT 学中英说话与工具调用，练能力 3/5/6/7）→ `Stage 4 & Runtime`（RLVR 自博弈 + CPU 2-bit TCQ Engram 承接第 8 域冷百科/私有事实，严禁喂 GPU）。
3. **如何找到并造出全世界最牛逼的数据（CTO 5 步实战寻源法）**：
   - **① 打擂台不看宣传**：盯紧 [OpenDataArena (arXiv:2512.14051)](https://arxiv.org/abs/2512.14051) 客观数据集价值榜与 [CCI4.0 (arXiv:2506.07463)](https://arxiv.org/abs/2506.07463) 35TB 中英双语底座，只挑消融实证能涨分的源头；
   - **② 样本级智能编排**：用 [DataOrchestra (arXiv:2607.24717)](https://arxiv.org/abs/2607.24717) 替代一刀切正则过滤（逐条决定丢弃/保留/改写），用 [DataChef (arXiv:2602.11089)](https://arxiv.org/abs/2602.11089) 在小代理模型上自动搜索最优配方；
   - **③ 抓「不可伪造过程轨迹」**：优先抓取带编译器/单测/Git Commit Diff/SymPy/Lean 4 客观验真闭环的真实过程，绝不喂无验证器的大模型口水合成；
   - **④ 教材级块改写 + 真实锚点锁死**：参考 `Fineweb-Edu-Chinese-V3` 与 `Nemotron-CC-v2` 对高价值 PDF/图书做块级重构，同时死锁 `>=80%` 真实人类锚点防递归坍缩；
   - **⑤ 2026 N-Gram 避坑**：外置 N-Gram 记忆表在首轮复读边界（`epoch >= 1.0`）必须冻结学习率防 [Forking 过拟合 (arXiv:2610.00394)](https://arxiv.org/abs/2610.00394)，并用 [2-bit TCQ (arXiv:2610.21810)](https://arxiv.org/abs/2610.21810) 将 CPU 内存压降 8×。


## 十、v10.32 2025–2026 工业级数据工程全栈管线与 5 大反直觉避坑定律（已实装代码）

1. **「去噪胜于淘金」原则（[arXiv:2510.00866](https://arxiv.org/abs/2510.00866)）**：质量分类器的最大收益在于剔除低熵垃圾/模板水文，而非过度严苛筛掉长尾高信息文本。已在 `run_data_orchestra_curation_pipeline()` 实装 zlib 压缩比异常过滤（`<0.22` 或 `>0.98`）。
2. **多粒度语义去重（[SemHash-LLM arXiv:2607.01601](https://arxiv.org/abs/2607.01601)、[H3D arXiv:2607.08382](https://arxiv.org/abs/2607.08382)）**：实装 64-bit Token-Shingle SimHash（海明距离 `<=3` 剔除近重复），消除复读过拟合与验证集泄露。
3. **样本级三分流 + 超级文档拼接（[DataOrchestra arXiv:2607.24717](https://arxiv.org/abs/2607.24717)、[Synthetic Megadocs arXiv:2603.18534](https://arxiv.org/abs/2603.18534)）**：代码/数学/出版教材原样保留（`keep_untouched`），含噪网页正文送入块级改写并拼接为 **Megadoc**，数据效率从 `1.48×` 跃升至 **`1.80×`**。
4. **别把最好的数据喂在学习率衰减死尾巴里（[arXiv:2511.18903](https://arxiv.org/abs/2511.18903)）**：已在 `check_curriculum_lr_decay_window()` 强制拦截 `lr_ratio < 0.20` 时才注入顶级退火语料的浪费行为。
5. **数据归因替代盲目配比（[MAGIC arXiv:2504.16430](https://arxiv.org/abs/2504.16430)、[DRIFT arXiv:2606.18307](https://arxiv.org/abs/2606.18307)、[Explaining Data Mixing Laws arXiv:2606.08167](https://arxiv.org/abs/2606.08167)）**：先在代理小模型上跑影响力归因与 `MIR` 掩码输入正则化（[arXiv:2606.06888](https://arxiv.org/abs/2606.06888)），再锁定最终多域配比。


## 十一、v10.32 Token 级精准选数、组级影响归因、去污染与本地流式 `.bin` 打包管线

1. **Token 级超额损失选数（[Rho-1 arXiv:2404.07965](https://arxiv.org/abs/2404.07965)）**：哪怕清洗过的语料里依然混杂大量低信息连接词与格式噪声；已在 `rho1_excess_loss_token_mask()` 实装 `student_loss - ref_loss` 超额损失前 $k\%$ 掩码，数学/代码预训练效率提升 `5×–10×`。
2. **组级关系影响选数（[Group-MATES arXiv:2502.14709](https://arxiv.org/abs/2502.14709)）**：超越单样本质量打分，按聚类关系权重选组，在 DCLM 上再省 `1.75×` 词元预算。
3. **防基准污染与字节效率闭环（[CoDeC arXiv:2510.27055](https://arxiv.org/abs/2510.27055)、[arXiv:2605.19999](https://arxiv.org/abs/2605.19999)、[The Efficiency Gap in Byte Modeling arXiv:2605.12928](https://arxiv.org/abs/2605.12928)）**：在 `build_local_clean_byte_corpus()` 实装端到端本地流式 JSONL 清洗器——单遍完成压缩比去噪、64-bit SimHash 去重、评测基准 n-gram 去污染、中文字符占比审计与 Stage 0.5 自动机样本混编，直接输出可训练 UTF-8 `.bin` 字节流。
