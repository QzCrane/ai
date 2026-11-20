# Sovereign AI — Route B Pure AI Brain & Deterministic Agent Harness (v1.0)

本仓库包含从第一性原理出发的自研主权 AI 全栈架构研究与可执行代码，严格执行**「纯 AI 大脑本体」与「确定性外骨骼 / Agent Harness」双轨解耦**，零历史冗余版本。

## 核心架构主线（Route B：时空解耦与递归涌现）

- **紧致共享循环核（时间换空间）**：三明治拓扑 `Prelude (2层) -> 中层异步 CPU 锁页 Engram -> 共享 3:1 循环核 (4层 × 8轮 = 32层) -> Coda (2层)`，有效深度 **36 层**，内置单标量输入注入、`Parcae` 有界搬运、`1/r` 循环残差缩放、`ARLoop` $W=2$ 滑窗深度路由、`FPRM` 相对不动点早停与 $O(1)$ 截断隐式反传。
- **3:1 双工工作记忆**：75% `OscillatoryDualDeltaNet`（DualDeltaNet + Mamba-3 复相旋转）+ 25% `HoPEHLAMELTAttention`（原生 PyTorch SDPA + MELT 门控共享 HLA 潜 KV + HoPE + Qwen 无 Sink 门控）。
- **0 显存可训练 CPU Engram**：通过 `_CPUSparseEngramFn` 仅将命中槽位梯度回传 CPU 原地稀疏更新，GPU 显存严格 `0.0 MB`。
- **RTX 3050 4GB 实机可执行**：训练期原生 `BF16` AMP + ICLR 2026 `Polar Express Muon`（比 AdamW 省一半优化器显存）。

## 仓库目录结构

```text
ai/
├── README.md
├── SYSTEM_CONSTITUTION.md
├── requirements.txt
├── docs/whitepapers/
│   ├── sovereign_ai_brain_whitepaper.md          # 纯 AI 大脑白皮书（Route B 冻结版）
│   └── deterministic_agent_harness_whitepaper.md # 确定性外骨骼 / Agent Harness 白皮书（零遗漏完整版）
├── src/
│   ├── brain/                                    # 纯 AI 大脑 PyTorch/CUDA 与 NumPy 核心实现
│   └── harness/                                  # 解耦确定性外骨骼运行时与组件
├── benchmarks/
│   ├── suites/                                   # 机制与压力验证套件
│   └── reports/                                  # 已通过的 JSON 实测基准报告
└── scripts/
    ├── push_to_github.sh                         # Linux / WSL2 一键推送至 QzCrane/ai
    └── push_to_github.ps1                        # Windows PowerShell 一键推送至 QzCrane/ai
```

## 快速运行

```bash
pip install -r requirements.txt
python3 src/brain/non_transformer_backbone.py
python3 src/brain/sovereign_brain_pytorch.py --config rtx3050_4gb_65m --steps 20
python3 benchmarks/suites/master_18_problems_harness.py
```
