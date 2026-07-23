# AAO：战略锚定型动态资产配置

AAO 是一个可复现的多资产 PPO 研究工程。仓库随附经过 SHA-256 校验的冻结
美国 ETF 数据，clone 后无需在线下载行情，即可在 GPU 或 CPU 环境中完成：

```text
冻结 OHLCV
  → 因果特征与仅训练期 scaler
  → PPO 训练
  → train / validation / test 确定性回放
  → 战略锚点基准对照
  → 单 seed 报告与多 seed 汇总
```

当前资产池为 `SPY, IWM, IEF, TLT, LQD, HYG, GLD, DBC, VNQ, BIL`，
周频决策。冻结数据公共日期范围为 2008-01-02 至 2026-07-22。

## 最快方式：GPU Docker

前提：已安装 Docker、NVIDIA 驱动和 NVIDIA Container Toolkit。

```bash
git clone https://github.com/X-sherlock/AAO.git
cd AAO
docker build -f docker/Dockerfile.gpu -t aao-gpu .
docker run --rm --gpus all -v "$PWD/reports:/workspace/reports" aao-gpu
```

PowerShell 的运行命令为：

```powershell
docker run --rm --gpus all `
  -v "${PWD}/reports:/workspace/reports" `
  aao-gpu
```

容器默认执行固定切分、5 个随机种子、每个 100,000 个 PPO 环境步的正式矩阵。
先做快速 GPU 预检可执行：

```bash
docker run --rm --gpus all -v "$PWD/reports:/workspace/reports" aao-gpu \
  --config configs/experiments/ppo_real_us_multi_asset.yaml \
  --device cuda --seeds 42 --total-steps 512 \
  --output-dir reports/gpu_smoke
```

## 原生 Python GPU 环境

Python 要求为 3.10 或更高版本。先按当前机器的 NVIDIA 驱动与 CUDA 条件安装
CUDA 版 PyTorch，然后安装本项目：

```bash
git clone https://github.com/X-sherlock/AAO.git
cd AAO
python -m venv .venv
python -m pip install -e ".[train,report,dev]"
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
python experiments/run_experiment_matrix.py \
  --config configs/experiments/ppo_real_us_multi_asset.yaml \
  --device cuda
```

如果 PyTorch 自检显示 `False`，不要开始训练；应先改用与本机兼容的 CUDA 版
PyTorch，或直接使用上面的 GPU 容器。

中断后从已完成的 seed 继续：

```bash
python experiments/run_experiment_matrix.py \
  --config configs/experiments/ppo_real_us_multi_asset.yaml \
  --device cuda --resume
```

## 输出

默认正式输出目录为：

```text
reports/real_us_multi_asset_v1/fixed_v1/ppo/
├── matrix_result.json
├── seed_42/
│   ├── model_state_dict.pt
│   ├── training_result.json
│   └── evaluation_timeseries.json
└── seed_43/ ...
```

`training_result.json` 同时包含 train、validation、test 的 PPO 和战略锚点
指标。`evaluation_timeseries.json` 保留逐期权重、收益、成本、换手、财富与
回撤，便于复核。`matrix_result.json` 汇总多个 seed 的均值、标准差和范围。

详细口径见 [实验与验收协议](docs/experiment_protocol.md)。

## 本地测试与 CPU smoke

```bash
python -m pip install -e ".[train,dev]"
python -m pytest
python experiments/train_ppo.py \
  --config configs/experiments/ppo_real_us_multi_asset_smoke.yaml \
  --device cpu \
  --output-dir reports/cpu_smoke
```

训练入口默认拒绝覆盖已有模型和结果。确需覆盖时必须显式传入
`--overwrite`；更推荐指定新的输出目录。

## 重新获取或导入行情

正式训练不会联网。只有创建新的冻结数据版本时才需要下载依赖：

```bash
python -m pip install -e ".[download,report]"
python -m asset_allocation.data_download.real_ohlcv \
  --config configs/data/real_us_multi_asset.yaml
python experiments/build_real_dataset.py \
  --config configs/data/real_us_multi_asset.yaml
```

冻结目录不可覆盖；变更日期、Provider 或资产池时必须使用新的
`dataset.id`。

## 研究边界

本项目用于研究和工程验证，不构成投资建议。短步数、单随机种子或单一切分
结果不得解释为模型已在真实市场中证明有效。
