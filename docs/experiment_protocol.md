# 实验与验收协议

## 数据边界

- 正式入口只读取 `data/processed/real_us_multi_asset_v1/`；
- 每次加载都会验证 `checksums.sha256`，校验失败立即终止；
- scaler 仅使用当前 split 的训练区间拟合；
- PPO 的环境只覆盖训练区间；
- validation 和 test 只进行确定性策略回放，不参与参数更新；
- test 不用于超参数或模型选择。

## 单次训练产物

每个随机种子使用独立目录，包含：

- `model_state_dict.pt`：模型、资产顺序、特征顺序、scaler 和 PPO 配置；
- `training_result.json`：运行环境、训练历史以及三段数据的汇总指标；
- `evaluation_timeseries.json`：逐期收益、成本、换手、权重、财富和回撤。

每段同时评估：

- `ppo`：高斯策略均值对应的确定性动作；
- `strategic_anchor`：回到配置战略锚点的可交易基准。

指标包括年化收益、年化波动率、Sharpe、Sortino、最大回撤、Calmar、
CVaR、年化换手、累计交易成本、平均锚点偏离和硬约束违反次数。
当前真实数据为周频决策，年化周期数固定为 52。

## 多随机种子汇总

`experiments/run_experiment_matrix.py` 为每个随机种子创建独立子目录，并在
`matrix_result.json` 中报告各指标的样本数、均值、标准差、最小值和最大值。
正式配置默认使用 5 个随机种子。

固定 split 适合首次完整验收；扩展窗口和滚动窗口已在冻结数据清单中生成，
后续应为每个 split 重复同一套多随机种子流程，再进行跨窗口汇总。

## 结论边界

单个 seed、单个 fixed split 或短步数 smoke run 只能证明工程链路可运行。
只有完成预先声明的多随机种子、walk-forward、交易成本情景、基准和消融矩阵
后，才可以讨论统计稳定性；任何输出均不构成投资建议。
