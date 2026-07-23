# 真实多资产数据字典

## 原始冻结层

`ohlcv.parquet` 使用长表，主键为 `date + asset_code`：

| 字段 | 含义 |
| --- | --- |
| `date` | Provider 返回的美国市场交易日期，不做未来回填 |
| `asset_code` | 配置中的 ETF 代码 |
| `open/high/low/close` | Provider 原始、未复权 OHLC |
| `adjusted_close` | Provider 调整后收盘价；连续收益唯一价格来源 |
| `volume` | Provider 原始成交量；缺失不解释为零 |
| `currency` | 第一版固定为 USD，但仍逐行保存 |
| `source` | 行情来源标识 |

日收益和相邻决策日周收益均为
`adjusted_close[t] / adjusted_close[t-1] - 1`。近似成交额为原始
`close × volume`，只作为流动性代理；它不是精确的可成交金额，也不会把
复权价和原始成交量混合解释为真实现金流。

## 离线市场特征

`features.parquet`、`returns.parquet`、`risk_matrices.npz` 和
`rebalance_calendar.parquet` 只包含不依赖策略路径的数据：

- 当前及历史收益、20/60/120 日波动率与下行波动率；
- 样本/收缩协方差、相关性、平均相关性和最大特征值；
- 成交额、成交量历史分位数、Amihud 代理；
- HYG/IEF 与 LQD/HYG 信用压力代理；
- 各资产回撤；
- 每周最后一个有效交易日及下一持有期收益。

所有特征日期 `t` 的最大依赖日期不晚于 `t`。滚动窗口不足产生的行不会被
填充；处理样本从全部必要资产和必要特征可用的最晚日期开始。

## 在线组合状态

下列变量只由 `PortfolioEnvironment` 根据策略历史生成，冻结市场数据中不
存在伪造列：

- 当前持仓与调仓前漂移权重；
- 组合财富、历史峰值和组合回撤；
- 实际换手和实际交易成本；
- 硬约束投影后的最终权重。

环境时序固定为 `s_t -> w_t -> r_{t+1}`。下一持有期收益只作为环境反馈，
不进入当前状态。
