# Sequential Online 实验协议(v2)

本文定义 RSI4TSM 的实验协议:如何把 GIFT-Eval 改造为「连续到达的数据集序列」,
让 Sequential Online 成为唯一的正式实验场景,Independent Online 作为控制实验。

撰写日期:2026-11-10(v2)
状态:协议冻结稿。v2 相对 v1 的五项修改见 §1.2。

**论文主线(全文贯穿,协议不得偏离):**

```
Historical Tasks → Experience Accumulation → Knowledge Transfer → Better Future Forecasting
```

---

## 1. 协议总览

### 1.1 角色分配

| 角色 | 内容 |
| --- | --- |
| Problem Setting | Sequential Cross-Dataset Online Forecasting |
| Dataset Source | GIFT-Eval |
| Control Baseline | Independent Online |
| 核心研究问题 | Cross-Dataset Knowledge Transfer 是否真的有效 |

### 1.2 v2 的五项修改

| # | 修改项 | 具体修改 | 目的 |
| --- | --- | --- | --- |
| 1 | Source / Target | 按 Dataset Family 隔离,Source 只用训练区,Target 只用官方测试窗口 | 避免数据泄漏 |
| 2 | Online Extension | **删除** Target 扩展轮;Source 保留训练区扩展轮 | Target 真正 unseen |
| 3 | Online Interaction | 固定 Predict → Feedback → Update 三段 | 保证在线交互正确 |
| 4 | State Management | Sequential 保留全局经验;Independent 在 Dataset 边界重置 | 保证对照公平 |
| 5 | Evaluation | Stage A 为**主要**迁移证据,Stage B 为**辅助**在线适应分析 | 简化评价体系 |

核心不变式(以下三条必须同时成立,否则协议无效):

1. **Target 的任何标签都不参与 Source 阶段的任何计算。**
2. **Sequential 与 Independent 使用完全相同的 Agent、反馈、预算、随机种子与任务顺序,唯一差别是跨 Dataset 经验是否保留。**
3. **Target 阶段的每一轮只使用官方 GIFT-Eval 测试窗口,不扩展、不回退、不借用训练区做计分。**

### 1.3 数据流全景

```
                    ┌─────────────────── Source Phase ───────────────────┐
   GIFT-Eval   ──▶  │ D_1^S  D_2^S  ...  D_K^S     (训练区 + 扩展轮)        │
   (28 families)   │  ↓ 经验沉淀(指纹 + 已核验程序 + 验证误差)              │
                    └───────────────────────┬───────────────────────────────┘
                                            │  跨 Dataset 经验(Memory)
                    ┌───────────────────────▼───────────────────────────────┐
                    │            Target Phase(官方测试窗口,不再扩展)        │
                    │  D_1^T  D_2^T  ...  D_M^T                            │
                    │  Stage A = 每个 Target 的窗口 0                      │
                    │  Stage B = 每个 Target 的窗口 1..W-1                 │
                    └───────────────────────────────────────────────────────┘
```

---

## 2. Source / Target 划分(修改 1)

### 2.1 Dataset Family 的定义

GIFT-Eval 官方 `all_datasets.yaml` 列出 **55 个 short 档任务配置**。
按「去掉频率子目录 + 归一化名称」归并后得到 **28 个 Dataset Family**。

Family 归并规则(与 `run_eval.py` 的 `PRETTY_NAMES` 一致):

| 步骤 | 规则 | 例 |
| --- | --- | --- |
| 1 | 取 `/` 前的主名 | `electricity/15T` → `electricity` |
| 2 | 去掉 `_with_missing` 后缀 | `car_parts_with_missing` → `car_parts` |
| 3 | 大小写归一 + 别名映射 | `saugeenday` → `saugeen`,`LOOP_SEATTLE` → `loop_seattle` |
| 4 | 同一 family 的不同频率视为**同一 family** | `electricity/{15T,H,D,W}` → 1 个 family |

同一 family 的不同频率版本(如 `electricity/D` 与 `electricity/H`)
共享底层序列,只是重采样频率不同。因此 **family 是泄漏隔离的最小单位**:
同一个 family 绝不能同时出现在 Source 和 Target。

### 2.2 28 个 Family 的完整清单(已核验)

| # | Family | Domain | 可用频率配置 | 分配 |
| --- | --- | --- | --- | --- |
| 1 | `m4_yearly` | Econ/Fin | A | Source |
| 2 | `m4_quarterly` | Econ/Fin | Q | Source |
| 3 | `m4_monthly` | Econ/Fin | M | Target |
| 4 | `m4_weekly` | Econ/Fin | W | Source |
| 5 | `m4_daily` | Econ/Fin | D | Target |
| 6 | `m4_hourly` | Econ/Fin | H | Source |
| 7 | `electricity` | Energy | 15T/H/D/W | Source |
| 8 | `solar` | Energy | 10T/H/D/W | Target |
| 9 | `ett1` | Energy | 15T/H/D/W | Source |
| 10 | `ett2` | Energy | 15T/H/D/W | Target |
| 11 | `hospital` | Healthcare | M | Source |
| 12 | `covid_deaths` | Healthcare | D | Target |
| 13 | `us_births` | Healthcare | D/M/W | Source |
| 14 | `saugeen` | Nature | D/M/W | Target |
| 15 | `temperature_rain` | Nature | D | Source |
| 16 | `kdd_cup_2018` | Nature | D/H | Target |
| 17 | `car_parts` | Sales | M | Source |
| 18 | `restaurant` | Sales | D | Target |
| 19 | `hierarchical_sales` | Sales | D/W | Target |
| 20 | `loop_seattle` | Transport | 5T/H/D | Source |
| 21 | `sz_taxi` | Transport | 15T/H | Target |
| 22 | `m_dense` | Transport | D/H | Source |
| 23 | `jena_weather` | Nature | 10T/H/D | Target |
| 24 | `bitbrains_fast_storage` | Web/CloudOps | 5T/H | Source |
| 25 | `bitbrains_rnd` | Web/CloudOps | 5T/H | Target |
| 26 | `bizitobs_application` | Web/CloudOps | 10S | Source |
| 27 | `bizitobs_service` | Web/CloudOps | 10S | Target |
| 28 | `bizitobs_l2c` | Web/CloudOps | 5T/H | Source |

划分结果:**14 个 Source family + 14 个 Target family,严格不相交**。
Domain 覆盖:两个集合各自覆盖全部 7 个 domain。
频率覆盖:Source 含 A/Q/M/W/H/10S/5T/15T,D 频在 Target;
Target 含 M/D/H/10S,W 在 Source——这是刻意的,
因为**跨频率迁移是本研究的核心考察对象**,Source/Target 频率不重合
才能让"迁移是否需要频率匹配"成为一个可证伪的问题。

> 注:`m4` 系列的 6 个配置在 GIFT-Eval 里是 6 个独立 family
> (年/季/月/周/日/小时),不是同一 family 的频率变体。
> 它们底层数据不重叠,但风格高度相似,分配时按 family 严格隔离处理。

### 2.3 Source / Target 的数据使用规则(硬约束)

| | Source Phase | Target Phase |
| --- | --- | --- |
| 可用数据 | **仅训练区**(`training_dataset`) | 仅**官方测试窗口** |
| 窗口构造 | 训练区末尾的滚动窗口(可扩展) | GIFT-Eval 官方 `test_data`,不扩展 |
| 反馈来源 | 扩展轮自己产生的后续真值 | 已完成窗口的官方标签 |
| 是否计分 | 否(只沉淀经验) | 是(主表) |
| Stage A | 不适用 | 各 Target 的窗口 0 |
| Stage B | 不适用 | 各 Target 的窗口 1..W-1 |

**泄漏检查表(实现时逐项断言):**

- [ ] Source 阶段的任何 `training_dataset` 切片都不与任何 Target family 重叠。
- [ ] Target 阶段的计分只读官方 `test_data`。
- [ ] Source 阶段的经验条目中不含任何 Target family 的统计量。
- [ ] Target 阶段的经验条目**只**由该 Target 已完成窗口的反馈产生,
      且写入发生在对应窗口结束后。
- [ ] `actual` 字段永远不进 Agent 的预测请求体(现有 `AgentPredictor` 已保证)。

---

## 3. Online Interaction:Predict → Feedback → Update(修改 3)

### 3.1 GIFT-Eval 原生窗口就是在线流(已实证)

在 `hierarchical_sales/D`(pl=30, windows=7)上实测:

```
win0: ctx_len=1615 label_len=30
win1: ctx_len=1645 label_len=30   ← 正好接上 win0 的 label 末尾
...
win6: ctx_len=1795 label_len=30
```

逐窗口验证 `ctx(w).end == label(w-1).end` 严格成立。

结论:**GIFT-Eval 的测试窗口本身构成一条滚动原点的在线流**。
不需要重新构造数据,只需按窗口顺序把真值交回 Agent。
现有 `evaluation/gift_eval/windows.py` 的 `window_indexes` /
`restore_window_order` 就是现成的窗口重分组原语,直接复用。

### 3.2 三段交互的精确定义

对数据集中的每个在线轮次(数据集, 窗口序号, 序列):

```
┌─ Predict ────────────────────────────────────────────────┐
│ 输入:                                                │
│   ctx   = 该序列到当前预测原点为止的全部历史            │
│   mem   = 检索到的跨 Dataset 经验条目(条件相关时)       │
│   fb    = 上一轮(feedback)的实际值与误差                │
│ 输出: 点预测 + 分位数 + 本轮候选经验条目                │
│ 硬约束: 本轮真值绝不出现在输入中                        │
└──────────────────────────────────────────────────────────┘
                          ↓
┌─ Feedback ──────────────────────────────────────────────┐
│ 预测完成后才生成:                                       │
│   实际值 actual_r = 序列在预测原点之后的 pl 个点        │
│   本轮误差 err_r   = 预测 vs actual_r                   │
│ 硬约束: actual_r 在 Predict 阶段不可见                  │
└──────────────────────────────────────────────────────────┘
                          ↓
┌─ Update ────────────────────────────────────────────────┐
│ 基于 (ctx, 预测, actual_r, err_r):                      │
│   1. 评估本轮选择的程序是否优于备选(留出核验)          │
│   2. 若更优,把该程序作为经验条目写入记忆                │
│   3. 经验条目携带:指纹 / 程序 / 已核验误差 / 来源标注   │
└──────────────────────────────────────────────────────────┘
```

**顺序不可交换。** Feedback 必须在 Predict 之后,
Update 必须在 Feedback 之后。实现上,每个窗口的调用是**串行**的:
一轮跑完并写入反馈后,下一轮才开始。

### 3.3 Agent 接口契约(对现有代码的最小增量)

现有 `/v1/runs` 是单次请求、无状态的。要支持 Sequential Online,
按 v1 方案增加三个字段(不新开端点,保持与 `AgentPredictor` 兼容):

**请求侧新增:**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `memory_context` | `list[dict]` | 检索到的跨 Dataset 经验条目(Independent 条件下恒为空) |
| `feedback` | `dict \| None` | 上一轮的 `{point, q10, q90, actual, err}` |
| `dataset_id` | `str` | 当前数据集标识,用于经验条目的来源标注 |

**响应侧新增:**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `memory_write` | `list[dict]` | 本轮沉淀的候选经验条目 |

**状态侧:**

- Sequential:记忆载体在 Dataset 边界**保留**
- Independent:在 Dataset 边界**重置**

关键约束不变:**`actual` 永远不进预测请求体**。
反馈走 `feedback` 字段,且只包含**已经完成的那一轮**的真值。
实现时用断言守住:请求体出现 `actual` 即报错。

### 3.4 跨 Dataset 经验的内容边界

经验条目只允许包含:

- **指纹**:序列的统计特征向量(长度、周期、季节性强度、趋势斜率、缺失率、偏度等)
- **已核验的预测程序**:季节长度、是否差分、模型选择、上游代码改写
- **验证误差**:该程序在**已揭晓标签**上的实测误差

禁止包含:

- 任何数据集的未来真实值
- 未被反馈流确认的标签区内容
- 具体序列的原始数值拷贝(避免退化成查表)

判定标准:记忆条目在**当前数据集尚未开始**时就已经存在,
因此它不可能包含当前数据集的标签。

现有 `baselines/timeclaw/memory.py` 的 `MemoryBank` 已实现
「指纹 → 程序」的结构与最近邻检索,可作为经验载体的参考实现。

---

## 4. Online Extension 的重新定位(修改 2)

### 4.1 Target:只用官方测试窗口,不扩展

**Target 阶段的轮次集合严格等于 GIFT-Eval 官方 `test_data`**,
一个不多一个不少。Target 绝不从训练区借用轮次。

这带来一个明确后果:**官方窗口只有 1 个的 Target 仍然参与实验**,
它只贡献 Stage A 证据,不贡献 Stage B 证据。
**不强制扩展 Target 的测试轮次**——扩展会污染「Target 真正 unseen」这一性质。

### 4.2 Source:保留训练区扩展轮

Source 阶段需要足够长的轮次流来积累经验。
Source 的扩展轮**从训练区借用**,绝不触碰标签区:

设 `L_train` = `training_dataset` 中序列的长度,`pl` = `prediction_length`。

```
可用扩展轮数 = floor(L_train / pl) - 1
```

实测 4 个本地数据集:

| 数据集 | pl | 官方 windows | 训练区长度 | Source 可用扩展轮数 |
| --- | --- | --- | --- | --- |
| covid_deaths | 30 | 1 | 152 | 4 |
| car_parts_with_missing | 12 | 1 | 27 | 1 |
| m4_weekly | 13 | 1 | 2166 | 165 |
| hierarchical_sales/D | 30 | 7 | 1585 | 51 |

Source 扩展轮**不计分**,只作为反馈信号与经验积累的载体。
Source 的经验来自这些扩展轮上的「预测 → 反馈 → 更新」循环。

### 4.3 两阶段轮次来源对照

| | Source Phase | Target Phase |
| --- | --- | --- |
| 轮次来源 | 训练区末尾滚动窗口(自造) | GIFT-Eval 官方 `test_data` |
| 轮次数 | `floor(L_train/pl) - 1` | 官方 `windows`(1..20) |
| 是否计分 | 否 | 是 |
| 是否计入主表 | 否 | 是 |
| 扩展策略 | 允许 | **禁止** |

---

## 5. State Management(修改 4)

### 5.1 两种条件

| 条件 | 跨 Dataset 经验 | Dataset 内在线学习 | 任务顺序 | 反馈 | 预算 |
| --- | --- | --- | --- | --- | --- |
| **Sequential Online** | **保留**(全局) | 有 | 相同 | 相同 | 相同 |
| **Independent Online** | **重置**(边界清空) | 有 | 相同 | 相同 | 相同 |

Independent Online **不是另一个算法**,而是**同一个 Agent 在切换数据集时清空跨 Dataset 经验**。
除了这一处,两条链路的所有条件逐一相同。

### 5.2 实现方式

经验载体是一个 `MemoryStore` 对象:

```
Sequential:  一个 store 实例贯穿整个 Target 阶段
Independent:  每个 Target 数据集开始前,新建空 store
```

`baselines/timeclaw/memory.py` 的 `MemoryBank` 已是这个结构:
Sequential 时同一实例跨数据集存活,Independent 时在数据集边界新建。

Agent 侧同理——`agent/hooks.py` 的 `Policy` 与
`agent/app.py` 的 `thread_id` 机制提供了实现位置:
Sequential 用**同一个 `thread_id`** 贯穿整个阶段,
Independent 在**每个 Target 边界换新 `thread_id`**。

### 5.3 公平性检查表

实现后逐项核对:

- [ ] 两条链路的 Agent 版本、prompt、模型、采样参数完全一致。
- [ ] 两条链路的 `max_react_steps` / `max_tool_calls` / `max_context` 完全一致。
- [ ] 两条链路使用同一条任务顺序、同一随机种子、同一抽样策略与 limit。
- [ ] 两条链路的指标聚合代码完全一致(同一份 `build_metrics()` 与季节性口径)。
- [ ] Independent 的 store 确实在每个 Target 边界清空(加断言)。
- [ ] Independent 仍保留 Dataset 内在线学习(不清空 Dataset 内状态)。

### 5.4 Stage A 的对照为什么干净

这是本协议最关键的设计。

Target 数据集 `D_i` 的**窗口 0** 上:

- Sequential 携带:Source 全部经验 + 已完成的 `D_1..D_{i-1}` 的经验
- Independent 携带:**空**(边界刚重置)

此刻**两个条件都没有 `D_i` 的任何反馈**,所以 Dataset 内在线学习对两者的贡献完全相同(都是零)。
Stage A 的差值因此是**纯粹的跨 Dataset 迁移效应**,不掺杂任何在线适应成分。

这正是「验证历史经验的迁移价值」所需要的证据形态。

---

## 6. Evaluation(修改 5)

### 6.1 Stage A:主要迁移证据

**定义**:Target 数据集 `D_i` 的**窗口 0**——进入新数据集后的第一次预测,
此刻尚未获得该数据集的任何反馈。

**取数**:每个 Target 的窗口 0 单独评估,得到该 Target 的 Stage A 指标。

**主比较**:Stage A 上 Sequential vs Independent 的逐 Target 差值。

**回答的问题**:进入全新数据集时,历史经验是否让 Agent 的第一次预测更好。

### 6.2 Stage B:辅助在线适应分析

**定义**:同一 Target 内,拿到反馈之后继续推进的窗口 1..W-1。

**取数**:每个 Target 内按窗口序号分别评估;再跨 Target 按窗口序号取平均,
得到一条「误差 vs 窗口序号」曲线。

**回答的问题**:反馈驱动的持续适应能力如何,以及跨 Dataset 经验是否还有额外贡献。

**Stage B 是辅助的**,原因有三:

1. 官方窗口数差异大(1..20),多数 Target 只有 1 个窗口,Stage B 样本稀疏。
2. Stage B 上 Sequential 与 Independent **都**有 Dataset 内学习,
   差值的解释力弱于 Stage A。
3. Stage A 是论文主张的直接检验,Stage B 是佐证。

### 6.3 官方指标与归一化

沿用 `build_metrics()` 的 11 项官方指标,保持与 GIFT-Eval 榜单可比。
主表以 **MASE[0.5]** 与 **mean_weighted_sum_quantile_loss** 为准。

**MASE 季节性口径**必须用 `get_seasonality(dataset.freq)`,与官方一致。
实测口径:A→1, Q→4, M→12, W→1, D→1, H→24, 10S→360, S→3600。
注意仓库 `agent/workers.py` 的 `season_of`(D→7)与 GluonTS(D→1)**不一致**,
主表以 GluonTS 口径为准;Agent 内部自评分用 `season_of` 属实现细节,论文脚注说明。

**跨数据集聚合必须归一化**:不同数据集 MASE 数量级差异巨大
(`covid_deaths` 在几十量级,`car_parts` 在 1.7 量级)。采用 GIFT-Eval 榜单口径:

1. 每个任务配置先算出一条本地 `Seasonal_Naive` 参考行
   (同一 `get_seasonality` 季节长度、同一切分、同一采样)。
2. 报告 `rel_MASE = MASE_model / MASE_SeasonalNaive` 与**平均排名**。
3. 表格同时给出原始指标与归一化指标,避免归一化掩盖绝对性能。

### 6.4 分阶段取数的实现

GluonTS 的 `evaluate_model` 一次性聚合整个 `test_data`,要分阶段必须**按窗口分别评估**。

可行路径(已核验 `evaluate_forecasts` 接受任意 `test_data` 迭代器):

- 把 `dataset.test_data` 按窗口切分,每个窗口子集是一个只含该窗口序列的
  `TestData`-like 对象(参照 `run_eval.py` 里 `LimitedTestData` 的写法,
  只需提供 `.input` / `.label` / `.__len__`)。
- 对每个窗口子集单独调用 `evaluate_forecasts`。
- 按 (Target, 窗口序号) 聚合出 Stage A / Stage B 两组行。

**采样必须两个条件完全一致**(同样的 `sample-strategy`、同样的 limit、
同样的随机种子),否则 Stage A 的差值没有意义。

---

## 7. 实验矩阵

### 7.1 Experiment 1:Main — Sequential Online

在连续到来的 Target Dataset 上运行 Agent,比较:

| 方法 | 说明 |
| --- | --- |
| Seasonal Naive | 归一化基准 |
| Chronos-2 | 官方协议,零样本参考 |
| TimesFM-3 | 零样本参考 |
| TimeClaw | 已有对照算法,含指纹记忆 |
| TimeAgent | 已有对照算法,无跨数据集记忆 |
| **Proposed Method** | 本文方法(Sequential) |

**回答**:新算法是否通过跨 Dataset 经验迁移提升预测性能。

### 7.2 Experiment 2:Control — Independent Online

同一套 Agent、同一条顺序、同一份预算,仅切换记忆策略:

| | Sequential | Independent |
| --- | --- | --- |
| 保留历史 Dataset 经验 | 是 | 否(边界重置) |
| 持续 Online Feedback | 是 | 是 |

**重点观察两个指标**:

- **Stage A**:进入新 Dataset 的第一次预测 → 验证历史经验的迁移价值
- **Stage B**:获得当前 Dataset 反馈后的表现 → 验证持续适应能力

这是证明 Cross-Dataset Transfer 有效的关键实验。

### 7.3 Experiment 3:Ablation

| Ablation | 验证问题 |
| --- | --- |
| Without Memory | 历史经验是否有价值? |
| Without Transferability Selection | 经验筛选是否有价值? |
| Without Feedback Update | 在线反馈是否有价值? |
| Recent Memory Only | 长期经验是否必要? |

实现约定:每个消融是一个**配置开关**,不改变其他任何条件。
「Without Memory」与 Experiment 2 的 Independent 是不同层面的东西——
前者关掉新算法内部的记忆模块,后者只重置跨 Dataset 边界。

### 7.4 Experiment 4:Transfer Analysis

1. **经验积累曲线**:随历史 Dataset 数量增加,Target 初始预测(Stage A)是否持续改善?
2. **正迁移识别**:哪些 Dataset 之间存在正迁移?
3. **负迁移检测**:是否出现 Negative Transfer?
4. **顺序敏感性**:不同任务顺序是否影响结果?

**顺序要求:至少 5 种任务顺序**,含 1 条人工交错序 + 4 条固定随机种子随机序。

### 7.5 任务顺序

顺序定义为 Source family 序列与 Target family 序列的**交错**。

- **人工交错序**:Target family 按 domain 交错插入 Source 之间,
  保证相邻数据集不同 domain、不同频率,避免迁移效果与分布相似度混淆。
- **随机序**:4 条,固定随机种子(如 20261110..20261113),对
  (Source ∪ Target) 全集做随机排列后,再按 Source/Target 角色切分。

**顺序敏感性如何进入结论:**

- Experiment 1/2/3 的主表**按顺序分别报告**,不做跨顺序平均
  (不同顺序下 Target 的到达时刻不同,平均会掩盖「经验积累效应」)。
- Experiment 4 的顺序敏感性分析,报告 5 条顺序下 Stage A 差值的均值与标准差。
- 随机序的种子必须写进结果文件,保证可复现。

**Target family 顺序**对 Stage A 尤其关键:Target 越晚到达,
Source 经验越充分,Stage A 的迁移效应应当越强。
若经验积累无效,Stage A 差值不会随 Target 位置单调改善——
这本身就是一个可证伪的预测,写进 Experiment 4。

### 7.6 统计检验

- 14 个 Target family 是**配对样本**(同顺序、同采样、同种子),
  主检验用 Wilcoxon signed-rank;辅以 paired bootstrap CI。
- 顺序作为重复测量因素,报告跨顺序的均值±标准差。
- 多重比较(5 顺序 × 2 指标)用 Holm 校正。

---

## 8. Transfer Analysis 细则

### 8.1 经验条目的溯源字段

每次写入记忆条目时记录:

```
{ fingerprint, routine, verified_loss,
  source_dataset_family, source_window, timestamp }
```

`source_dataset_family` 是关键:它让每条经验都能回答
「我从哪个 family 学到的,在当前 Target 上有没有用」。

### 8.2 三张分析表

**(a) 经验采纳率 × 来源距离**

按来源与当前的 domain / frequency 是否匹配,统计经验条目被采纳的比例。
若同频经验采纳率显著更高,说明迁移是「频率特定的」。

**(b) 迁移收益矩阵**

行 = 来源 family,列 = Target family,
值 = 在该 Target 上采用该来源的经验相对不采用的误差变化。
期望看到明显的对角线与邻近结构。

**(c) 负迁移检测**

Negative Transfer = 有记忆时误差**变差**。判定:

1. 逐轮比较 Sequential 与 Independent 在同一 (Target, 窗口) 上的误差。
2. 统计变差轮次占比;显著高于随机基线即判定存在负迁移。
3. 对变差轮次回溯当时采纳了哪些经验条目,定位负迁移来源。

需要一个**归零基线**:把经验条目的采纳概率置零(等价于不写回记忆),
用来区分「记忆有害」与「本来就更难」。

### 8.3 最有说服力的一张图

> 横轴 = Target 在序列中的位置(即已积累的历史 Dataset 数量),
> 纵轴 = Stage A 的归一化 MASE,两条线分别对应 Sequential 与 Independent。

Sequential 的曲线应系统性低于 Independent,并随经验积累收敛。
这张图直接把「持续改进」讲清楚,是全文机制解释的核心图。

---

## 9. 落地顺序

| 阶段 | 内容 | 依赖 | 产出 |
| --- | --- | --- | --- |
| P0 | 下载全部 28 个 family(镜像可得性已核验) | 网络 | 55 个任务配置可加载 |
| P1 | Source/Target 划分器 + Family 归并 | P0 | `evaluation/online/split.py` |
| P2 | 窗口调度器:Source 扩展轮 + Target 官方轮次 | P1 | `evaluation/online/rounds.py` |
| P3 | 按窗口分别评估(复用 `evaluate_forecasts`) | P2 | 每 (数据集, 窗口) 指标行 |
| P4 | 状态管理:Sequential / Independent 开关 | P3 | 两种条件的主表 |
| P5 | 顺序调度:5 条顺序跑通 | P4 | 顺序级结果文件 |
| P6 | 分析:配对检验、迁移矩阵、负迁移检测 | P5 | 论文表格与图 |
| P7 | Proposed Method 接入 + Ablation | P6 | 完整实验 |
| P8 | LLM 模式接入(当前被配额阻塞) | P7 | ReAct 下的完整实验 |

**P1–P6 全部可用固定流水线和 `baselines/` 里的对照算法跑通,不依赖 LLM。**
LLM 配额当前返回 403(DashScope Forbidden),但协议正确性可以先在无 LLM 条件下验证。

---

## 10. 已知风险

| 风险 | 说明 | 应对 |
| --- | --- | --- |
| Target 窗口数少 | 多数短序列 Target 只有 1 个官方窗口 | 不扩展 Target;只贡献 Stage A;Stage B 样本量在论文中说明 |
| Source 轮次不足 | `car_parts` 训练区仅 27 点,扩展轮 1 轮 | 换 family 或接受经验少;不强行构造 |
| 上下文截断 | Agent 策略 `max_series_length=2048`,`context_tail=512` | 结果表记录 `n_context_truncated`,作为协变量 |
| 九档分位近似 | Agent 只输出 point/q10/q90,其余档位线性插值 | 影响 WQL 与 MSIS;论文注明,或让适配器直接输出九档 |
| horizon 上限 64 | 只覆盖 short 档(最长 60) | 主表锁定 short 档;medium/long 需放宽策略 |
| LLM 配额 | 当前 403 | 先跑无 LLM 通路(P1–P6) |
| 顺序敏感性 | 单条顺序结论不稳 | 5 条顺序,主表逐序报告,分析报告均值±标准差 |
| 指标量级不可比 | MASE 跨数据集差几个数量级 | Seasonal_Naive 归一化 + 平均排名 |
| Family 划分主观 | 28 个 family 的人工二分 | 划分规则与清单在协议中固定;换划分的结果作为敏感性分析 |

---

## 11. 事实核验记录

本文所有数据性结论均在本地核验:

1. **窗口时序语义** — `hierarchical_sales/D` 实测
   `ctx(w).end == label(w-1).end` 逐窗口成立。
2. **prediction_length 映射** — 28 条 `dataset_properties.json` 逐条计算
   `norm_freq_str(to_offset(freq).name)` 与官方 map 匹配,28/28 全部可解析。
3. **官方任务数** — `all_datasets.yaml` 含 55 个 `term: short` 配置。
4. **Family 归并** — 按 §2.1 规则归并得 28 个 family,§2.2 清单已逐一核对。
5. **镜像可得性** — `Salesforce/GiftEval` 在 `hf-mirror.com` 上 28 个 family
   全部可得,频率子目录结构与 §2.2 一致。
6. **季节性口径** — `get_seasonality` 实测:
   A→1, Q→4, M→12, W→1, D→1, H→24, 10S→360, S→3600。
7. **轮次容量** — 实测 4 个本地数据集的训练区长度与官方 `windows`,
   得出 §4.2 表格。
8. **LLM 状态** — DashScope `chat/completions` 返回 HTTP 403 Forbidden。
9. **代码接口** — `agent_predictor.py`、`windows.py`、`run_eval.py`、
   `baselines/timeclaw/memory.py` 的实际签名与行为。
