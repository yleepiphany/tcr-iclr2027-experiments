# FIG4-BASELINES-215：增加三个主要融合基线

用户授权：Figure 4 需要增加方法对比，并要求安排执行。此任务独立于 LIBERO-PRO 的 seed 讨论。

## 范围与责任

- 现有本机监督 Codex 为请求接手方，请先在自己的 `local-codex-night-status.md` ACK/CLAIM，再冻结独立计划。当前状态为 **requested / awaiting_ack**，不是 queued/running。
- 主会话不启动另一套同名主管，不改活队列。Claude C319 正在收尾和总交接，不给其新增 GPU 队列；远端/接手 Codex 仅只读共享，跨机接手须精确释放和签收。
- 只用本机允许的 GPU 4–7；现有 PRO、RoboTwin、已启动子集任务正常运行。新任务按安全资源门槛接续，每卡先一个新增 worker，不抢占或终止其他任务。
- 不训练、不新增专家组合、不按结果调参，不恢复 OpenVLA/Fast-WAM。

## 九个目标点

新方法为 **TIES、RegMean++、FeatCal**，与现有 TCR、Model Soups 合计五条曲线。

| ID | 专家 | 正式范围 | 每方法回合 |
| --- | --- | --- | --- |
| spatial-goal | Spatial + Goal | 两套件，各 100 | 200 |
| spatial-object-goal | Spatial + Object + Goal | 三套件，各 100 | 300 |
| spatial-object-goal-long | Spatial + Object + Goal + Long | 四套件，各 100 | 400 |

固定 repeat-01，评测 seed 274001，复用 Goal-first Figure 4 的冻结 selection/reset bank。每方法 9 个套件作业、900 回合；三方法上限 27 个作业、2,700 回合。成功率低也完整交付，不用部分回合替代完整点。
先核验四专家已存在正式结果，完全相同 checkpoint、输入处理、原生执行窗口、selection/reset/seed 的 400 回合可复用，不重复评测。不把主表三重复平均当成本图单点，不挑最好的 seed。

## 实现要求

1. 复用已完成 Goal-first 专家集合和 10k common-base bank；参考 `claude-pi05-expert-count-20260923/plan-v1/plan.json`、`goal-first-build-attempt-03/plan.json`、`goal-first-formal-attempt-04/plan.json` 及 `coordination/2026-09-24/goal-first-*-independent-audit.json`。只读旧文件。
2. TIES 使用正文已固定的全局 task-vector 筛选、符号合并、density/scale，不改成逐层 top-k，不扫描参数。
3. RegMean++ 使用正文独立 stabilized VLA adaptation（一次顺序校准、等专家质量等原设定），不是把两轮 TCR 改名；FeatCal 使用其独立 teacher/student 特征及原方程，不借用 TCR 更新公式。超参数沿用正文已选值。
4. 子集构建必须只含对应专家，回放缓存按 subset 过滤。优先复用 Goal-first 物理身份已核验的 A 缓存和每专家预算；缓存不同或方法需要额外统计时显式记录实际请求/行数，不能虚称完全等预算。实现不兼容先报具体缺口，不静默替换算法。
5. 先 CPU 形状/专家集合/模型身份检查和最小原生加载验证，再正式运行；调试不计入正式分母。每个新模型通过导出与原生动作检查后就接其固定套件评测，不等所有九个模型全部构建。
6. 使用唯一隔离根 `vla-merge-runtime/experiments/fig4-baselines-codex-20260925`，源码置 `vla-merge/experiments/fig4-baselines-codex-20260925`；不得覆盖现有冻结源码、模型、失败证据或各 owner 状态。

优先补 FeatCal、RegMean++ 两条强基线；TIES 可先做 CPU 构建准备。现有正式实验优先级不因本任务被中止，未启动任务可由其原 owner 在安全边界自主排序。

## 回执与论文交付

本机请在下一次低频监督时 ACK `FIG4-BASELINES-215`，列出每方法各 subset 的可复用项/缺项、实现检查、冻结计划路径及排队顺序。有进程才报 running，有完整逐回合/模型/reset 独立核验才报 complete。
返回每套件 successes/episodes、subset 总平均、完整模型与结果 SHA256、复用来源。报告不能只给总百分比。
主会话负责把核验通过的完整三点系列加入 Figure 4；数据未完成前不画占位曲线或伪造误差棒。保留四边框及紧凑矢量样式。论文统一覆盖 `paper-build/latest/root.pdf`，不再生成日期编译目录。
