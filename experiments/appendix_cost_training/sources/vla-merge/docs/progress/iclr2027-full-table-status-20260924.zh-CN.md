# ICLR 2027 论文逐表进度与剩余评测

快照：2026-09-24 15:29 CST。论文表号按当前 `paper/iclr2027/root.aux`；只有模型/seed/reset/任务身份与完整回执一致的结果计为正式完成。

## 正文 Table 1--6

| 表 | 已完成的正文数据 | 剩余与当前执行 |
|---|---|---|
| Table 1：π0.5 clean LIBERO | 10行×5格均已填。Average：Experts 96.42±0.76、Soups 44.83±1.81、TA 46.00±0.75、TIES 44.25±1.30、WUDI 46.00±0.43、KnOTS 47.50±2.63、RegMean 53.67±0.63、RegMean++ 58.17±1.04、FeatCal 68.33±0.76、TCR 77.83±2.52。 | 当前无待补正式格。新RegMean α0.3结果53.67取代旧9/22谱平滑57.67；旧进度文档已过时。标准差来自三个评测repeat，并非所有基线都三次独立融合。 |
| Table 2：real robot | Experts、Soups、TIES、RegMean++、TCR五行×四格均空。 | 无真实机器人正式回执；需硬件/任务试验，ETA未定。 |
| Table 3：机制消融 | Full 77.83±2.52、无expert rollout 73.00±1.52、无中间生成输入 73.17±2.31、无merged-prefix 64.50±1.52；动作指标已填。 | 表格已闭环。后两消融使用另一套matched Full，不可直接从主表Full相减当配对效应。 |
| Table 4：cross-model LIBERO | π0.5列6/6格已从Table1复用。 | OpenVLA-OFT列0/6、Fast-WAM列0/6正式可填格。OFT官方Experts12/12、Soup11/12采用不同的6000集reset协议，不能直接填匹配的3×400表。Fast-WAM四专家指定权重已验收，正式Experts仍0/12 job、0/1200集；Soup开发0/40，TCR开发0/40，另外三法无已验收融合权重/正式回执。 |
| Table 5：cross-benchmark | RoboTwin Experts 56.85±0.85、Soups 12.04±1.40已正式填。PRO Experts/TCR标`n.r.`并保留部分档案。 | RoboTwin TIES 0/9完成、8/9已启动；RegMean++、FeatCal、TCR列仍空。PRO保留Soups 18/480、TIES 4/480；RegMean++与FeatCal各0/480、旧checkpoint selection须重绑。 |
| Table 6：continued training | Base、single expert、Soups、RegMean++、FeatCal、TCR六行×四格均空。 | 尚无可填的正式继续训练结果，需单独确定训练/评测预算，ETA未定。 |

## 附录 Table 7--16

| 表 | 状态 |
|---|---|
| T7 architecture | 六行结构参数齐全，属于结构清单。 |
| T8 cache sequence controls | 六行成功率均空。 |
| T9 calibration budget controls | 四行成功率均空；部分仅有目标行预算。 |
| T10 full recipe controls | 五行成功率均空。 |
| T11 action agreement | 九行success复用T1；RegMean的MSE/cos两格仍空。 |
| T12 three-level suite results | Full与三消融的4×5成功率已填；刷新池消融的比较基线另有matched Full。 |
| T13 LIBERO-PRO robustness | Experts/TCR按用户取消标`n.r.`；其余四方法×五格均空。 |
| T14 build cost | 九行仅部分设计预算/rows，实际merge耗时与峰值显存仍缺，不能称成本表完成。 |
| T15 continued suites | 六行×四套件全空。 |
| T16 cross-benchmark groups | RoboTwin Experts与Soups的三组+overall已填；PRO Experts/TCR为`n.r.`；其余PRO四方法×五格、RoboTwin其余四行×四格均空。 |

## 已取消但保留的PRO记录

- 10k PRO Experts **361/480**、10k PRO TCR **360/480**有效完整任务，两个attempt去重后保存；不足三次完整重复，不能作为正式整体mean±std。
- 15k PRO Experts **164/480**有效回执，879/1640次成功；与正文要求的10k专家bank不可拼接。
- 1016 `claude-pro-main10k`及1023的15k Experts接力均已按用户取消指令精确SIGTERM停止；停止报告和所有完整/部分数据保留。20分钟巡检已禁止恢复这两行。

## Fast-WAM身份差异与去重

| 指定专家 | 用户给的旧口径 | 仓库旧seed1000筛选summary |
|---|---:|---:|
| Long 9k | 398/468 | 435/500 |
| Spatial 12k | 492/500 | 419/500 |
| Goal 10k | 483/500 | 478/500 |
| Object 10k | 485/500 | 495/500 |

两种旧口径都不是Table4三次procedural正式结果：用户记录文件标明仅按原提供口径登记；旧sweep summary缺checkpoint SHA，reset也与正式bank不同。已完成的Fast-WAM开发诊断是Experts 35/40、等权Soup 0/40；TCR另有0/40开发门槛，均不复跑也不填正式格。四指定权重及normalizer SHA已冻结；正式Experts旧v1等待lane在0/12、0 episode时暂停，因为发现其episode RNG seed未与Table1按任务ordinal 0..9配对。v2脚本与合同已完成、seed/EGL及12 job/1200 reset CPU预检通过；截至15:29尚无GPU smoke和正式episode，只能按v2合同去重重挂。

## 当前队列与资源顺序

1. 完成OpenVLA-OFT官方Soup最后Goal seed3000（11/12），保留已完成11项；其结果单列官方6000集协议。RoboTwin TIES保持原九项队列（8项已启动），按receipt续跑。
2. Fast-WAM Experts只按已预检的v2合同重挂四lane；第一集核对seed、reset hash和EGL后继续12 job/1200集。旧stock seed1000 sweep和80集诊断完全跳过。
3. PRO 10k Soups/TIES从原12/4完整回执加新attempt续跑；当前Soups联合18/480、TIES 4/480，后者包含一条旧SIGTERM中断job需按原job id在新attempt补。PRO Experts/TCR不再续跑。
4. PRO RegMean++、FeatCal各0/480：当前旧selection分别指向Table1旧版checkpoint，先重绑新版模型SHA（RegMean++ `fea99a8c...`，FeatCal `b0da07ed...`）、冻结新manifest，再排正式评测。
5. Fast-WAM Soup使用已物化、绑定四专家SHA的checkpoint，但先通过原生GPU加载与开发门槛；TIES/RegMean++/FeatCal须先构建并验收专属融合checkpoint。Fast-WAM TCR仅在修正版开发门槛通过后才进入正式队列；当前0/40不启动1200集。
6. 再处理continued training、成本与真实机器人等没有正式数据的表；这些工作需独立预算，当前不能给可信完工时间。

容量粗估：PRO四个保留方法余量约300 GPU小时；稳定4张A100约75小时、8张约38小时，另加排队/失败重试。当前Soup单卡早期吞吐约8个10集job/小时，剩余462项约58小时持续GPU时间。Fast-WAM Experts拿到四张合格空卡后约2--4小时，一张卡约7--10小时。两台16卡为共享资源，GPU利用率低但有compute PID或租约时不视为真空。

## 证据根

```text
/mnt/workspace/Wilson/parameter-fusion/vla-merge/paper/iclr2027/tabs
/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/iclr2027-fastwam-experts-formal-20260924
/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/iclr2027-fastwam-experts-formal-v2-20260924
/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/openvla-fastwam-diagnostics-20260923
/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/claude-pro-main10k-20260923/formal-attempt-02
/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/pi05-pro-resume-codex-20260924
/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/iclr2027-table5-20260910/evaluation-queues/robotwin-three-expert-table5-formal-methods-v1
```
