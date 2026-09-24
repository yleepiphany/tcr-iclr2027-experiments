# ICLR 2027 实验与表格简报

更新时间：2026-09-21 21:12 CST

## 一句话结论

Clean LIBERO主表和机制消融已经形成主要结果；当前算力集中在OpenVLA-OFT与RoboTwin正文表，LIBERO-PRO按新的cross-benchmark定位后置，停止扩展无关方法。

## 当前三条主线

| 优先级 | 实验 | 当前进度 | 下一步 |
|---:|---|---|---|
| 1 | OpenVLA-OFT Table 4 | 官方Experts完成5/12正式jobs，即2,500/6,000 episodes | 其余7个job空卡即跑；之后只评估表内可忠实实现的方法 |
| 2 | RoboTwin Table 5 | Experts完成409/540 episodes；5/9 group-repeat jobs完成 | 完成Experts；Model Soups checkpoint和9个正式job manifest已冻结，按OpenVLA之后的优先级启动 |
| 3 | LIBERO-PRO | Experts 87/480 jobs | 归入cross-benchmark；只跑正文最终保留的方法，当前让卡给前两项 |

## 已有正式结果

### Table 1：Clean LIBERO

- 11行已有数字；其中8行是统一正式协议。
- 当前最佳非专家合并结果：FeatCal `70.92±2.16%`。
- RegMean++严格三独立构建仍缺s0-r01四suite。
- TCR-E/TCR-O为历史异协议结果，已加脚注。

![Table 1](table-screenshots/table-01-main_baselines.png)

### Table 3：机制消融

- 8/8行完成。
- Full TCR-E：`69.75%`。
- 最大负向变化来自Action-only calibration：`-14.00pp`。

![Table 3](table-screenshots/table-03-mechanism_ablation.png)

## 正在形成的正文结果

### Table 4：OpenVLA-OFT

官方release Experts的seed=1000已完成：

| Suite | Success |
|---|---:|
| Spatial | 97.8% |
| Object | 98.2% |
| Goal | 97.8% |
| Long | 93.0% |

新增完成：Goal seed=3000为`97.4%`。当前合计5/12正式jobs；其余seed=2000/3000共7条waiter等待真空GPU。

![Table 4](table-screenshots/table-04-cross_backbone.png)

### Table 5：RoboTwin

- Experts：409/540 episodes，231成功、178失败。
- 9个group-repeat jobs：5完成、4运行。
- Precision三重复正式结果：`51.67±4.41%`。
- Receptacle失稳seed已按下一个预注册SHA槽位修订，队列已恢复，不再无限重复失败。
- 三专家Model Soups checkpoint已完成物化：3专家、LoRA输出rank 192；9个正式group-repeat manifests与同seed协议均已冻结。

![Table 5](table-screenshots/table-05-cross_benchmark.png)

![RoboTwin group结果](table-screenshots/table-23-cross_benchmark_groups.png)

### LIBERO-PRO / cross-benchmark扩展

- 当前Experts：87/480 jobs。
- 以后按cross-benchmark表记账，不再作为旧Table 4的附属矩阵。
- 不会默认跑所有融合方法，只执行正文最终保留的行。

![LIBERO-PRO](table-screenshots/table-13-libero_pro_robustness.png)

## 暂停、取消或阻塞

- 真机Table 2：缺硬件。
- Continued training Table 6：已取消。
- RoboCasa365与long-horizon：本地数据缺失。
- OpenVLA RegMean++：缺native activation cache与已验证solver，不能伪造结果。
- OpenVLA TIES development-only实现不进入正式表。
- LIBERO-PRO额外方法：在正文行未冻结前不启动。

## 当前资源策略

- OpenVLA空卡优先级最高。
- RoboTwin已运行的formal jobs继续，不增加quick gate或开发筛选。
- PRO与旧RegMean++ waiter暂停，避免抢OpenVLA空卡。
- 每20分钟巡检一次，只汇报新结果、故障恢复或阶段完成。
