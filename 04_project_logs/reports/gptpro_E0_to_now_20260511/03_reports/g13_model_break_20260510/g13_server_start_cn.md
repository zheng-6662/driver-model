# G13 服务器启动记录

更新时间：2026-05-10

## 前置恢复

G13 启动前已经先恢复旧脚本和关键旧版本检查点：

- 旧实验脚本已从服务器恢复到本地；
- E2、E4、E5A、E6、E7C、E10C、E11A 的关键检查点已重新补跑或从服务器拉回；
- 详情见：
  `F:/data_set_process/data_process/04_project_logs/reports/restore_checkpoint_audit_20260510/restore_status_cn.md`

## 服务器信息

- 连接命令：`ssh -p 24309 root@connect.westd.seetacloud.com`
- 密码不写入日志、报告或代码。
- 远程项目路径：`/root/autodl-tmp/data_process`
- Python：`/root/miniconda3/bin/python`
- GPU：`NVIDIA GeForce RTX 4080`，约 32GB 显存。

## G13 已同步代码

已同步到服务器的关键文件：

- `event_conditioned_baseline_model.py`
- `conditioned_trajectory_head.py`
- `run_event_conditioned_trajectory_baseline.py`
- `prediction_plotting.py`
- `run_g13_breakthrough_candidates.py`

服务器端语法检查已通过。

## 烟雾测试

已通过两个小样本测试：

- G13C：肌电 + 响应类型辅助学习 + 响应类型影响轨迹预测；
- G13H：脑电教师 + 肌电学生 + 响应类型辅助学习。

说明：

- 响应类型模块能训练；
- G13 预测图能生成；
- 恢复出的 E4 教师检查点能被蒸馏路线调用。

## 正式队列

已启动 G13 seed2026 正式筛选队列：

- 队列脚本：
  `/root/autodl-tmp/data_process/04_project_logs/reports/g13_model_breakthrough_20260510/run_g13_seed2026_queue.sh`
- 主日志：
  `/root/autodl-tmp/data_process/04_project_logs/reports/g13_model_breakthrough_20260510/g13_seed2026_queue_master.log`
- 子任务日志：
  `/root/autodl-tmp/data_process/04_project_logs/reports/g13_model_breakthrough_20260510/full_seed2026_logs`
- 队列进程号：`6624`
- 并行数：`5`

第一批已启动：

- G13A seed2026；
- G13B seed2026；
- G13C seed2026；
- G13F seed2026；
- G13H seed2026。

等待空位后自动启动：

- G13I seed2026。

## 当前原则

- 本轮是 seed2026 完整筛选，训练完整 40 epoch；
- 不再使用小样本或提前停止作为正式结果；
- 每个版本都要保存检查点、指标、预测图和中文可读记录；
- seed2026 结果出来后，再决定哪些版本补 2027/2028。
