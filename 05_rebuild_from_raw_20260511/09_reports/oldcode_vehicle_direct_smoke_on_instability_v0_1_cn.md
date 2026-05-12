# 旧深度车辆模型 smoke：全原始失稳高置信样本 v0.1

生成时间：2026-05-12

## 为什么做

用户要求用之前的旧代码测试重新筛出的车辆失稳样本。为了避免直接把旧模型当成正式结论，本次只做一个小规模 smoke run，目标是验证旧深度模型入口能否读取新的失稳样本 manifest、完成训练/验证/测试闭环。

## 使用的旧代码

- 旧入口：`F:/data_set_process/data_process/02_code/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
- 旧模式：`vehicle_direct`
- 新 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split.csv`
- 运行设备：本地 CPU
- 是否使用服务器：否
- 是否读取服务器密码文件：否

## 运行设置

```text
--conditioning-mode vehicle_direct
--teacher-forcing-ratio 0
--event-loss-weight 0
--selection-mode legacy_rmse
--smoke-test
```

smoke-test 内部子集：

```text
train=96
val=32
test=32
epochs=2
batch_size=16
```

## 运行结果

- 旧代码样本构建：成功
- 训练/验证/测试闭环：成功
- 丢弃样本数：0
- 输入维度：12
- 输出维度：未来 400 点，方向盘增量 + 车速增量
- run 目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_SMOKE_INSTABILITY_HIGHCONF_V0_1_20260512_165950`
- best epoch：2
- best val steer RMSE：0.976725
- smoke test steer RMSE：0.400123
- smoke test tail RMSE：0.252727
- smoke test peak time absolute error：0.892857 s

## 怎么解释

这次结果只能说明旧深度车辆模型代码可以在新的失稳样本 manifest 上跑通，不能说明旧模型已经有效，也不能和全量强车辆基线直接比较。原因是：

1. 只用了 smoke 子集，不是全量 906 个可用失稳事件。
2. 只跑了 2 个 epoch，训练远未充分。
3. 使用的是旧 `vehicle_direct` 结构，没有重新设计响应类型、幅值、错侧、尾段和困难样本目标。
4. 未使用固定预测图、坏样本图和多 seed 稳定性验证。

## 下一步建议

如果继续沿旧代码测试，应先做两个动作：

1. 用同一份 manifest 跑一个全量但仍只含车辆输入的旧 `vehicle_direct` 基线，并输出固定预测图和坏样本图。
2. 把旧深度模型结果和本次无学习/车辆 ridge 诊断表放在一起比较，尤其看错侧率、严重幅值不足率、尾段误差和困难样本，而不是只看 RMSE。

在这些完成前，不能用该 smoke run 支持连续风格或生理数据有效。
