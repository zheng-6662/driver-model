# 阶段 2 补充：键盘式人工事件标注播放器 v0.1

生成时间：2026-05-12

## 为什么做

人工填写整张事件表太复杂，因此新增本地键盘标注播放器。用户可以播放原始车辆时间线，用键盘标记事件开始和结束，标签由本地 Python 服务写入 CSV。

## 使用入口

- 本地页面：`http://127.0.0.1:8766/`
- 标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/run_manual_event_keyboard_player.py`

## 默认按键

- 空格：播放/暂停。
- `A`：把当前时间标记为事件开始。
- `S`：把当前时间标记为预测锚点；不按则默认锚点等于开始时间。
- `D`：把当前时间标记为事件结束并保存一行标签。
- 左/右方向键：小步后退/前进；按住 Shift 为大步。
- `N` / `P`：切换下一条/上一条记录。
- `U`：撤销最后一条保存的标签。

## 边界

本工具只读取原始车辆 CSV 和候选事件表，不修改原始 CSV，不训练模型，不读取服务器密码。输出的人工标签需要再经过一致性检查，才能升级为 `manual_verified` 样本清单。
