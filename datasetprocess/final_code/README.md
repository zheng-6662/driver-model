# final_code

这个目录只保留当前最值得继续维护的一套代码，目标是把你真正还在用、或者最接近最终版本的处理链路和模型代码单独拎出来。

## 目录说明

- `processing/vehicle/`
  - 车辆数据最终预处理代码。
- `processing/physio/`
  - 生理数据预处理和二次重洗代码。
- `processing/eeg/`
  - 脑电预处理、ICA 后处理、时间戳对齐代码。
- `dataset/`
  - 事件级多模态数据集构建代码。
- `model/training/`
  - 当前最终训练代码。
- `model/diagnostics/`
  - 当前最终模型评估代码。

## 当前保留的最终代码

### 1. 车辆预处理

- `processing/vehicle/preprocess_vehicle_v14.py`
  - 当前最像最终版的车辆预处理脚本。
  - 理由:
    - 版本号最高且命名明确。
    - 集成了重采样、LTR 估计、启动段剔除、事件检测、事件分级、可视化和日志输出。
    - 比 `正式车辆数据处理/重洗时间处理finally.py` 更收敛，职责更清楚。

### 2. 生理预处理

- `processing/physio/生理数据处理.py`
  - 原始生理 CSV 到 200Hz 清洗结果的主流程。
- `processing/physio/进一步处理生理数据.py`
  - 对初次处理结果再做有选择的重洗，属于后处理增强步骤。
  - 理由:
    - 目录名已经明确指向 `reclean_v2`。
    - `进一步处理生理数据.py` 直接依赖前一步产物，说明这是同一条最终链路的下一阶段。

### 3. 脑电预处理

- `processing/eeg/脑电数据处理.py`
  - 原始 EEG CSV 到基础预处理结果的主流程。
- `processing/eeg/finally.py`
  - ICA 自动清理和质量报告相关代码集中在这里，虽然文件里有历史注释段，但仍是最后汇总版本。
- `processing/eeg/脑电数据与车辆数据对齐时间戳.py`
  - 把脑电结果和车辆时间线对齐时需要用。

### 4. 事件级数据集构建

- `dataset/build_event_dataset_v2_pad_mask_multipeak.py`
  - 当前更可信的最终数据集构建脚本。
  - 理由:
    - 比 `build_event_dataset_final.py` 更新。
    - 代码是活的，不是大段注释稿。
    - 明确支持 `Pad + Mask + Multi-Peak`，更接近你后面的模型输入方式。

### 5. 模型训练和评估

- `model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - 当前最终训练代码。
  - 理由:
    - 位于 `模型代码/` 而不是历史运行结果目录。
    - 版本号最高，命名带 `fixed`，修改时间也是最新一批。
- `model/diagnostics/future_steer_event_rollpeak_transformer_v5_8_diag_eval.py`
  - 对应 v5.8 模型的诊断评估代码。

## 建议处理顺序

1. 车辆预处理
   - `processing/vehicle/preprocess_vehicle_v14.py`
2. 生理预处理
   - `processing/physio/生理数据处理.py`
   - `processing/physio/进一步处理生理数据.py`
3. 脑电预处理
   - `processing/eeg/脑电数据处理.py`
   - `processing/eeg/finally.py`
   - `processing/eeg/脑电数据与车辆数据对齐时间戳.py`
4. 事件级数据集构建
   - `dataset/build_event_dataset_v2_pad_mask_multipeak.py`
5. 模型训练
   - `model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
6. 模型诊断评估
   - `model/diagnostics/future_steer_event_rollpeak_transformer_v5_8_diag_eval.py`

## 暂时不纳入 final_code 的内容

- 低版本 `v1/v2/v3/v4/v5.0~v5.6` 训练脚本
- `程序运行结果/` 里的代码副本
- 命名不清楚的临时脚本，比如 `1.py`、日期命名脚本
- 以统计、绘图、一次性分析为主的辅助脚本
- `build_event_dataset_final.py`
  - 名字像最终版，但当前文件内容更像旧稿，不作为主版本保留

## 使用原则

从现在开始，优先只看这个目录里的代码。原始目录保留，作为追溯历史时再查。
