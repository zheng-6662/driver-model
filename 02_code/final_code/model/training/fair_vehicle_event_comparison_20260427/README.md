# 十二组公平对比实验代码

这个文件夹用于本轮“车辆数据 / 显式事件注入 / 教师强制 / 粗细输出头 / 教师状态 / 连续驾驶风格”公平对比。前六组保留原来的结构对照；第 07-12 组先把显式事件全部拿掉，只比较车辆直接预测、无事件粗细双头、教师状态和连续驾驶风格的贡献。

## 运行入口

当前系统默认 `python` 指向 Python 3.5，不能运行这个项目。请优先用 `py -3.11`，或者使用你自己已经配好 PyTorch 的 Python 3.11 环境。

在 PowerShell 里进入本文件夹后分别运行：

```powershell
py -3.11 ".\01_只有车辆数据_直接预测轨迹.py"
py -3.11 ".\02_车辆数据_显式事件信息注入.py"
py -3.11 ".\03_车辆数据_显式事件信息注入_无教师强制.py"
py -3.11 ".\04_车辆数据_显式事件信息注入_粗细两个状态头.py"
py -3.11 ".\05_车辆数据_教师状态_连续驾驶风格.py"
py -3.11 ".\06_车辆数据_教师状态_连续驾驶风格_直接预测.py"
py -3.11 ".\07_车辆数据_教师状态_直接预测.py"
py -3.11 ".\08_车辆数据_连续驾驶风格_直接预测.py"
py -3.11 ".\09_车辆数据_粗细双头_无显式事件.py"
py -3.11 ".\10_车辆数据_粗细双头_教师状态_无显式事件.py"
py -3.11 ".\11_车辆数据_粗细双头_连续驾驶风格_无显式事件.py"
py -3.11 ".\12_车辆数据_粗细双头_教师状态_连续驾驶风格_无显式事件.py"
```

也可以顺序运行指定集合：

```powershell
.\run_all_four.ps1
.\run_all_five.ps1
.\run_all_six.ps1
.\run_no_event_controls.ps1
.\run_all_twelve.ps1
```

训练完成后，脚本会在终端打印每一组的 `输出文件夹`。按当前底层训练脚本，默认结果会进入：

```text
F:\data_set_process\data_process\tmp\event_conditioned_runs\
```

最终以每个脚本实际打印出来的 `输出文件夹` 为准。

## 十二组实验含义

| 脚本 | 中文含义 | 结构设置 | 教师强制 | 事件损失 |
|---|---|---|---|---|
| `01_只有车辆数据_直接预测轨迹.py` | 只输入车辆历史/上下文，轨迹头直接输出未来轨迹 | `conditioning_mode=vehicle_direct`，轨迹头不读取事件嵌入、不做事件 FiLM、不做结构轨迹注入 | `0.0` | `0.0` |
| `02_车辆数据_显式事件信息注入.py` | 车辆数据 + 显式事件结构注入 | `conditioning_mode=structured_v2`，启用事件嵌入、结构轨迹、结构 FiLM、方向盘结构残差 | `1.0` | `0.5` |
| `03_车辆数据_显式事件信息注入_无教师强制.py` | 同样显式事件结构注入，但训练时不喂真实事件 | `conditioning_mode=structured_v2`，结构和第 2 组一致 | `0.0` | `0.5` |
| `04_车辆数据_显式事件信息注入_粗细两个状态头.py` | 显式事件结构注入 + 粗趋势头 + 细残差头 | `conditioning_mode=structured_v2_coarse_fine`，事件注入和第 2 组一致，输出端拆成粗/细两个头后相加 | `1.0` | `0.5` |
| `05_车辆数据_教师状态_连续驾驶风格.py` | 车辆数据 + 教师状态 + 连续驾驶风格 | `conditioning_mode=structured_v2_coarse_fine`，结构和第 4 组一致；额外开启 `enable_teacher_state_context=True` 和 `enable_driver_style_context=True` | `1.0` | `0.5` |
| `06_车辆数据_教师状态_连续驾驶风格_直接预测.py` | 车辆数据 + 教师状态 + 连续驾驶风格，直接预测轨迹 | `conditioning_mode=vehicle_direct`，结构和第 1 组一致；额外开启 `enable_teacher_state_context=True` 和 `enable_driver_style_context=True` | `0.0` | `0.0` |
| `07_车辆数据_教师状态_直接预测.py` | 车辆数据 + 教师状态，直接预测轨迹 | `conditioning_mode=vehicle_direct`，结构和第 1 组一致；额外开启 `enable_teacher_state_context=True` | `0.0` | `0.0` |
| `08_车辆数据_连续驾驶风格_直接预测.py` | 车辆数据 + 连续驾驶风格，直接预测轨迹 | `conditioning_mode=vehicle_direct`，结构和第 1 组一致；额外开启 `enable_driver_style_context=True` | `0.0` | `0.0` |
| `09_车辆数据_粗细双头_无显式事件.py` | 车辆数据 + 粗细双头，不使用显式事件 | `conditioning_mode=vehicle_direct_coarse_fine`，轨迹头不读取事件嵌入、不做事件 FiLM、不做结构轨迹注入，只使用 coarse/fine 输出分解 | `0.0` | `0.0` |
| `10_车辆数据_粗细双头_教师状态_无显式事件.py` | 车辆数据 + 粗细双头 + 教师状态，不使用显式事件 | `conditioning_mode=vehicle_direct_coarse_fine`，额外开启 `enable_teacher_state_context=True` | `0.0` | `0.0` |
| `11_车辆数据_粗细双头_连续驾驶风格_无显式事件.py` | 车辆数据 + 粗细双头 + 连续驾驶风格，不使用显式事件 | `conditioning_mode=vehicle_direct_coarse_fine`，额外开启 `enable_driver_style_context=True` | `0.0` | `0.0` |
| `12_车辆数据_粗细双头_教师状态_连续驾驶风格_无显式事件.py` | 车辆数据 + 粗细双头 + 教师状态 + 连续驾驶风格，不使用显式事件 | `conditioning_mode=vehicle_direct_coarse_fine`，额外开启教师状态和连续驾驶风格上下文 | `0.0` | `0.0` |

第 1 组新增的 `vehicle_direct` 是为了避免把旧代码里的 `baseline` 误当成“只有车辆数据”。旧 `baseline` 仍然会使用事件嵌入；本轮第 1 组不会把事件信息送进轨迹解码器。

第 4 组和第 09-12 组的“粗细两个状态头”在代码里具体实现为两个轨迹输出头：粗头先在时间维上池化出低频趋势，再上采样回 400 个未来点；细头直接输出逐时刻残差；最后两者相加得到预测轨迹。区别是第 4 组仍然使用显式事件，第 09-12 组完全不使用显式事件。

第 5-8 组和第 10-12 组都不是把驾驶员硬分成“保守/激进/稳健”类别，而是把当前 `driver_style_vectors.csv` 中的驾驶行为统计压缩成连续 `style_vector_1..4`；教师状态也不是人工给定标签，而是把生理/EEG 基础特征在训练集上标准化后用 PCA 压缩成 `teacher_state_1..4`，再作为额外上下文拼到 `ctx` 后面。

第 5 组回答的是“在显式事件 + 粗细双头结构上，教师状态和驾驶风格有没有额外帮助”；第 6 组回答的是“即使不使用显式事件结构，只做车辆直接轨迹预测，教师状态和驾驶风格有没有帮助”。

第 07-12 组回答的是“先完全去掉显式事件时，教师状态、连续驾驶风格、粗细双头各自有没有帮助，以及它们组合后有没有额外收益”。

## 输入到底有哪些

十二组使用同一套样本构造和同一份 manifest：

```text
F:\data_set_process\data_process\02_code\final_code\model\training\protocol_allphase_control_v2_context_full2s\sample_manifest.csv
```

所有版本共同输入：

1. 过去 3 秒车辆历史窗口 `src`：600 个历史点，200 Hz。包含 roll、yawrate、ay、ax、speed、z；如果文件里有车道横向距离，还会包含 lateraldistance、lane_rate、lane_unwrap、lane_unwrap_rate；另外包含未来道路相关建模常用的历史 lane curvature、yaw、当前 steering wheel、LTR_est、steer_rate、speed_rate。
2. 当前锚点上下文 `ctx`：当前方向盘角、当前速度、当前方向盘角速度、当前横向加速度 ay、当前 yawrate。
3. 未来 2 秒道路曲率 `curve_norm`：400 个未来点，作为道路预瞄信息输入每个版本。它不是未来方向盘标签，但严格说它属于未来道路信息，不是纯粹只看过去车辆历史。
4. 训练标签 `y_true`：未来 2 秒的方向盘相对变化 `steer_rel` 和速度变化 `speed_delta`，只作为监督目标。
5. 有效长度 mask：用于忽略不足 2 秒的未来片段。

事件相关信息不是额外 CSV 输入，而是由未来真实轨迹标签 `y_true` 自动计算出来的事件监督目标，包括：

```text
第一次明显转向是否出现、第一次明显转向时间、第一次明显转向方向、
第一次反转是否出现、第一次反转时间、
主峰时间、主峰方向
```

第 2 组、第 4 组和第 5 组训练时会用真实事件摘要做教师强制，所以训练阶段轨迹头会看到由真实未来轨迹计算出来的事件结构；验证/测试阶段仍然使用模型自己预测的事件摘要。第 3 组训练、验证、测试都不喂真实事件摘要，事件注入来自模型自己的事件头预测。第 1 组、第 6-12 组不把事件摘要送入轨迹头，也不训练事件损失。

带教师状态的组额外拼接 4 维 `teacher_state_1..4`；带连续驾驶风格的组额外拼接 4 维 `style_vector_1..4`；两者都开启时 `ctx` 从 5 维车辆上下文扩展到 13 维。训练脚本会在每次运行目录保存 `context_augmentation_meta.json` 和 `context_feature_names.json`，用于检查具体使用了哪些特征、PCA 解释方差和每个主成分权重最大的原始行为统计。

## 共同配置

十二组共同配置都集中在：

```text
common_compare_runner.py
```

当前共同 fullrun 设置如下：

```text
manifest = protocol_allphase_control_v2_context_full2s/sample_manifest.csv
seed = 2026
device = auto
epochs = 40
min_epochs = 40
patience = 99
batch_size = 64
lr = 0.001
weight_decay = 0.0
grad_clip = 1.0
selection_mode = legacy_rmse
d_model = 128
nhead = 2
enc_layers = 2
dec_layers = 2
ffn_dim = 256
dropout = 0.1
event_embed_dim = 96
event_bin_size = 20
structure_width = 0.065
gate_temperature = 0.040
event_residual_scale = 1.0
max_train_samples/max_val_samples/max_test_samples = None
smoke_test = False
```

如果后面要统一改训练轮数、batch size、学习率或设备，只改 `common_compare_runner.py` 里的 `COMMON_FULLRUN_CONFIG`。不要分别改各个入口脚本，否则就不是公平对比。

## 这轮对比要回答的问题

1. “只看车辆数据直接预测轨迹”能达到什么基础效果。
2. 在相同数据切分和训练设置下，加入显式事件结构注入是否改善方向盘轨迹，尤其是整体 RMSE、尾段误差、峰值时间、转向次数和边界偏移。
3. 显式事件结构注入的收益是否依赖训练阶段的真实事件教师强制。
4. 在显式事件结构注入基础上，把输出拆成粗趋势头和细残差头，是否比单一轨迹头更稳定。
5. 在第 4 组结构不变的情况下，加入教师状态和连续驾驶风格是否能带来额外收益。
6. 在第 1 组车辆直接预测结构不变的情况下，加入教师状态和连续驾驶风格是否能带来额外收益。
7. 在完全无显式事件时，教师状态、连续驾驶风格、粗细双头各自是否有效。
8. 在完全无显式事件时，教师状态、连续驾驶风格和粗细双头组合后是否有额外收益。

## 注意事项

- 第 1-4 组和第 9 组不包含生理/EEG 教师状态，也不包含驾驶风格；其他相应控制组才加入教师状态或连续驾驶风格。
- 所有带驾驶风格的组使用的是连续驾驶风格向量，不是驾驶风格离散分组。
- 第 2 组、第 4 组和第 5 组的教师强制只发生在训练阶段；验证、测试和后续独立评估仍然使用模型自己预测的事件摘要。
- 旧的 `baseline` 不是“无事件直接预测”，所以这轮专门加了 `vehicle_direct` 模式。
- 当前 `device = "auto"`：有 CUDA 的 Python 环境会自动用 CUDA，没有 CUDA 会用 CPU。若要强制固定设备，把 `common_compare_runner.py` 里的 `device` 统一改成 `"cuda"` 或 `"cpu"`，所有要对比的组必须一起改。
