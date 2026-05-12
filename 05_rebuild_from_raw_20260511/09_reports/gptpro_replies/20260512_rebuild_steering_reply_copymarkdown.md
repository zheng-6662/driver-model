我会把回答组织成可交给 Codex 执行的项目重启方案：先审计原始数据与标签可信度，再用可证伪实验验证风格/生理增量，最后规划模型、评价和 GPTPro 协作。

我会把阶段设计成带停止规则的证据链，并补充失败样本档案与方法候选的定位；任何强模型都只能在数据审计通过后作为验证工具使用。

# 总判断：应该从原始 `.csv` 重建流程，而且这是合理选择

在你们当前状态下，继续在旧流程上堆模型，很可能只能得到“RMSE 小幅波动、预测图仍不物理”的结果。原因不是某个网络头不够复杂，而是整个任务的因果边界、事件锚点、时间同步、标签窗口、输入可用时刻、输出形式和评价指标还没有围绕“事件后物理正确响应”重新组织。

从原始 `.csv` 重建的主要价值不是“重洗一遍数据”，而是解决以下根本问题：

1. **确认模型到底在预测什么时刻之后的响应**

如果事件锚点偏早、偏晚，或者锚点本身由方向盘变化反推得到，那么模型学习到的就不是事件触发响应，而是被标签污染后的对齐结果。
2. **确认输入是否真的因果可用**

生理信号尤其危险。肌电如果包含动作开始后的信息，可能会让模型看见“已经发生的动作”。这不是不能用，但必须改成“早期感知后预测剩余轨迹”的任务，而不能和“事件发生瞬间预测完整未来轨迹”混在一起。
3. **确认时间同步和重采样没有制造假规律**

车辆、生理、脑电的采样率、时间戳、延迟、漂移、缺失、重复点，都可能让模型学到错误对应关系。
4. **确认标签窗口覆盖完整响应**

旧模型尾段回正不稳定、反向修正被抹平，可能不是模型不会学，而是标签窗口、损失函数和样本定义没有显式保留这些结构。
5. **确认标准化、基线校正和特征提取没有泄漏测试集信息**

用全体数据统计量、整段 trial 的生理基线、事件后窗口提取风格，都可能让结果不可解释。
6. **建立“连续驾驶风格 + 生理数据”的证据链**

你们最终不是只要一个 RMSE 更低的模型，而是要证明或者否定：连续驾驶风格和生理数据是否提供了车辆历史难以替代的信息。这个问题必须从数据集定义阶段开始设计，不是模型训练完再解释。

结论：**建议暂停旧流程新增模型，先从原始 `.csv` 做一次可审计重建。旧流程可以作为问题清单、失败样本库、对照基线和日志来源，但不能再作为默认正确的数据处理依据。**

---

# 新长期目标名称

**Raw-to-Evidence 方向盘转角事件响应预测重建计划**

也可以简称：

**R2E-Steering：从原始数据到连续风格与生理增量价值证据链**

---

# 长期最终目标

建立一套从原始车辆、生理、脑电 `.csv` 出发的、可追溯、可复现、无泄漏的事件触发后方向盘转角轨迹预测流程，用来回答两个核心科学问题：

1. **连续驾驶风格是否提供了普通车辆历史模型无法替代的个体差异信息？**
2. **生理数据是否提供了车辆历史与连续风格之外的驾驶员内部状态、运动准备、认知负荷、响应类型、困难样本或可靠性信息？**

同时，模型预测不应只追求整体 RMSE，而应在以下方面明显改善：

- 方向正确；
- 大幅响应不被压小；
- 主峰时间和幅值合理；
- 尾段回正稳定；
- 反向修正和多段修正不被平均成平滑单峰；
- 困难样本失败原因可追溯；
- 生理和风格的增量价值有置乱、对照和分层证据支持。

---

# 阶段总览

## 阶段 0：冻结旧流程，建立重建准则

### 核心问题

旧流程哪些信息可以保留为参考？哪些必须重新验证？

### 输入材料

- 旧项目日志；
- 旧代码；
- 旧模型结果；
- 旧失败样本；
- 旧事件设定；
- 旧预处理说明；
- 当前原始数据目录结构。

### 输出文件

建议生成：

```
docs/00_old_pipeline_review.mddocs/00_rebuild_principles.mddocs/00_failure_taxonomy_from_old_models.mddocs/00_do_not_trust_list.md
```

### 完成标准

必须明确列出：

- 旧流程可参考内容；
- 旧流程不可默认相信内容；
- 新流程必须重新审计的项目；
- 旧困难样本 ID 与新样本清单之间的追溯方式。

### 停止规则

如果旧代码、旧日志和原始数据之间无法建立基本映射，例如被试 ID、事件 ID、道路 ID 完全不一致，则不能进入模型阶段，只能进入“数据映射修复”。

### 进入下一阶段条件

能回答：

- 原始文件在哪里；
- 旧事件日志在哪里；
- 哪些文件代表车辆、生理、脑电；
- 旧失败样本能否追溯到原始 trial 或事件。

---

## 阶段 1：原始数据审计

这是最重要的阶段。不要先训练模型。

### 核心问题

原始数据是否足以支持一个因果、同步、无泄漏的事件后方向盘转角预测任务？

### 输入材料

```
F:/data_set_process/data_process/01_datasets/数据预处理/原始车辆数据/F:/data_set_process/data_process/01_datasets/数据预处理/原始生理数据/F:/data_set_process/data_process/01_datasets/数据预处理/原始脑电数据/旧道路信息日志旧事件设定日志旧事件锚点记录旧预处理代码，仅作参考
```

### 输出文件

建议生成：

```
audit/file_inventory.csvaudit/file_checksums.csvaudit/subject_session_inventory.csvaudit/modality_availability.csvaudit/timestamp_audit.csvaudit/sampling_rate_audit.csvaudit/time_overlap_audit.csvaudit/event_anchor_audit.csvaudit/prediction_window_coverage.csvaudit/leakage_risk_report.mdaudit/phys_signal_quality.csvaudit/eeg_signal_quality.csvaudit/emg_action_leakage_audit.csvaudit/audit_summary.mdfigures/audit/timestamp_gap_examples/figures/audit/sampling_rate_histograms/figures/audit/modality_overlap_examples/figures/audit/event_anchor_examples/figures/audit/steering_response_window_examples/figures/audit/phys_quality_examples/figures/audit/eeg_artifact_examples/figures/audit/emg_vs_steering_onset_examples/
```

### 完成标准

至少要完成以下审计：

- 每个原始文件可读；
- 每个文件有 checksum；
- 每个被试、session、trial、模态覆盖情况清楚；
- 每个模态时间戳连续性清楚；
- 每个模态实际采样率清楚；
- 车辆、生理、脑电时间重叠范围清楚；
- 事件锚点来源和可信度清楚；
- 每个事件的输入窗口和标签窗口是否完整清楚；
- 标准化、基线校正、重采样、样本切分是否存在泄漏风险清楚；
- 生理与脑电质量有量化标记；
- 肌电是否包含事件后动作结果有专门审计。

### 停止规则

出现以下情况，必须停在数据阶段：

1. 大量文件不可读、缺列、ID 无法匹配；
2. `StorageTime` 大量非单调、重复、跳变，且无法修复；
3. 模态之间没有足够时间重叠；
4. 事件锚点不是外部事件，而是由方向盘响应反推出；
5. 标签窗口普遍不覆盖完整响应；
6. 标准化或基线校正依赖测试集、整段 trial 或事件后未来窗口；
7. 生理信号质量普遍不可用；
8. 肌电在当前输入窗口中明显包含动作结果，但任务仍被定义为“事件瞬间预测完整未来”。

### 进入下一阶段条件

可以为每个事件样本生成一条清楚记录：

```
这个样本来自哪个原始文件？哪个被试？哪个事件？锚点是什么时间？输入窗口是什么？标签窗口是什么？车辆、生理、脑电是否可用？时间同步是否可信？信号质量如何？是否有泄漏风险？
```

---

## 阶段 2：事件定义与样本清单重建

### 核心问题

每一个训练样本是否能完整追溯到原始数据、事件锚点、输入窗口、标签窗口和信号质量？

### 输入材料

- 阶段 1 审计结果；
- 原始车辆数据；
- 原始生理数据；
- 原始脑电数据；
- 道路和事件日志；
- 旧事件锚点，仅作候选来源。

### 输出文件

建议生成：

```
dataset_manifest/samples_master.csvdataset_manifest/samples_master.jsonldataset_manifest/sample_windows.parquetdataset_manifest/modality_segments_index.parquetdataset_manifest/event_anchor_table.csvdataset_manifest/split_table.csvdataset_manifest/exclusion_reasons.csvdataset_manifest/dataset_version_card.md
```

每个样本至少记录以下字段。

### 样本身份字段

```
sample_iddataset_versionsubject_idsession_idtrial_idroad_idevent_idevent_typeevent_subtypeold_sample_idsource_vehicle_filesource_phys_filesource_eeg_fileraw_file_checksum_vehicleraw_file_checksum_physraw_file_checksum_eeg
```

### 事件锚点字段

```
event_anchor_timeevent_anchor_sourceevent_anchor_confidenceevent_anchor_ruleevent_anchor_manual_noteevent_anchor_time_in_vehicle_clockevent_anchor_time_in_phys_clockevent_anchor_time_in_eeg_clocktime_sync_offset_vehicle_phystime_sync_offset_vehicle_eegtime_sync_drift_estimate
```

### 输入窗口字段

```
vehicle_input_start_timevehicle_input_end_timephys_input_start_timephys_input_end_timeeeg_input_start_timeeeg_input_end_timeinput_available_modecausal_setting
```

其中 `causal_setting` 建议至少分成：

```
T0_event_onset_onlyTplus_early_post_event_100msTplus_early_post_event_300msTplus_early_post_event_500ms
```

注意：不同 causal setting 不能混在同一个主实验里比较。

### 标签窗口字段

```
label_start_timelabel_end_timelabel_durationlabel_sampling_ratesteering_label_columnlabel_complete_flagresponse_onset_timeresponse_peak_timeresponse_peak_valueresponse_tail_valueresponse_type_labelmulti_phase_flagreverse_correction_flaglarge_response_flag
```

响应类型标签可以先由规则生成，再人工抽查，不要一开始就当作绝对真值。

### 模态可用性和质量字段

```
vehicle_availablephys_availableeeg_availableecg_quality_scoreemg_quality_scoreeda_quality_scoreresp_quality_scoreeeg_quality_scorevehicle_quality_scoretimestamp_quality_scoresync_quality_scoreoverall_sample_quality
```

### 泄漏风险字段

```
leakage_risk_flagleakage_risk_typenormalization_groupbaseline_window_startbaseline_window_enduses_post_event_physuses_post_event_steeringstyle_window_startstyle_window_endstyle_uses_future_flag
```

### 数据切分字段

```
split_randomsplit_by_subjectsplit_by_sessionsplit_by_roadsplit_by_event_typefold_idcalibration_available_for_subject
```

### 失败追溯字段

```
old_failure_flagnew_failure_flagfailure_categoryfailure_notesplot_pathraw_segment_path
```

### 完成标准

- 每个样本有唯一 `sample_id`；
- 每个样本能回到原始 `.csv` 行范围；
- 每个样本有明确事件锚点；
- 每个样本有输入窗口、标签窗口、质量标记、泄漏标记；
- 每个样本知道自己属于哪个 split；
- 每个预测失败样本可以生成固定格式复盘图。

### 停止规则

如果样本清单不能追溯到原始文件和行范围，不能进入正式训练。

---

## 阶段 3：无学习基线与纯车辆基线

### 核心问题

在不使用连续风格和生理数据时，任务本身能做到什么水平？当前数据的可预测上限和基本错误类型是什么？

### 输入材料

- `samples_master.csv`
- 车辆输入窗口；
- 标签轨迹；
- 事件类型；
- 道路信息。

### 必须先做的模型

1. **零变化基线**

预测未来方向盘保持输入末端值。
2. **历史延续基线**

用输入末端速度、斜率、局部趋势延续。
3. **事件均值基线**

按事件类型、道路类型、被试组别求平均响应。
4. **近邻基线**

用事件前车辆状态、车速、曲率、横向偏移、方向盘历史寻找相似样本。
5. **纯车辆时序模型**

输入车辆历史，输出未来方向盘轨迹。
6. **车辆 + 事件模型**

加入事件类型、道路几何、道路曲率等信息。

### 输出文件

```
results/baselines/no_learning_metrics.csvresults/baselines/vehicle_model_metrics.csvresults/baselines/event_vehicle_model_metrics.csvresults/baselines/fixed_plot_panel/results/baselines/error_taxonomy.csvmodels/baselines/
```

### 完成标准

- 每个基线在同一套 split 上评估；
- 指标不只包括 RMSE；
- 固定预测图面板生成；
- 困难样本名单更新；
- 明确车辆模型的失败模式。

### 停止规则

如果车辆 + 事件模型仍然出现大量方向错侧、主峰压小、尾段漂移，需要先确认标签、锚点、输出形式和损失函数，而不是直接加生理。

### 进入下一阶段条件

纯车辆和车辆 + 事件基线稳定，可作为后续风格、生理增量对照。

---

## 阶段 4：连续驾驶风格有效性验证

### 核心问题

连续驾驶风格是否提供了车辆历史和事件信息之外的个体差异信息？它是不是只是驾驶员编号的替代物？

### 风格从哪些数据提取

只允许从**事件前、非标签、因果可用**的数据中提取，例如：

```
事件前 30s / 60s / 120s 车辆历史当前 trial 中事件前的稳定驾驶段当前 session 中该事件之前的非事件驾驶段独立 calibration driving 段，如果存在
```

可提取的风格信息包括：

- 方向盘平滑性；
- 方向盘变化率分布；
- 转向修正频率；
- 车速控制习惯；
- 横向距离保持习惯；
- 曲率跟随方式；
- yaw 与 steering 的耦合特征；
- lane curvature 下的平均 steering 策略；
- 对道路曲率变化的提前量；
- 驾驶稳定性；
- 事件前短时驾驶激进程度。

### 必须避免的东西

不能从以下来源提取风格：

```
事件后的方向盘标签窗口当前事件的峰值、响应方向、响应延迟整段 trial 的统计量，如果包含当前事件之后所有被试全数据的全局统计后再反用到测试测试集未来样本
```

### 推荐风格表示

分三层做，不要一开始就端到端黑盒：

#### 1. 手工连续风格特征

用于解释：

```
steering_std_presteering_rate_mean_presteering_rate_p95_prespeed_mean_prespeed_std_prelateral_distance_mean_prelateral_distance_std_precurvature_following_error_preyaw_rate_variability_premicro_correction_count_pre
```

#### 2. 自监督车辆风格 embedding

用事件前车辆序列训练，例如：

- 预测被遮蔽的车辆片段；
- 预测未来短时车辆状态；
- 对比同一被试不同非事件片段；
- 学习道路条件归一化后的驾驶习惯。

#### 3. 分层个体随机效应

用轻量模型估计：

```
全局响应模式 + subject/session/style 偏置
```

这可以检验个体差异是否存在，而不直接依赖驾驶员 ID。

### 如何证明它不是驾驶员编号

必须做以下对照：

1. **驾驶员 ID-only 模型**

只给 subject_id，不给连续风格。看 ID 本身能带来多少提升。
2. **车辆 + 驾驶员 ID 模型**

检查是否仅靠记住被试就能提升。
3. **车辆 + 连续风格模型**

不给 subject_id，只给从事件前行为提取的风格。
4. **车辆 + ID + 连续风格模型**

检查风格是否在 ID 之外还有增量。
5. **风格置乱模型**

保持车辆和标签不变，打乱风格 embedding。
6. **同道路同事件置乱**

在相同道路、相同事件类型、相近速度样本之间置乱风格，避免置乱破坏数据分布后得到虚假结论。
7. **同被试异时段置乱**

检查风格是否只是 subject_id，还是短时连续驾驶状态。
8. **风格可识别性审计**

训练一个小模型从风格 embedding 预测 subject_id。如果非常容易预测 subject_id，则需要报告：该风格表示可能混入强身份信息。

### 分被试和跨被试评估

至少做三种设置：

#### 设置 A：随机样本切分

只能作为开发调试，不能作为最终证据。

#### 设置 B：按 session 或 trial 切分

同一被试不同 session 或 trial 分开，用于验证风格是否在同一驾驶员内泛化。

#### 设置 C：跨被试切分 + 校准段

训练集被试和测试集被试不同。测试被试只允许用事件前或独立 calibration 段提取风格，不允许用测试事件标签。

这是证明“连续风格不是简单 ID 记忆”的关键设置。

### 什么时候可以说连续驾驶风格有效

同时满足以下条件，才可以说有效：

1. 风格全部来自事件前或独立 calibration 数据；
2. 在车辆 + 事件强基线之上有稳定提升；
3. 提升不仅体现在整体 RMSE，也体现在方向、峰值幅值、峰值时间、尾段或困难样本指标上；
4. 多个随机种子、多折 split 下稳定；
5. 风格置乱后提升显著下降；
6. 驾驶员 ID-only 不能完全替代风格；
7. 跨被试 + 校准段设置下仍有价值，或至少在同被试跨 session 中有清楚价值；
8. 不依赖事件后信息；
9. 不是只在某一个被试或某一个事件类型上有效。

### 什么时候不能说有效

以下情况只能说“当前证据不足”：

- 只在随机样本切分中提升；
- 风格置乱后仍然提升；
- 驾驶员 ID-only 与风格效果几乎一样；
- 风格提取窗口包含事件后标签；
- 提升只来自一两个被试；
- 只提升 RMSE，但方向错侧、大幅响应召回、尾段指标没有改善；
- 风格 embedding 本质上只是 subject_id 编码。

---

## 阶段 5：生理数据角色重建与有效性验证

### 核心原则

不要默认把生理信号拼到轨迹回归模型里。生理数据要先回答：

```
它是状态信号？它是动作准备信号？它是认知负荷信号？它是响应类型信号？它是困难样本或可靠性信号？它是训练教师？它是部署时推理输入？还是当前数据质量不足？
```

---

# 各生理模态建议角色

## 1. 心率 / ECG

### 适合角色

心率更适合作为：

- 基线状态；
- 唤醒水平；
- 疲劳或紧张程度；
- 慢变个体状态；
- 样本可靠性或困难样本风险；
- 风格之外的驾驶员状态调制信号。

### 不适合一开始承担的角色

不建议一开始让 ECG 直接预测：

- 瞬时方向盘主峰方向；
- 细粒度反向修正；
- 毫秒级响应轨迹。

原因是它通常更慢、更状态化，而不是直接动作轨迹信号。

### 推荐实验

```
车辆 + 事件车辆 + 事件 + ECG状态车辆 + 事件 + 风格车辆 + 事件 + 风格 + ECG状态ECG置乱对照ECG质量分层对照
```

重点看：

- 困难样本识别；
- 响应延迟；
- 大幅响应风险；
- 模型置信度校准；
- 预测失败概率。

---

## 2. 皮电 / EDA

### 适合角色

EDA 更适合做：

- arousal / stress 状态；
- 事件前紧张程度；
- 困难样本概率；
- 响应可靠性；
- 模型门控；
- 预测不确定性调制。

### 风险

EDA 事件后反应较慢，如果使用事件后 EDA 预测同一事件短时方向盘轨迹，很可能因果意义较弱。必须区分：

```
事件前 EDA 状态事件后 EDA 变化
```

事件后 EDA 更适合解释，而不一定适合作为实时预测输入。

### 推荐实验

EDA 不应先作为轨迹主输入，而应先做：

- 困难样本分类；
- 大幅响应风险；
- 响应延迟分层；
- 不确定性估计；
- 样本可靠性判断。

---

## 3. 肌电 / EMG

### 适合角色

EMG 是最有可能对方向盘响应有直接贡献的生理信号，尤其可能用于：

- 动作准备；
- 响应启动延迟；
- 方向判断；
- 幅值趋势；
- 大幅响应召回；
- 反向修正或多段修正的早期线索。

### 最大风险

EMG 也最容易造成“动作结果泄漏”。

必须明确两种任务：

#### 任务 T0：事件发生瞬间预测完整未来

输入只能到事件锚点之前：

```
input_end_time <= event_anchor_time
```

此时 EMG 只能代表事件前状态或准备，不能包含事件后肌肉激活。

#### 任务 T+Δ：事件后早期感知预测剩余未来

例如允许使用事件后 300ms EMG：

```
input_end_time = event_anchor_time + 300mslabel_start_time = event_anchor_time + 300ms
```

这样 EMG 可以作为早期动作信号，但模型评估必须只看之后的剩余轨迹，不能把已经发生的方向盘变化也算成预测成绩。

### 必须做的 EMG 审计

```
EMG onset timesteering onset timeEMG onset - steering onsetevent anchor - EMG onsetEMG energy pre-eventEMG energy post-event earlyEMG是否饱和EMG是否平线EMG是否被车辆运动伪迹污染
```

### 推荐实验

EMG 最值得做，但必须先做因果版本：

```
T0车辆 + 风格 + EMG_preT+100ms车辆 + 风格 + EMG_0_100msT+300ms车辆 + 风格 + EMG_0_300msT+500ms车辆 + 风格 + EMG_0_500ms
```

每个任务要单独报告，不能混在一起。

---

## 4. 脑电 / EEG

### 适合角色

EEG 更适合作为：

- 离线教师；
- 训练期辅助监督；
- 运动准备或认知状态表征；
- 困难样本解释；
- 响应类型辅助任务；
- 跨模态蒸馏来源。

### 不建议一开始作为部署主输入

除非你们目标是实验室系统，否则 EEG 作为推理输入的部署难度高、信号质量风险高、伪迹复杂。更合理的路线是：

```
训练时使用 EEG 教师推理时使用 车辆 + 风格 + EMG/ECG/EDA
```

### 必须先做的 EEG 审计

- 通道平线；
- 极端值；
- 通道掉线；
- 高频噪声；
- 眼动或运动伪迹；
- accelerometer 与 EEG 异常同步；
- 事件附近 EEG 是否被身体动作污染；
- 每个样本 EEG 可用通道比例；
- 每个被试 EEG 质量差异。

### 推荐实验

不要一开始训练大 EEG 端到端模型。先做：

```
EEG是否能预测响应类型EEG是否能预测方向EEG是否能预测大幅响应EEG是否能预测延迟EEG教师蒸馏是否改善无EEG推理模型EEG置乱后教师增益是否消失
```

---

## 5. 呼吸 / RESP

### 是否保留

建议保留，但优先级低于 EMG、ECG、EDA、EEG。

### 适合角色

- 慢变生理状态；
- 疲劳或紧张；
- ECG/EDA 质量辅助；
- 模型可靠性；
- 困难样本分层。

### 不建议角色

不建议直接作为主轨迹回归核心输入。

---

# 哪些信号适合作为推理输入

优先级建议：

```
高优先级：车辆历史、事件信息、连续驾驶风格、EMG中优先级：ECG状态、EDA状态、RESP状态低优先级：EEG作为在线推理输入
```

# 哪些信号适合作为训练教师

```
EEG：强候选教师EMG：动作准备教师或早期响应教师ECG/EDA/RESP：状态教师、可靠性教师、困难样本教师
```

# 哪些信号适合做辅助任务

```
方向：EMG、EEG幅值：EMG、车辆历史、风格延迟：EMG、EEG、ECG状态响应类型：车辆历史、EMG、EEG困难样本：ECG、EDA、RESP、EEG质量、车辆状态可靠性：ECG、EDA、RESP、生理质量分数
```

# 生理数据什么时候可以说有效

必须满足：

1. 信号质量审计通过；
2. 输入窗口因果合法；
3. 在车辆 + 事件 + 风格强基线之上有稳定增益；
4. 生理置乱后增益消失或明显下降；
5. 生理质量高的样本中增益更强，低质量样本中增益减弱；
6. 增益符合信号生理角色，例如 EMG 改善方向、延迟和幅值，EDA/ECG 改善困难样本或可靠性；
7. 不只是随机样本切分有效；
8. 不只是某一个被试有效；
9. 不依赖事件后标签泄漏；
10. 在固定预测图中能看到物理错误减少，而不只是 RMSE 微降。

# 什么时候只能说“当前数据不足以证明生理有效”

出现以下情况时，不要强行声称有效：

- 生理信号质量差；
- 模态缺失严重；
- 时间同步不可信；
- 生理增益只在随机 split 出现；
- 生理置乱后仍然有同样提升；
- 生理模型只提升 RMSE，但方向、峰值、大幅响应、尾段没有改善；
- EMG 增益来自事件后动作结果，但任务仍声称是事件瞬间预测；
- EEG 教师蒸馏没有比无 EEG 模型更好；
- ECG/EDA/RESP 只能预测被试 ID，不能解释响应差异；
- 结果跨种子、跨被试、跨事件不稳定。

---

# 阶段 6：结构化轨迹建模

### 核心问题

如何避免模型把多种合理响应平均成一条平滑、幅值偏小、方向可能错侧的轨迹？

### 不建议一开始做

不要一开始就做：

- 所有模态直接拼接的大 Transformer；
- 复杂 EEG 端到端模型；
- 扩散模型；
- 多候选轨迹；
- 蒸馏 + 物理约束 + 多任务全部一起上。

原因是如果数据定义、锚点、窗口和基线没清楚，复杂模型只会掩盖问题。

### 推荐模型路线

## 第 1 层：普通轨迹模型

```
车辆历史 -> steering trajectory车辆历史 + 事件 -> steering trajectory车辆历史 + 事件 + 风格 -> steering trajectory车辆历史 + 事件 + 风格 + 单一生理 -> steering trajectory
```

用途：建立增量对照。

## 第 2 层：响应分解模型

把轨迹拆成可解释因素：

```
方向 sign主峰幅值 amplitude主峰时间 peak_time响应启动延迟 onset_delay响应类型 response_type尾段回正 residual_tail完整轨迹 residual trajectory
```

模型结构：

```
输入 -> 分类/回归头 -> 轨迹生成头
```

这样可以专门解决：

- 方向错侧；
- 大幅响应被压小；
- 峰值时间错；
- 尾段漂移；
- 平滑单峰问题。

## 第 3 层：关键点 + 残差轨迹模型

先预测关键点：

```
起点响应启动点主峰点反向修正点尾段终点
```

再生成残差轨迹。

适合解决：

- 主峰错；
- 回正错；
- 多段响应；
- 反向修正。

## 第 4 层：多假设模型

用于解决单一轨迹平均化。

可选形式：

```
Mixture Density NetworkConditional VAE多候选轨迹 + winner-takes-best损失扩散式轨迹生成轨迹聚类原型 + 残差预测
```

必须同时报告：

```
best-of-K 误差top-1 误差概率校准多样性错侧率大幅响应召回
```

不能只报 best-of-K，否则会高估模型实用性。

## 第 5 层：多模态时序融合模型

用于解决车辆、生理、脑电时间尺度不同的问题。

建议结构：

```
车辆编码器：TCN / GRU / TransformerEMG编码器：高时间分辨率时序编码器ECG/EDA/RESP编码器：慢变状态编码器EEG教师编码器：训练期编码器融合层：cross-attention / gating / mixture-of-experts输出层：响应分解 + 轨迹生成
```

重点不是“模型大”，而是：

- 不同模态不同采样率；
- 不同模态不同延迟；
- 不同模态不同因果窗口；
- 缺失模态可处理；
- 质量差模态可降权。

## 第 6 层：可靠性与困难样本模型

增加一个可靠性头：

```
输入 -> 轨迹预测输入 -> 预测不确定性输入 -> 困难样本概率输入 -> 是否需要多候选
```

这个模块尤其适合使用 ECG、EDA、RESP、EEG 质量、EMG 质量作为辅助信息。

---

# 新型算法优先建议

下面不是简单堆名词，而是按你们现在的错误类型匹配算法。

## 1. 响应分解模型

### 解决的问题

- 方向错侧；
- 大幅响应被压小；
- 峰值幅值错；
- 峰值时间错；
- 尾段回正不稳定。

### 思路

先预测：

```
方向是否大幅响应响应启动延迟主峰幅值主峰时间响应类型
```

再生成轨迹。

这比直接用 MSE 回归整条轨迹更适合你们的问题。

---

## 2. 轨迹原型 + 残差模型

### 解决的问题

- 反向修正被平均掉；
- 多段修正被预测成单峰；
- 少数复杂响应学不到。

### 思路

先把真实轨迹按形状聚类：

```
小幅单峰大幅单峰先正后反先反后正多段修正快速回正尾段漂移
```

模型先预测轨迹原型，再预测残差。

---

## 3. 多假设轨迹模型

### 解决的问题

- 同样车辆状态下可能有多种驾驶员响应；
- 单一均值预测导致幅值变小；
- 模型不敢预测大动作。

### 思路

输出 K 条候选轨迹：

```
candidate_1candidate_2...candidate_Kprobability_1probability_2...probability_K
```

同时优化：

- 最佳候选接近真实；
- top-1 不能太差；
- 候选之间有足够差异；
- 概率排序合理。

---

## 4. 分位数 / 异方差轨迹模型

### 解决的问题

- 不确定性无法表达；
- 困难样本被当成普通样本；
- 模型过度平滑。

### 思路

不只预测均值，还预测区间：

```
median trajectorylower quantileupper quantileuncertainty
```

如果生理数据真的反映状态，它可能会改善不确定性估计，而不一定只降低均值 RMSE。

---

## 5. 跨模态门控模型

### 解决的问题

- 生理信号质量参差不齐；
- 不是每个样本都需要生理；
- 直接拼接导致模型忽略生理或被噪声污染。

### 思路

模型学习：

```
当前样本是否信任 EMG？是否信任 ECG？是否信任 EDA？是否信任 EEG教师？
```

输入中显式加入质量分数。

---

## 6. EEG 教师蒸馏模型

### 解决的问题

- EEG 有信息但部署困难；
- 车辆 + EMG 推理模型可能缺少认知准备信息。

### 思路

训练时：

```
车辆 + 风格 + EMG + EEG -> 教师表示车辆 + 风格 + EMG -> 学生表示
```

推理时不用 EEG。

必须做 EEG 置乱和教师消融，否则不能证明 EEG 教师有效。

---

## 7. 事件时间对齐鲁棒模型

### 解决的问题

- 锚点可能有小偏差；
- 峰值时间误差大；
- 启动延迟预测不稳。

### 思路

在训练中显式建模：

```
anchor uncertaintyonset delaytime-warp tolerancepeak time loss
```

但这必须在事件锚点审计之后再做。

---

# 评价体系：不能只看 RMSE

建议将评价分成 6 组。

---

## 1. 整体误差

```
overall_RMSEoverall_MAEnormalized_RMSE_by_true_peaktrajectory_correlation
```

用途：保留和旧流程可比性。

但它不是主指标。

---

## 2. 主响应阶段误差

先定义主响应阶段，例如：

```
response_onset_time 到 response_peak_time + margin
```

指标：

```
main_phase_RMSEmain_phase_MAEmain_phase_directional_errormain_phase_area_error
```

用途：防止模型整体 RMSE 看起来不错，但真正转向阶段失败。

---

## 3. 尾段误差

```
tail_RMSEtail_MAEfinal_value_errortail_slope_errorreturn_to_zero_error
```

用途：评估回正和漂移。

---

## 4. 方向与峰值指标

```
peak_direction_accuracypeak_wrong_side_ratelarge_response_recalllarge_response_precisionpeak_amplitude_errorpeak_amplitude_ratio = predicted_peak_abs / true_peak_abspeak_time_error
```

其中：

- `peak_wrong_side_rate` 是重点；
- `large_response_recall` 是重点；
- `peak_amplitude_ratio` 可以直接发现模型是否总是把大动作压小。

---

## 5. 响应启动和形状指标

```
onset_delay_errorzero_crossing_error_countreverse_correction_detection_accuracymulti_phase_detection_accuracyturning_point_count_errortrajectory_shape_cluster_accuracy
```

用途：专门评估反向修正、多段修正、过度平滑。

---

## 6. 分层表现

必须报告：

```
by_subjectby_sessionby_roadby_event_typeby_response_typeby_true_peak_amplitude_binby_signal_quality_binby_old_hard_sample_flagby_new_hard_sample_flag
```

重点看：

- 是否只对某些被试有效；
- 是否只对小幅响应有效；
- 是否对旧困难样本有效；
- 生理高质量样本是否更有效；
- 大幅响应是否改善。

---

# 固定规则预测图

每个模型必须生成固定图，不允许只挑好看的样本。

建议每次生成：

```
固定样本集 A：随机样本 50 个固定样本集 B：真实大幅响应 50 个固定样本集 C：方向错侧高风险样本 50 个固定样本集 D：反向修正样本 50 个固定样本集 E：多段修正样本 50 个固定样本集 F：旧流程困难样本 50 个固定样本集 G：每个被试各若干样本固定样本集 H：每个事件类型各若干样本
```

每张图必须包含：

```
事件锚点输入窗口标签窗口真实方向盘预测方向盘车辆速度道路曲率横向距离EMG能量，如果该模型使用 EMG生理质量标记样本ID被试ID事件类型主要错误标签
```

---

# 数据审计详细流程

## 1. 文件完整性

检查：

```
文件路径文件大小文件编码CSV分隔符列名行数是否空文件是否重复文件checksum修改时间
```

输出：

```
file_inventory.csvfile_checksums.csvfile_read_errors.csv
```

## 2. 被试完整性

检查：

```
每个 subject_id 有哪些车辆文件有哪些生理文件有哪些脑电文件有哪些 session有哪些 trial事件数量缺失模态数量
```

输出：

```
subject_session_inventory.csvsubject_missing_modality_report.csv
```

## 3. 模态完整性

按样本或 trial 检查：

```
车辆是否存在生理是否存在脑电是否存在时间是否重叠字段是否完整
```

输出：

```
modality_availability.csvmodality_overlap_matrix.csv
```

## 4. 时间戳连续性

对每个文件的 `StorageTime` 检查：

```
是否单调递增是否重复是否倒退相邻时间差分布最大 gapgap 出现位置是否有整段跳变是否有时间戳格式混乱
```

输出：

```
timestamp_audit.csvtimestamp_gap_examples/
```

## 5. 采样率检查

不要相信目录名里的 `200Hz`，必须从时间戳估计。

对每个模态计算：

```
median_dtmean_dtp01_dtp99_dteffective_sampling_ratejittermissing_ratioduplicate_ratio
```

输出：

```
sampling_rate_audit.csvsampling_rate_histograms/
```

## 6. 车辆、生理、脑电时间同步

检查：

```
车辆-生理重叠时间车辆-脑电重叠时间生理-脑电重叠时间是否存在固定 offset是否存在随时间变化的 drift事件锚点在各模态是否可定位
```

如果有同步标记，用同步标记。

如果没有同步标记，只能做间接审计，例如：

- 事件附近车辆响应与 EMG 激活相对时间；
- EEG accelerometer 与车辆运动变化；
- 生理文件开始时间和车辆文件开始时间差异；
- 多事件 offset 是否稳定。

输出：

```
time_overlap_audit.csvsync_offset_estimates.csvsync_confidence_by_trial.csv
```

## 7. 事件锚点合理性

每个事件锚点必须标记来源：

```
日志触发时间道路事件时间车辆状态规则人工标注旧流程锚点未知来源
```

检查：

```
锚点是否早于方向盘响应锚点是否落在车辆/生理/脑电有效时间内锚点附近道路曲率/横向距离是否符合事件定义锚点是否由未来方向盘变化反推同类事件锚点分布是否合理
```

输出：

```
event_anchor_audit.csvevent_anchor_examples/
```

## 8. 预测窗口是否覆盖完整响应

对每个事件计算：

```
响应启动时间主峰时间主峰幅值尾段回正程度是否有反向修正是否有多段修正标签窗口是否覆盖主峰标签窗口是否覆盖尾段
```

输出：

```
prediction_window_coverage.csvsteering_response_window_examples/
```

如果大量样本主峰或尾段不在标签窗口内，必须重新定义标签长度。

## 9. 未来信息泄漏审计

重点检查：

```
标准化是否用全数据统计量生理基线是否用了事件后窗口风格特征是否用了事件后数据样本切分是否先构造再泄漏统计同一事件相邻窗口是否同时进入训练和测试同一被试同一 trial 是否随机切到训练和测试辅助标签是否从测试标签反向喂给训练重采样滤波是否使用未来窗口事件锚点是否由方向盘响应反推
```

输出：

```
leakage_risk_report.mdnormalization_audit.csvsplit_leakage_audit.csvbaseline_window_audit.csv
```

## 10. 生理信号质量

对 ECG、EMG、EDA、RESP 分别审计：

### ECG

```
平线比例异常尖峰R峰可检测性心率序列稳定性缺失段饱和段
```

### EMG

```
平线比例饱和比例异常尖峰事件前能量事件后早期能量EMG onset与 steering onset 的相对时间
```

### EDA

```
平线比例突变跳变缓慢漂移缺失段事件前 tonic 水平事件附近 phasic 变化
```

### RESP

```
平线比例振幅异常频率异常缺失段运动伪迹风险
```

输出：

```
phys_signal_quality.csvphys_quality_examples/
```

## 11. 脑电异常值和伪迹

检查：

```
每通道平线比例每通道极端值比例通道掉线全通道同步尖峰异常高振幅accelerometer 同步大幅运动事件附近伪迹每个样本可用通道比例每个被试 EEG 质量分布
```

输出：

```
eeg_signal_quality.csveeg_artifact_examples/
```

## 12. 肌电是否包含事件后动作结果

这是关键审计。

对每个样本估计：

```
event_anchor_timeemg_onset_timesteering_onset_timeemg_onset_minus_event_anchorsteering_onset_minus_event_anchoremg_onset_minus_steering_onset
```

然后分组：

```
EMG事件前已激活EMG早于方向盘启动EMG晚于方向盘启动EMG与方向盘几乎同步EMG无法检测
```

如果模型使用了事件后 EMG，就必须将任务定义为：

```
早期事件后观测 -> 预测剩余轨迹
```

而不能声称是：

```
事件发生瞬间 -> 预测完整未来
```

---

# 模型阶段顺序：哪些必须先做，哪些后做

## 必须先做

```
无学习基线纯车辆模型车辆 + 事件模型响应分解标签生成固定预测图错误类型分析连续风格手工特征验证风格置乱对照生理质量审计单模态生理辅助任务
```

## 中期再做

```
车辆 + 连续风格模型车辆 + 风格 + EMG车辆 + 风格 + ECG/EDA状态生理门控模型响应类型/方向/幅值/延迟分解模型关键点 + 残差轨迹模型
```

## 后期再做

```
多假设轨迹模型概率轨迹模型扩散轨迹模型EEG教师蒸馏跨模态 cross-attention多模态 mixture-of-experts跨被试 meta-learning
```

## 不应该一开始做

```
所有模态直接拼接的大模型EEG端到端主模型复杂蒸馏模型多候选 + 生理 + 风格 + 物理约束一次性全加只追求 leaderboard RMSE 的模型
```

---

# GPTPro 与 Codex 协作流程

OpenAI 官方资料将 Codex 描述为面向软件开发的 coding agent，可用于读、改、运行代码；Codex CLI 可在本地终端的选定目录中读写和运行代码，Codex cloud 可在云环境中执行任务；官方 changelog 也提到 Codex 的 Chrome 扩展可协助处理浏览器中的网站任务。不过，涉及 GPTPro 回复的流程仍应作为“外部专家审查输入”归档和人工复核，而不是让它直接改变数据结论。

## 哪些问题应该问 GPTPro

适合问：

```
实验设计审查阶段目标是否严谨泄漏风险清单是否完整评价指标是否覆盖物理错误风格有效性证据链是否充分生理模态角色是否合理失败样本归因是否有遗漏模型路线是否过早复杂化论文叙事和结果解释是否严谨
```

不适合让 GPTPro 直接决定：

```
原始CSV是否可读某个文件是否缺列某个样本是否时间戳异常某个模型是否真实提升某个生理信号是否一定有效
```

这些必须由 Codex 本地执行和统计。

## 哪些问题应该由 Codex 本地直接执行

```
扫描原始文件生成checksum读取CSV列名和行数解析StorageTime计算采样率检查时间戳gap检查模态时间重叠生成样本manifest生成审计图实现无学习基线实现车辆模型实现评价指标跑置乱对照跑split对照保存预测结果生成失败样本图生成报告
```

## 每次问 GPTPro 应提供哪些材料

每次不要只问“下一步怎么办”，而要提供：

```
当前阶段目标已完成文件列表关键统计表异常样本比例代表性预测图固定失败样本图当前模型配置当前split方式当前指标表你希望GPTPro判断的问题候选决策方案
```

建议问题格式：

```
背景：当前阶段：已有证据：关键表格：代表图：我准备做的决策：请你审查：1. 是否存在泄漏风险？2. 是否可以进入下一阶段？3. 哪些结论不能下？4. 下一步最小实验是什么？
```

## GPTPro 回复如何归档

建议每次保存：

```
gptpro_reviews/YYYYMMDD_phaseXX_prompt.mdgptpro_reviews/YYYYMMDD_phaseXX_response.mdgptpro_reviews/YYYYMMDD_phaseXX_decision.mdgptpro_reviews/YYYYMMDD_phaseXX_action_items.md
```

并在 `decision.md` 中写清：

```
GPTPro建议了什么团队采纳了什么团队拒绝了什么拒绝原因对应代码commit对应数据版本对应结果文件
```

## 是否需要项目专用 skill

建议需要。

项目专用 skill 不应该只是“帮我训练模型”，而应该包含以下标准任务：

```
读取项目目录结构扫描原始CSV生成文件清单和checksum执行时间戳审计执行采样率审计执行模态重叠审计执行事件锚点审计生成样本manifest检查泄漏风险生成固定预测图计算物理评价指标执行风格置乱执行生理置乱生成阶段报告整理GPTPro提问材料归档GPTPro回复更新decision log
```

---

# 可直接转化为 Codex 长期目标的方案

## 1. 新长期目标名称

```
R2E-Steering：从原始数据重建方向盘事件响应预测与连续风格/生理增量证据链
```

## 2. 阶段划分

```
阶段0：旧流程冻结与重建准则阶段1：原始数据审计阶段2：事件锚点与样本manifest重建阶段3：无学习基线与纯车辆基线阶段4：连续驾驶风格有效性验证阶段5：生理信号角色验证与辅助任务阶段6：结构化轨迹模型阶段7：多假设/概率轨迹模型阶段8：EEG教师蒸馏与多模态强模型阶段9：证据链汇总、失败样本复盘、最终报告
```

## 3. 每阶段完成标准

### 阶段 0 完成标准

```
旧流程参考清单完成旧流程不可信清单完成旧失败样本类型整理完成新流程硬性原则完成
```

### 阶段 1 完成标准

```
所有原始CSV完成文件审计所有模态完成时间戳和采样率审计车辆/生理/脑电完成重叠和同步审计事件锚点完成合理性审计生理和脑电完成质量审计肌电动作泄漏审计完成泄漏风险报告完成
```

### 阶段 2 完成标准

```
samples_master.csv/jsonl完成每个样本可追溯到原始文件和行范围每个样本有事件锚点、输入窗口、标签窗口每个样本有模态质量和泄漏风险标记split_table完成dataset_version_card完成
```

### 阶段 3 完成标准

```
无学习基线完成纯车辆模型完成车辆+事件模型完成固定预测图完成多指标评价完成困难样本清单完成
```

### 阶段 4 完成标准

```
连续风格特征完成风格模型完成驾驶员ID对照完成风格置乱完成同被试/跨被试评估完成风格有效性结论完成
```

### 阶段 5 完成标准

```
ECG/EDA/EMG/RESP/EEG分别完成质量分层每个生理模态完成至少一个合理角色实验生理置乱完成生理质量分层完成生理有效/无效/证据不足结论完成
```

### 阶段 6 完成标准

```
响应方向/幅值/延迟/类型分解模型完成关键点+残差模型完成物理错误指标显著改善或明确失败原因
```

### 阶段 7 完成标准

```
多候选轨迹模型完成best-of-K和top-1同时报告概率和多样性报告完成大幅响应和多段修正表现完成
```

### 阶段 8 完成标准

```
EEG教师模型完成无EEG学生模型完成蒸馏对照完成EEG置乱完成推理输入版本明确
```

### 阶段 9 完成标准

```
最终证据链报告完成风格结论完成生理结论完成模型提升结论完成失败样本复盘完成可复现实验包完成
```

---

# 第一阶段具体任务清单

第一阶段只做原始数据审计，不训练模型。

## 任务 1：建立项目目录

建议：

```
R2E-Steering/  audit/  dataset_manifest/  figures/  results/  models/  scripts/  configs/  docs/  gptpro_reviews/
```

## 任务 2：扫描原始文件

扫描：

```
原始车辆数据/原始生理数据/原始脑电数据/
```

生成：

```
audit/file_inventory.csvaudit/file_checksums.csvaudit/file_read_errors.csv
```

## 任务 3：读取列名和基础统计

对每个 `.csv` 输出：

```
行数列数列名ID唯一值StorageTime范围缺失率重复行数
```

生成：

```
audit/raw_csv_schema_report.csv
```

## 任务 4：被试和模态匹配

生成：

```
audit/subject_session_inventory.csvaudit/modality_availability.csv
```

检查：

```
哪些被试有车辆无生理哪些被试有生理无车辆哪些被试有脑电无车辆哪些被试三模态都有
```

## 任务 5：时间戳审计

对每个文件检查：

```
StorageTime是否可解析是否单调递增重复时间戳比例负时间差比例最大gapmedian_dteffective_sampling_rate
```

生成：

```
audit/timestamp_audit.csvaudit/sampling_rate_audit.csvfigures/audit/sampling_rate_histograms/figures/audit/timestamp_gap_examples/
```

## 任务 6：模态重叠审计

按被试/session/trial 检查：

```
车辆时间范围生理时间范围脑电时间范围共同重叠范围重叠时长
```

生成：

```
audit/time_overlap_audit.csvfigures/audit/modality_overlap_examples/
```

## 任务 7：事件锚点初审

使用旧日志中的事件锚点作为候选，但不默认正确。

检查：

```
锚点是否落在车辆数据范围内锚点是否落在生理数据范围内锚点是否落在脑电数据范围内锚点前后方向盘是否已有变化锚点附近道路曲率/横向距离是否符合事件
```

生成：

```
audit/event_anchor_audit.csvfigures/audit/event_anchor_examples/
```

## 任务 8：预测窗口覆盖审计

尝试不同标签窗口，例如：

```
0-1s0-2s0-3s0-4s
```

统计：

```
主峰是否落入窗口尾段是否落入窗口反向修正是否落入窗口多段修正是否落入窗口
```

生成：

```
audit/prediction_window_coverage.csvfigures/audit/steering_response_window_examples/
```

## 任务 9：生理质量审计

对 ECG、EMG、EDA、RESP 生成质量分数。

输出：

```
audit/phys_signal_quality.csvfigures/audit/phys_quality_examples/
```

## 任务 10：脑电质量审计

输出：

```
audit/eeg_signal_quality.csvfigures/audit/eeg_artifact_examples/
```

## 任务 11：EMG 动作泄漏审计

计算事件附近：

```
EMG能量变化EMG onsetsteering onset二者相对时间
```

生成：

```
audit/emg_action_leakage_audit.csvfigures/audit/emg_vs_steering_onset_examples/
```

## 任务 12：泄漏风险报告

检查：

```
旧标准化方式旧基线校正方式旧风格提取方式旧生理特征提取方式旧split方式旧事件锚点来源
```

生成：

```
audit/leakage_risk_report.md
```

## 任务 13：第一阶段总结报告

生成：

```
audit/audit_summary.md
```

报告必须回答：

```
原始数据是否可用于重建？哪些被试可用？哪些模态可用？时间同步是否可信？事件锚点是否可信？标签窗口应多长？哪些样本应排除？哪些泄漏风险必须修复？是否可以进入样本manifest阶段？
```

---

# 第一阶段需要生成的文件和图

## 文件

```
audit/file_inventory.csvaudit/file_checksums.csvaudit/file_read_errors.csvaudit/raw_csv_schema_report.csvaudit/subject_session_inventory.csvaudit/modality_availability.csvaudit/timestamp_audit.csvaudit/sampling_rate_audit.csvaudit/time_overlap_audit.csvaudit/event_anchor_audit.csvaudit/prediction_window_coverage.csvaudit/phys_signal_quality.csvaudit/eeg_signal_quality.csvaudit/emg_action_leakage_audit.csvaudit/leakage_risk_report.mdaudit/audit_summary.md
```

## 图

```
figures/audit/sampling_rate_histograms/figures/audit/timestamp_gap_examples/figures/audit/modality_overlap_examples/figures/audit/event_anchor_examples/figures/audit/steering_response_window_examples/figures/audit/phys_quality_examples/figures/audit/eeg_artifact_examples/figures/audit/emg_vs_steering_onset_examples/
```

---

# 是否建议立即开始写代码

建议立即开始写代码，但只写**审计代码和 manifest 代码**。

不建议立即写训练代码。

第一批代码应该是：

```
scripts/scan_raw_files.pyscripts/audit_csv_schema.pyscripts/audit_timestamps.pyscripts/audit_sampling_rate.pyscripts/audit_modality_overlap.pyscripts/audit_event_anchors.pyscripts/audit_prediction_windows.pyscripts/audit_phys_quality.pyscripts/audit_eeg_quality.pyscripts/audit_emg_leakage.pyscripts/generate_audit_summary.py
```

训练相关代码等到 `samples_master.csv` 完成后再写。

---

# 是否建议先控制服务器开机

不建议第一步就依赖服务器。

建议顺序是：

1. **先在本地做文件扫描和小规模审计**；
2. 确认原始文件结构、列名、时间戳、ID 匹配；
3. 再决定是否开服务器做全量处理；
4. 服务器主要用于大规模重采样、样本构建、模型训练和多 seed 实验。

如果本地原始目录就存在文件缺失、时间戳混乱、事件锚点不可信，服务器先开也解决不了根本问题。

---

# 是否建议先做本地原始数据审计

强烈建议。

本地原始数据审计是整个新流程的第一优先级。

第一阶段目标不是得到好结果，而是回答：

```
这批原始数据能不能支持我们想证明的问题？如果能，哪些样本能用？如果不能，缺陷在哪里？
```

---

# 旧流程哪些结论可以作为参考

可以参考：

1. **旧失败样本类型**

例如方向错侧、大幅响应压小、反向修正抹平、尾段漂移。
2. **旧模型结果作为历史下限**

可作为旧流程 baseline，但不要当作新流程的正式对照。
3. **旧事件设定和道路信息**

可作为候选事件来源。
4. **旧困难样本清单**

可作为固定预测图集合。
5. **旧模型架构经验**

粗细双头、物理约束、多候选、生理融合等可以作为后续候选路线。
6. **旧日志中的异常记录**

对新审计很有价值。

---

# 旧流程哪些部分不能再默认相信

不能默认相信：

1. **旧对齐后数据一定正确**；
2. **旧事件锚点一定正确**；
3. **旧预测窗口一定覆盖完整响应**；
4. **旧标准化没有泄漏**；
5. **旧生理基线校正没有使用未来信息**；
6. **旧风格特征没有混入事件后标签**；
7. **旧训练/测试 split 没有同 trial 或同事件泄漏**；
8. **旧生理拼接方式能代表生理有效性**；
9. **旧 EEG 教师蒸馏失败就说明 EEG 无效**；
10. **旧模型 RMSE 接近就说明所有模型等价**；
11. **旧困难样本一定是模型问题，而不是锚点、窗口、同步或标签问题**。

---

# 最终建议

你们现在最正确的动作不是继续堆模型，而是启动：

```
R2E-Steering 阶段 1：原始数据审计
```

第一阶段完成前，不要宣称：

```
连续驾驶风格有效生理数据有效脑电教师无效肌电有效旧模型已达到上限
```

第一阶段完成后，如果数据质量、事件锚点、时间同步和窗口覆盖都能通过，再进入样本 manifest 和基线阶段。

只有当以下证据链完整时，才能得出最终结论：

```
原始数据可信事件锚点可信输入因果合法标签窗口合理split无泄漏车辆强基线稳定连续风格在强基线上有置乱可证的增量生理在强基线+风格上有置乱可证的增量物理指标改善，而不只是RMSE微降失败样本可解释
```

如果走完这条链后，生理数据没有稳定增益，也不是失败。那将是一个有价值的结论：

```
在当前数据质量、任务定义和可用输入条件下，尚不足以证明生理数据对事件后方向盘转角轨迹预测具有不可替代的增量价值。
```

这比继续堆模型后勉强解释“生理可能有效”要严谨得多。
