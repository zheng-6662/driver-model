# 极限工况驾驶员反应研究

这个仓库现在按四块整理：

- `01_datasets/`：原始数据、预处理数据、事件数据集、补充采集数据、下载材料
- `02_code/`：当前维护代码、历史脚本、工具脚本、启动脚本、外部工作区
- `03_results/`：实验输出、临时运行目录、图表、备份、权重和分析产物
- `04_project_logs/`：项目日志、实验登记、状态说明、项目文档

## 当前应该从哪里开始看

如果你只想看现在还在维护、还应该继续改的代码，从这里开始：

1. `02_code/final_code/README.md`
2. `02_code/final_code/processing/`
3. `02_code/final_code/dataset/`
4. `02_code/final_code/model/training/`
5. `04_project_logs/references/current-state.md`

## 四块目录分别放什么

### 1. 数据集

`01_datasets/` 里是数据本体，不建议把代码和结果再混进去。

- `01_datasets/多模态数据/被试数据集合/`：当前训练和诊断会直接读的主数据集
- `01_datasets/多模态数据/Event_Dataset*`：事件级数据集
- `01_datasets/数据预处理/`：车辆 / 生理 / 脑电预处理产物
- `01_datasets/补充采集数据/`：补充采集与清洗数据
- `01_datasets/downloads/`：外部下载材料

### 2. 代码

`02_code/` 里是脚本和工具。

- `02_code/final_code/`：当前维护主线代码
- `02_code/tools/`：分析、导出、重算、可视化工具
- `02_code/startup/`：启动与辅助脚本
- `02_code/legacy_multimodal/`：从旧多模态工作区拆出来的历史代码
- `02_code/workspace/`：辅助代码工作区

### 3. 结果

`03_results/` 里是运行产生的东西。

- `03_results/tmp/`：临时运行目录与协议安全实验输出
- `03_results/output/`：文档、PDF、PPT、Visio 等导出
- `03_results/artifacts/`：归档结果与分析产物
- `03_results/多模态数据/程序运行结果/`：从旧工作区拆出的程序运行结果
- `03_results/backups/`：Nutstore / Zotero 等备份

### 4. 项目日志

`04_project_logs/` 里是你以后查“什么时候做了什么”的地方。

- `04_project_logs/reports/progress/`：实验登记表和日更日志
- `04_project_logs/reports/project_progress_master.md`：总进度主日志
- `04_project_logs/references/current-state.md`：当前项目状态锚点
- `04_project_logs/dataset_docs/`：从旧数据工作区拆出来的目录说明

## 现在默认的工作约定

- 任何实质性推进，先写 `04_project_logs/reports/project_progress_master.md`
- 当前维护代码默认从 `02_code/final_code/` 改，不去改结果目录里的脚本副本
- 当前实验状态先看 `04_project_logs/references/current-state.md`
- 如果脚本仍有硬编码路径，优先改成基于仓库根目录推导的新路径

## 兼容性说明

根目录里还保留了少量历史壳目录，例如 `datasetprocess/` 和空的 `startup/`。它们是为了不做破坏性删除而暂时保留，日常工作可以忽略。
