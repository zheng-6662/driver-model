# Claude / Codex Windows 命令执行规范

日期：2026-04-09
适用范围：`F:/data_set_process/data_process` 仓库内的 Claude 与 Codex 协作、命令生成、脚本执行

## 目标

本规范用于减少以下问题：

- PowerShell / 终端中文乱码
- 长 `-Command` 内联命令的引号、数组、换行拼接错误
- Windows 下 `powershell.exe` 编码不一致导致的报错可读性差
- Claude / Codex 在仓库内给出不稳定的一次性命令

---

## 一、总原则

### 1. 复杂 PowerShell 不要内联

不推荐：

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "...很多步逻辑..."
```

推荐：

```bash
pwsh -NoProfile -File "F:/data_set_process/data_process/tmp/task.ps1"
```

规则：

- 只要命令包含多步逻辑、变量、数组、循环、压缩、批量复制、复杂引号，就不要继续写长 `-Command`
- 统一改成 `tmp/*.ps1` 脚本后再执行

### 2. 优先使用 `pwsh`

如果机器已安装 PowerShell 7，默认优先：

```bash
pwsh -NoProfile -File "F:/data_set_process/data_process/tmp/task.ps1"
```

仅在明确需要兼容旧环境时再使用：

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "..."
```

### 3. 文件处理优先 Python

对本仓库而言，涉及以下任务时，优先使用 Python 而不是长 PowerShell：

- 文件整理
- 批量复制/重命名
- 路径扫描
- 结果汇总
- 生成小型报告
- 数据检查辅助脚本

默认命令：

```bash
conda run -n predict2 python ...
```

### 4. Python 默认环境

本仓库中，Claude 与 Codex 生成 Python 命令时，默认优先：

```bash
conda run -n predict2 python ...
```

如果是模型训练、评估、诊断，默认优先考虑 GPU，除非用户明确要求 CPU 或脚本本身限制了设备。

---

## 二、编码与字符规则

### 1. PowerShell 脚本默认按 UTF-8 安全方式写

当确实需要写 `.ps1` 脚本时，脚本开头建议加：

```powershell
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

如脚本涉及文件输出，可补充：

```powershell
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
```

### 2. 禁止混入非 ASCII 命令符号

必须避免：

- 中文引号 `“ ”`
- 弯引号 `‘ ’`
- 长横线 `–`
- 全角空格
- 从网页或富文本复制来的特殊标点

尤其注意：

```powershell
Compress-Archive
```

中的 `-` 必须是普通 ASCII 连字符，不能是 `–`。

### 3. 路径统一双引号

推荐写法：

```powershell
"F:/data_set_process/data_process/..."
```

规则：

- 路径包含空格、中文、特殊字符时，必须加双引号
- 在 PowerShell 和 Python 命令里尽量保持一致写法
- 优先使用正斜杠，减少转义噪音

---

## 三、推荐执行模板

### 模板 A：简单查看类命令

适用于非常短的单条命令：

```bash
pwsh -NoProfile -Command "Get-ChildItem 'F:/data_set_process/data_process/reports'"
```

只适合：

- `Get-ChildItem`
- `Test-Path`
- 简短 `Get-Content`
- 很短的单步查询

不适合多步逻辑。

### 模板 B：多步 PowerShell 脚本

适用于：

- 压缩文件
- 批量复制
- 多路径整理
- 临时打包
- 一次性系统层操作

示例：

```powershell
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$zipPath = "F:/data_set_process/data_process/reports/output.zip"
$sourcePaths = @(
    "F:/data_set_process/data_process/CLAUDE.md",
    "F:/data_set_process/data_process/README.md"
)

$tempDir = "F:/data_set_process/data_process/tmp/zip_staging"
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

foreach ($p in $sourcePaths) {
    Copy-Item -Path $p -Destination $tempDir -Force
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path "$tempDir/*" -DestinationPath $zipPath -Force
Write-Output "ZIP created: $zipPath"
```

执行方式：

```bash
pwsh -NoProfile -File "F:/data_set_process/data_process/tmp/zip_task.ps1"
```

### 模板 C：Python 方式处理仓库任务

适用于：

- 数据扫描
- 结果汇总
- JSON / CSV / 文本处理
- 小工具脚本
- 与项目逻辑相关的辅助分析

推荐：

```bash
conda run -n predict2 python "F:/data_set_process/data_process/tools/xxx.py"
```

---

## 四、Claude / Codex 协作时的默认要求

当 Claude 给 Codex 写 handoff、execution brief、command suggestion 时，默认应显式包含以下要求：

```text
Windows 下执行命令时：
1) 复杂 PowerShell 一律写 tmp/*.ps1 后用 pwsh -File 执行；
2) 不要用长 -Command 内联多步逻辑；
3) 脚本开头统一 UTF-8 输出设置；
4) 文件处理优先 conda run -n predict2 python；
5) 路径统一用双引号，避免特殊引号和长横线。
```

说明：

- Claude 的记忆不会自动无损共享给所有独立 Codex 会话
- 因此涉及 Codex 时，应把这套规则写进 handoff，而不是只在对话里默认假设

---

## 五、常见错误与判断方法

### 1. `[Image #1]` 不是乱码

如果界面里出现：

- `[Image #1]`
- `[Image #2]`
- `[Image: source: ...]`

这表示系统在引用图片，不是终端编码故障。

### 2. `����` / `ò�` 通常是编码问题

如果报错文本出现：

- `����`
- `ò�`
- 中文完全不可读

通常说明：

- PowerShell 输出编码与读取编码不一致
- 或命令在工具链中被按错误编码解释

### 3. 同时有 `CommandNotFoundException` / `ParserError`

如果乱码同时伴随：

- `CommandNotFoundException`
- `ParserError`
- `= @(...)` 被当成命令
- 引号明显没有闭合

则通常不是单纯编码，而是：

- `-Command` 拼接失败
- 命令结构本身已经损坏

此时先修命令结构，再看编码。

---

## 六、在本仓库中的推荐优先级

| 场景 | 默认推荐 |
|---|---|
| 简单查看文件/目录 | 短 `pwsh -Command` 或常规 shell 命令 |
| 多步系统操作 | `pwsh -File tmp/*.ps1` |
| 文件整理 / 批处理 | `conda run -n predict2 python ...` 优先 |
| 数据处理 / 报告生成 | `conda run -n predict2 python ...` |
| 模型训练 / 评估 / 诊断 | `conda run -n predict2 python ...`，默认优先 GPU |
| 超长一次性命令 | 禁止继续内联，改脚本文件 |

---

## 七、最简默认口令

以后在本仓库里，Claude / Codex 可默认遵循以下口令：

```text
Windows 下执行命令时：
- 复杂 PowerShell 不写成长 -Command，改成 tmp/*.ps1 + pwsh -File；
- 文件处理优先 conda run -n predict2 python；
- 默认考虑 UTF-8 输出安全；
- 路径统一双引号；
- 避免特殊引号、长横线和富文本字符。
```

---

## 八、结论

对本仓库，最稳的实践是：

1. 多步 shell 逻辑改成 `tmp/*.ps1`
2. 用 `pwsh -File` 执行，而不是长 `-Command`
3. 文件与数据相关任务优先走 `conda run -n predict2 python ...`
4. 对 Codex 交接时，把这些规则显式写进 brief

这样能明显减少 Windows 下的乱码、引号炸裂和命令结构损坏。