# Claude API 链路切换记录（2026-04-02）

## 目的

本记录用于固定两套链路：

1. 旧链路：Claude 项目内通过 `codex-bridge` 间接调用本地 Codex，再由 Codex 走本地代理 API。
2. 新链路：Claude Code 直接通过用户级 `ANTHROPIC_*` 配置访问新购买的 Claude API。

本文档不保存任何明文密钥。

## 旧链路快照

### 1. 项目内允许调用 Codex 桥接脚本

- 项目本地设置文件：
  - `F:\data_set_process\data_process\.claude\settings.local.json`
- 其中显式允许：
  - `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1`
  - `D:\ClaudeCode\codex-bridge\run-codex.cmd`

### 2. 项目内的 Codex 委派命令

- `F:\data_set_process\data_process\.claude\commands\codex-run.md`
- `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`

这两类命令会把任务转交给：

- `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1`
- `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`

### 3. Codex 桥接的真实执行入口

- `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
- 其中调用：
  - `D:\ClaudeCode\codex-bridge\codex.exe`

### 4. Codex 当前 provider 配置

- 用户级 Codex 配置文件：
  - `C:\Users\Administrator\.codex\config.toml`
- 关键项：
  - `model_provider = "zx"`
  - `base_url = "http://localhost:8317/v1"`
  - `wire_api = "responses"`
  - `requires_openai_auth = true`

### 5. 旧链路的本质

旧链路不是 Claude Code 直接访问 Claude API，而是：

`Claude 项目命令 -> codex-bridge -> codex.exe -> localhost:8317/v1 -> 特殊渠道 provider`

## 新链路快照

### 1. 新链路目标

改为让 Claude Code 直接读取用户级配置：

- `C:\Users\Administrator\.claude\settings.json`

并通过下列环境变量访问新 API：

- `ANTHROPIC_AUTH_TOKEN`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_SMALL_FAST_MODEL`

### 2. 新链路当前策略

- 用户级 `Claude Code` 默认模型固定为：
  - `claude-3-5-haiku-20241022`
- 原因：
  - 新购买的这套渠道文档明确建议使用该小模型；
  - 这样切换后更容易先跑通，避免默认模型不兼容。

### 3. 本次切换不删除旧链路

本次只做“新增直连链路 + 保留旧桥接链路”，不做以下动作：

- 不删除 `codex-bridge`
- 不删除 `/codex-run`、`/codex-workflow`
- 不改 `C:\Users\Administrator\.codex\config.toml`

## 如何切回旧链路

如果后续要恢复到“Claude 项目内继续通过 Codex 特殊渠道工作”的状态，按下面理解即可：

1. 旧链路本身仍然保留在项目和 `D:\ClaudeCode\codex-bridge\` 下。
2. `C:\Users\Administrator\.codex\config.toml` 仍保留原来的 provider 配置，无需重建。
3. 项目内的 `/codex-run` 与 `/codex-workflow` 命令仍可继续作为旧桥接入口。
4. 如果希望 Claude Code 不再优先走新 API，可撤销或改写：
   - `C:\Users\Administrator\.claude\settings.json`

## 回滚时优先检查的文件

- `F:\data_set_process\data_process\.claude\settings.local.json`
- `F:\data_set_process\data_process\.claude\commands\codex-run.md`
- `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`
- `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1`
- `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
- `C:\Users\Administrator\.codex\config.toml`
- `C:\Users\Administrator\.claude\settings.json`

## 备注

- 仓库文档中不记录明文密钥。
- 如果后续更换供应商，只要优先看本文件，就能迅速判断当前到底走的是“Codex 桥接链路”还是“Claude 直连链路”。

## 本次验证结果

- 已执行最小验证命令：
  - `claude -p --setting-sources user --model claude-3-5-haiku-20241022 "请只回复：OK"`
- 验证结果：
  - Claude Code 已通过新链路返回 `OK`
- 说明：
  - 本次切换不是只写入配置，已经完成了一次实际请求验证。

## 更高模型兼容性实测

- 测试日期：
  - 2026-04-02
- 测试方式：
  - 直接通过 Claude Code 新链路逐个发起最小请求，要求模型只回复 `OK`
- 已实测通过的模型：
  - `claude-3-5-haiku-20241022`
  - `claude-sonnet-4-6`
  - `claude-opus-4-6`
  - `sonnet`
  - `opus`
- 附加确认：
  - 调试日志中已看到 `model=claude-sonnet-4-6`
  - 调试日志中已看到 `model=claude-opus-4-6`
- 当前结论：
  - 该渠道不只支持文档里写的 Haiku，至少还支持 Sonnet 和 Opus 档位。
- 当前建议：
  - 若以稳定和成本为主，默认模型仍保留 Haiku；
  - 若要更强能力，可在需要时显式切换到 `claude-sonnet-4-6`；
  - `claude-opus-4-6` 也可用，但更适合高复杂度任务时按需启用，而不建议先直接改成默认。

## 当前默认模型状态

- 在后续用户确认后，默认模型已从 Haiku 调整为：
  - `claude-sonnet-4-6`
- 当前用户级配置文件：
  - `C:\Users\Administrator\.claude\settings.json`
- 当前实际配置包含：
  - `ANTHROPIC_BASE_URL=https://aixj.vip`
  - `model=claude-sonnet-4-6`
- 已再次做不带 `--model` 的最小验证，并在调试日志中确认：
  - `tmp/claude_default_model_debug.log` 中出现 `model=claude-sonnet-4-6`

## 2026-04-03 渠道切换补充

- 用户在确认新接口可计费后，又要求把 Claude 直连渠道从上一版代理切换到另一家同类型代理。
- 本次只切换 `C:\Users\Administrator\.claude\settings.json` 中的 Claude 直连接口配置，不改动：
  - `C:\Users\Administrator\.codex\config.toml`
  - 项目内 `codex-bridge`
  - 当前默认模型设置
- 当前 Claude 直连配置已经调整为：
  - `ANTHROPIC_BASE_URL=https://xuedingtoken.com`
  - 额外加入：
    - `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`
    - `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
- 仓库文档中仍不记录明文密钥；这里只记录渠道切换方向与附加环境变量。

## 2026-04-05 回切补充

- 用户在实际使用中发现 `xuedingtoken.com` 渠道表现为“问题发出后长时间不返回”，因此决定回切到上一个已验证可计费、可正常返回的 Claude 直连渠道。
- 本次回切只恢复 Claude 用户级直连配置，不改动：
  - `C:\Users\Administrator\.codex\config.toml`
  - 项目内 `codex-bridge`
  - 当前默认模型 `claude-opus-4-6`
- 当前 Claude 直连配置已恢复为：
  - `ANTHROPIC_BASE_URL=https://aixj.vip`
- 本次未额外执行探针请求；回切目标是优先恢复稳定可用状态，避免再次产生测试计费。

## 2026-04-07 再次切回 xuedingtoken

- 用户在 2026-04-05 回切到 `aixj` 后，又明确要求把 Claude 直连渠道再次切回 `xuedingtoken`。
- 本次仍然只修改：
  - `C:\Users\Administrator\.claude\settings.json`
- 保持不变：
  - `C:\Users\Administrator\.codex\config.toml`
  - 项目内 `codex-bridge`
  - 默认模型 `claude-opus-4-6`
- 当前 Claude 直连配置再次调整为：
  - `ANTHROPIC_BASE_URL=https://xuedingtoken.com`
  - `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`
  - `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
- 本次未主动执行探针请求；用户当前需求是先切换渠道本身，而不是立刻做计费验证。
