# Claude API Switch Record - 2026-04-08

## Scope

This change only updates the direct Claude Code API chain configured in:

- `C:\Users\Administrator\.claude\settings.json`

The following components were intentionally left unchanged:

- `C:\Users\Administrator\.codex\config.toml`
- `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1`
- `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
- `F:\data_set_process\data_process\.claude\commands\codex-run.md`
- `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`

## Provider Change

Switched Claude direct API base URL from:

- `https://xuedingtoken.com`

to:

- `https://api.guicore.com`

The auth token was also updated in user-level Claude settings.

## Preserved Settings

These settings were kept as-is:

- default model: `claude-opus-4-6`
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0`

## Verification Status

No probe request was sent during this switch.

Reason:

- the user asked to switch the chain itself
- previous turns showed sensitivity to token billing and channel stability
- verification can be done later with a minimal-cost request if needed

## Rollback Hint

If this provider needs to be rolled back later, first inspect:

- `C:\Users\Administrator\.claude\settings.json`
- `F:\data_set_process\data_process\reports\claude_api_chain_switch_20260402.md`
- `F:\data_set_process\data_process\reports\claude_api_chain_switch_20260408_guicore.md`

## 2026-04-08 Token Rotation

The user later requested a token-only update for the current `guicore` chain.

What changed:

- only `ANTHROPIC_AUTH_TOKEN` in `C:\Users\Administrator\.claude\settings.json`

What stayed the same:

- `ANTHROPIC_BASE_URL=https://api.guicore.com`
- default model `claude-opus-4-6`
- no changes to the old Codex bridge chain

For safety, the repository record still does not store the plaintext token.

## 2026-04-08 Second Token Rotation

The user requested another token-only replacement for the same `guicore` chain.

What changed:

- only `ANTHROPIC_AUTH_TOKEN` in `C:\Users\Administrator\.claude\settings.json`

What stayed the same:

- `ANTHROPIC_BASE_URL=https://api.guicore.com`
- default model `claude-opus-4-6`
- no changes to the old Codex bridge chain

No verification request was sent as part of this token replacement.

## 2026-04-08 Default Model Change

After the `guicore` chain became usable, the user requested changing the default Claude model from Opus to Sonnet.

What changed:

- `model` in `C:\Users\Administrator\.claude\settings.json`
- from `claude-opus-4-6`
- to `claude-sonnet-4-6`

What stayed the same:

- `ANTHROPIC_BASE_URL=https://api.guicore.com`
- current user token
- no changes to the old Codex bridge chain

No validation request was sent for this model-default change.

## 2026-04-08 Rollback To Original Codex Workflow

The user later requested rolling back to the original setup that relied on the existing Codex token / Codex bridge workflow.

What changed:

- cleared the user-level direct Claude API override in `C:\Users\Administrator\.claude\settings.json`

What this means:

- the temporary third-party direct Claude chain is no longer configured at the user level
- the original repository workflow based on:
  - `F:\data_set_process\data_process\.claude\commands\codex-run.md`
  - `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`
  - `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1`
  - `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
  - `C:\Users\Administrator\.codex\config.toml`
  remains in place

What stayed the same:

- no changes to the Codex provider config
- no changes to the Codex bridge scripts
- no plaintext token stored in repository logs

No verification request was sent during this rollback.
