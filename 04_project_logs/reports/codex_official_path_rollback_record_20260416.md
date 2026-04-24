# Codex Official Path Rollback Record - 2026-04-16

## Purpose

This document records the old custom Codex workflow before any future switch to the official ChatGPT/OpenAI path.

The goal is simple:

- preserve how the old path is wired
- preserve which files matter
- make rollback deterministic instead of memory-based

This record does not store any real secret token.

## What "the old way" currently means

The old path is not just "use a different login".

It is this chain:

1. Project-side Claude delegation commands call the local Codex bridge.
2. The bridge runs `D:\ClaudeCode\codex-bridge\codex.exe`.
3. Codex is configured to use provider `zx`.
4. Provider `zx` points at `http://localhost:8317/v1`.
5. Auth mode is currently stored as `apikey`.

Observed key values:

- `model_provider = "zx"`
- `base_url = "http://localhost:8317/v1"`
- `wire_api = "responses"`
- `requires_openai_auth = true`
- `auth_mode = "apikey"`

## Files that define the old path

### User-level Codex config

- `C:\Users\Administrator\.codex\config.toml`
- `C:\Users\Administrator\.codex\auth.json`

Important note:

- `C:\Users\Administrator\.codex` is a junction to `D:\ClaudeCode\codex-home`
- so these two path families are not independent live states on this machine
- editing one live tree affects the other view

### Bridge-home Codex config

- `D:\ClaudeCode\codex-home\config.toml`
- `D:\ClaudeCode\codex-home\auth.json`

### Bridge launchers

- `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
- `D:\ClaudeCode\codex-bridge\run-codex.cmd`

### Project-side delegation entrypoints

- `F:\data_set_process\data_process\.claude\commands\codex-run.md`
- `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`
- `F:\data_set_process\data_process\.claude\settings.local.json`

## Snapshot files saved in this repository

The following snapshots were added so the old path can be reconstructed later without opening live machine state first:

- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\user.codex.config.toml.snapshot`
- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\user.codex.auth.json.snapshot`
- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.codex-home.config.toml.snapshot`
- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.codex-home.auth.json.snapshot`
- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.invoke-codex.ps1.snapshot`
- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.run-codex.cmd.snapshot`
- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\project.codex-run.md.snapshot`
- `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\project.codex-workflow.md.snapshot`

## If you later switch to the official path

Switching to the official path should be treated as a login/request-chain change, not as repository cleanup.

What should remain preserved:

- `D:\ClaudeCode\codex-bridge\`
- the snapshot files listed above
- project-side `.claude` command files unless you intentionally replace them

What not to do during the switch:

- do not delete the bridge directory just because the official client works
- do not delete the `codex-home` config copy
- do not rely on the ChatGPT login screen alone as proof that all old-path dependencies are gone

## 2026-04-16 official activation note

On 2026-04-16, the active live Codex home was switched away from the custom provider path by:

- backing up the live config/auth into
  - `C:\Users\Administrator\.codex\switch_backups\official_switch_20260416_181358`
- removing the active `apikey` auth file from the live location
- replacing the live config with a minimal config that no longer contains:
  - `model_provider = "zx"`
  - `[model_providers.zx]`
  - `base_url = "http://localhost:8317/v1"`

Because `C:\Users\Administrator\.codex` is a junction to `D:\ClaudeCode\codex-home`, this official activation changed the single live Codex home exposed through both paths.

This document and the repository snapshots remain the rollback source of truth for restoring the old path later.

## How to roll back to the old bridge path later

Use this order.

1. Restore the old Codex provider files from the snapshot set.
   - Restore `C:\Users\Administrator\.codex\config.toml`
   - Restore `C:\Users\Administrator\.codex\auth.json`
   - Restore `D:\ClaudeCode\codex-home\config.toml`
   - Restore `D:\ClaudeCode\codex-home\auth.json`
2. Restore the bridge launchers if they were changed.
   - Restore `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
   - Restore `D:\ClaudeCode\codex-bridge\run-codex.cmd`
3. Restore the project-side delegation commands if they were changed.
   - Restore `F:\data_set_process\data_process\.claude\commands\codex-run.md`
   - Restore `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`
4. Re-check whether the active launch entrypoint is actually using the bridge path you expect.
5. Only after that, diagnose login or provider failures.

## Recommended rollback validation

After restoring the old path, validate in this order:

1. Confirm the config still points to `zx` and `http://localhost:8317/v1`.
2. Confirm the launch command still resolves into `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`.
3. Confirm the local provider at `localhost:8317` is reachable.
4. Only then test an actual Codex request.

## Known caution

During inspection on 2026-04-16, the bridge run logs already showed a failure mode on the old path:

- `unexpected status 502 Bad Gateway: Unknown error, url: http://localhost:8317/v1/responses`

That means a future rollback may require two separate checks:

- restore the old configuration
- verify the local provider behind `8317` is healthy

## Short summary

The old way is recoverable, but only if you preserve both:

- the provider/auth config
- the launch/delegation entrypoints

This record is meant to keep both pieces visible.
