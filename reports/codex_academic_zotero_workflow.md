# Codex 学术检索到 Zotero 工作流

当前工作流保持合规边界，只做元数据检索、开放可获取 PDF 尝试和 Zotero 入库：

1. 关键词检索：继续使用 OpenAlex。
2. DOI 导入：优先走结构化元数据源 `Crossref -> OpenAlex -> Unpaywall`。
3. URL 导入：继续使用 Zotero `translation-server`，如果能解析出 DOI，再用 Crossref/OpenAlex 做结构化补强。
4. 入库：通过 Zotero Web API 创建条目，并且只在 PDF 通过基础校验后才上传附件。

## 合规边界

- 不做付费墙、验证码、登录态或鉴权绕过。
- 不做 Google Scholar / CNKI 的非法全文抓取。
- PDF 仅尝试开放、合法可访问链接。
- 中文站点增强建议继续依赖 Zotero/Jasminum 生态，而不是单独写站点抓取器。

## DOI 导入的新优先级

`import-doi` 现在不再把 translator 作为第一元数据来源：

1. 先从 Crossref 拉 DOI 的核心书目信息。
2. 再用 OpenAlex 补充开放获取位置、摘要、引用计数等信息。
3. 若配置了 `unpaywall_email`，再用 Unpaywall 提供 OA 链接补充。
4. 如果本地有 `translation-server`，只把它当作 DOI 落地页增强或最终回退，不再是首选路径。

这意味着 DOI 导入在没有 translator 的情况下也能完成结构化元数据入库，只是少一层页面翻译补强。

## PDF 校验

`--download-pdf` 现在比之前更严格：

- 至少检查 `Content-Type` 或 PDF 文件签名。
- 拒绝过小或异常大的文件。
- 在附件 URL、`Content-Disposition` 文件名、落地页标题之间做简单标题/DOI 关联校验。
- 只有通过这些校验的文件才会作为 Zotero 附件上传。

这仍然是轻量校验，不等价于全文语义验证，但比单纯“下载到一个二进制文件就上传”更可靠。

## Zotero 去重规则

导入前会按以下顺序检查已有条目：

1. DOI 或强标识匹配，例如条目 URL 精确匹配。
2. 标题标准化后的精确匹配。
3. 弱匹配：标题相似度 + 第一作者 + 年份同时满足。

命中去重时，CLI 返回结果里会包含 `dedupe_reason`。

## 结构化日志

脚本现在会把关键决策写到 `stderr` 的 JSON 日志中，包括：

- 元数据源命中、缺失、失败。
- DOI 导入是否用了 translator 补强或回退。
- PDF 候选的尝试、拒绝原因、最终选中来源。
- 去重命中原因。
- 导入完成或命令失败。

标准输出 `stdout` 仍然保留机器可读结果 JSON，方便继续串联脚本。

## 初始化

1. 复制 `startup/academic_zotero_config.example.json` 为 `startup/academic_zotero_config.json`
2. 填入 Zotero `library_id` 和 `api_key`
3. 可选配置：
   - `openalex_mailto`
   - `unpaywall_email`
   - `translation_server_url`

说明：

- `translation_server_url` 对 `import-url` 是必需的。
- 对 `import-doi` 来说它现在是可选增强项，不再是首要依赖。

## 启动 translation-server

```powershell
powershell -ExecutionPolicy Bypass -File .\startup\start_zotero_translation_server.ps1
```

默认地址仍是 `http://127.0.0.1:1969`。

## 使用示例

### 搜索

```powershell
python .\tools\academic_search_to_zotero.py search "driver fatigue EEG" --limit 5
```

### 按关键词导入

```powershell
python .\tools\academic_search_to_zotero.py import-query "driver fatigue EEG" --limit 5 --pick 1 --collection "研究生论文/自动导入" --download-pdf
```

### 按 DOI 导入

```powershell
python .\tools\academic_search_to_zotero.py import-doi 10.1016/j.aap.2024.107786 --collection "研究生论文/自动导入" --download-pdf
```

### 按 URL 导入

```powershell
python .\tools\academic_search_to_zotero.py import-url "https://example.com/article-page" --collection "研究生论文/自动导入" --download-pdf
```

如果页面对应多个 translator 候选，仍然需要追加 `--pick N`。
