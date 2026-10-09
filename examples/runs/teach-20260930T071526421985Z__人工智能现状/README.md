# 一次完整运行的结果快照

本目录保留了主题为“人工智能现状”的一次真实教学运行快照。该运行完成了固定四章报告的检索、证据建库、章节写作、审核、Markdown 组装和 PDF 导出。

- [Markdown 报告](reports/report.md)：便于查看全文、引用和章节结构。
- [PDF 报告](reports/report.pdf)：便于查看最终排版效果。
- [来源元数据清单](sources.json)：来源标识、标题、URL、Provider、核验状态和内容指纹；不含网页摘录或正文。
- `outlines/`、`chapters/`、`reviews/`：主控大纲、章节 Markdown 与引用映射、审核结果。
- `traces/`：主控事件和 A2A 交互的 JSONL 记录。
- `service_logs/`：Chroma、网络搜索 Agent、文档 Agent 的服务端日志。
- [Token 账本](token_usage.json)：本次运行的模型用量汇总。
- [公开快照清单](public_snapshot_manifest.json)：本次复制范围、排除范围与脱敏计数。

该快照用于帮助读者在克隆仓库后观察从检索到交付的完整运行过程。复制时已删除本机绝对路径，并对结构化数据中的凭证类字段进行脱敏；未保留第三方网页全文、检索摘录、证据正文、锁文件、虚拟环境或本地凭证文件。它只反映当时的主题、模型、公开资料和运行环境；重新运行时，检索来源、生成内容和审核结果会随时间与配置变化。
