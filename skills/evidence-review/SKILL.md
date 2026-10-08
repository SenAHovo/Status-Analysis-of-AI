---
name: evidence-review
description: 在主控审核单个章节草稿时，依据章节成果、引用映射与证据包形成可执行的批准、补证或修订决定；不用于代替网络搜索、正文写作或 A2A 状态管理。
---

# 章节证据审核

## 适用范围

仅用于主控 Agent 审核一个已交付的 `DocumentSectionResult`。输入包括当前草稿版本及其 `chapter_markdown` Artifact、citation map、EvidenceBundle、主控持有的章节目标与约束，以及已有补证轮次。

审核关注可追溯性和任务满足度。搜索结果摘要、模型自信程度或正文流畅度不能单独证明事实正确。检索、原文读取、任务状态、补证轮次、预算和 A2A 通信由程序管理。

## 审核方法

1. 对照章节目标、证据要求和写作约束，确认正文是否回答当前章节的问题，且没有越界替代其他章节。
2. 对重要事实、数据、机构行动、时间判断和关键结论，检查邻近的读者引用是否可经 citation map 回到当前 EvidenceBundle 的实际来源与支持性 Chunk。
3. 区分以下问题：
   - `missing_evidence`：当前材料无法支持必要事实、数据、比较或关键结论；
   - `unsupported_claim`：正文措辞强度超过相邻证据的支持范围；
   - `conflicting_sources`：来源之间在口径、时间、地区、定义或结论上存在未处理冲突；
   - `citation`：引用缺失、无关或不能定位；
   - `structure`：章节没有满足目标或论证顺序不清；
   - `style`：表达、术语或语气不符合任务要求。
4. 每项问题只绑定当前章节和当前草稿版本，写明可定位位置、严重性、可回查证据以及具体修改要求。不要生成泛泛的“完善内容”“增强质量”意见。
5. 根据问题决定动作：
   - 无实质问题：批准当前版本；
   - 缺少外部材料、关键数据或可核验来源：请求补证；
   - 现有证据已经足够，问题属于措辞、结构、引用放置或证据范围收窄：请求修订；
   - 证据问题和写作问题并存：先补证，再以补证后的材料修订。

## 补证判断

- 只为影响章节核心结论的缺口请求补证。soft 缺口可通过限定措辞、明确适用范围或保留不确定性解决时，不必发起搜索。
- 复用 `EvidenceGap.suggested_queries` 与 `preferred_source_types`；查询必须直接对应问题，不得把整章主题重新搜索一遍。
- 单章补证最多两轮。达到程序给定上限后，将仍未解决的问题标记为 `accepted_limit`，要求正文准确呈现限制，不再循环发起搜索。
- 搜索没有新增有效、可引用证据时，视为本轮补证未解决；不得仅凭重复来源将问题标记为已解决。

## 交付约束

输出必须便于程序组装为审核决定和 `ReviewIssue`。审核决定只能是 `approve`、`need_evidence` 或 `need_revision`；每个问题必须包含 `issue_id`、当前 `section_id`、当前 `draft_version`、问题类型、`low|medium|high` 严重性、可定位位置、证据引用和明确修改请求。`missing_evidence` 因缺少可用证据可以没有 `evidence_refs`，其余问题应提供可回查引用。`need_evidence` 至少绑定一个 `missing_evidence`、`unsupported_claim`、`conflicting_sources` 或 `citation` 问题；`need_revision` 至少绑定一个不需要外部资料的问题。

审核决定和问题清单是内部机器交付数据，最终由程序创建 `ReviewIssue` 并控制轮次；不要把 JSON、字段名、审核意见或证据缺口写入读者 Markdown。

不得改写正文、调用搜索工具、伪造已核验结论，或把内部审核信息写入读者 Markdown。
