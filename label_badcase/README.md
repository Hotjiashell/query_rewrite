# Badcase 人工标注器

本地翻页式标注工具。前端使用原生 HTML、CSS、JavaScript；Python 标准库提供文件读取与实时保存服务，无需 npm 或额外 Python 依赖。

## 启动

在项目根目录执行（替换为你的真实文件路径）：

```bash
python label_badcase/server.py \
  --badcases results/badcases.json \
  --analysis results/badcase_analysis.json \
  --cases data/case_example.json \
  --output label_badcase/annotations.json
```

打开 <http://127.0.0.1:8765>。端口可用 `--port 8766` 修改。

用完整模拟数据体验：

```bash
python label_badcase/server.py --demo
```

模拟模式默认保存到 `label_badcase/demo_annotations.json`，不会污染正式标注文件。`demo_data.json` 中的内容均为虚构示例。

## 数据来源

| 输入 | 用途 |
| --- | --- |
| `--badcases` | `collect_badcases.py` 输出；提供对话、query、queries、GT ID、召回排名和标题 |
| `--analysis` | `analyze_badcases.py` 输出；提供模型筛选依据、各候选判断和理由 |
| `--cases` | 案例库 `case_id -> {case_name, text}`；补齐 GT 和召回案例正文 |

按 `sample_index` / `record_key` 关联模型分析记录，按 `case_id` 关联正文。只展示 rank 1–10 内的召回案例；GT 自身不会算作“其他正确案例”。重复 `sample_index` 会拒绝启动，避免误关联。模型分析缺失或失败时会明确展示，可以手动核验召回案例。

## 标注流程

1. 左侧阅读对话、query / queries、GT 标题和内容，以及 top-10 召回案例。
2. 右侧核验模型 `same=true` 的案例，每个案例填写“是 / 否”、理由和负责人。
3. 如模型漏选，可在左侧召回列表点击“补充核验”，再填写判断。模型否决、未选中、判断不完整的案例都可以补充。
4. 至少一个案例被人工确认“是”时，记录归为“存在其他正确案例”。
5. 没有其他正确案例时，填写错误类型、详细说明和分类负责人。
6. 点击“完成标注”。未填写完整的记录会保留为草稿；后续修改已完成记录会自动重新变为待完成。

支持筛选、对话/标题/ID搜索、页码跳转、上一条/下一条，以及非输入状态下的左右方向键翻页。数据集展示顺序和原始 badcase 文件一致。若当前记录完成后不再满足筛选条件，页面仍保留当前记录供确认，点击下一条可继续处理筛选结果中的下一条。

错误类型初始只提供“其他”，支持在“管理错误类型”中添加或删除类型。“其他”不可删除。删除已经被使用的类型时，旧标注自动转入“其他”，原类型写入详细说明，历史结论和负责人保留。负责人填写姓名。

## 保存与统计

- 文本编辑后约 450 ms 自动保存，选择判断后同样自动保存。
- 翻页、筛选和导出前等待当前草稿保存；失败时停止跳转，保留页面草稿并显示重试按钮。
- 每次保存都会将整个标注文件写到同目录临时文件，flush/fsync 后原子替换；不会修改输入数据。
- 页面显示写盘成功/失败状态。关闭未保存页面时浏览器会提醒。刷新、重启服务后读取已有输出文件继续标注。
- 自动保存不是“完成标注”。完成比例只统计人工点击完成且信息完整的记录。
- “人工确认有其他正确案例”统计至少已有一个“是”的记录，包含草稿；同一 badcase 多个“是”只计一次。
- “真实 badcase”与错误类型分布只统计已完成且没有任何“是”的记录。
- “模型认为有其他正确案例”单独统计模型建议，不作为人工结论。
- “模型认为没有其他正确案例”筛选模型分析成功、且没有非GT召回案例被模型判为 `same=true` 的记录；模型分析缺失或失败的记录不在此筛选中。该筛选不受人工标注结果影响。
- 页面支持导出 JSON 副本；正式结果持续保存在 `--output` 指定文件。

输出格式为 `badcase_annotations`，`annotations` 按记录键保存：

```json
{
  "artifact_type": "badcase_annotations",
  "schema_version": 1,
  "dataset_fingerprint": "...",
  "sources": {"badcases": "...", "analysis": "...", "cases": "..."},
  "revision": 1,
  "types_revision": 0,
  "updated_at": "...",
  "error_types": ["其他"],
  "annotations": {
    "12": {
      "record_key": "12",
      "revision": 1,
      "sample_index": 12,
      "reviews": {
        "KT000123": {"verdict": "yes", "reason": "能解决该用户问题", "owner": "张三"}
      },
      "error_type": "",
      "error_detail": "",
      "owner": "",
      "completed": true,
      "outcome": "alternative_found",
      "updated_at": "..."
    }
  },
  "summary": {
    "total": 100,
    "completed": 1,
    "drafts": 0,
    "alternative": 1,
    "genuine": 0,
    "model_positive": 20,
    "error_counts": {"其他": 0}
  }
}
```

`outcome` 为 `alternative_found` / `genuine_badcase` / `pending`。`verdict` 为 `yes` / `no` / 空字符串。`owner` 是真实 badcase 分类负责人，单个候选的负责人保存在 `reviews` 内。

工具支持在同一个服务进程下多人同时标注，监听 `0.0.0.0`。其他电脑通过 `http://服务电脑内网IP:8765` 访问。

- 每条 badcase 有独立 `revision`；不同人编辑不同记录互不阻塞。
- 错误类型有独立 `types_revision`，修改类型不会使无关记录产生冲突。删除类型会增加受影响记录的版本，避免旧页面把已删除类型写回来。
- 每 3 秒自动同步其他人的标注、筛选状态及统计；网络中断时会提示并自动重连。
- 当前记录没有本地修改且未在输入框编辑时，会自动显示远端最新标注。输入框聚焦时暂缓替换，离开后再同步。
- 如果当前存在未保存草稿，同步不会覆盖它；同一记录远端版本变化时会显示冲突面板。可查看最新标注，选择采用最新标注、确认以自己的草稿替换，或下载草稿。
- 未保存草稿会按数据集与记录备份到当前浏览器的 localStorage，刷新后恢复；保存成功后清理对应备份。备份只存在当前浏览器，同一浏览器多个标签共享备份空间，建议每位标注者只开一个工作标签。浏览器禁用/清除本地存储时无法恢复该备份。
- 旧标注文件自动补充每条记录的版本信息，不需要重新标注。服务升级后请重启服务并刷新所有标注页面。
- 单个服务进程用锁串行写盘，仍持续保存到同一 JSON 文件；不要同时启动多个服务进程写同一个输出文件。

输入数据或模型结果变化时，需指定新的输出文件，避免新旧数据的标注混用。当前未提供登录或记录认领，请团队按页码/样本分工，负责人仍填写姓名。

## 验证

```bash
python -m unittest discover -s label_badcase -p 'test_*.py'
```
