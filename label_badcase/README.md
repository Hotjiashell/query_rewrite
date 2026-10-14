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

## 尝试检索

每条 badcase 的右侧提供“尝试检索”。输入 query 并点击“尝试检索”，即可调用
`search.py` 的检索接口，查看返回案例的排名、标题、ID、分数和可展开的正文。
支持“填入原 query”和“填入 GT 标题”，填入后点击检索即可。

提示以当前记录的 GT 案例 ID 为准：命中时显示排名并高亮对应案例；未命中时
明确提示返回结果中没有该 ID；当前记录缺少 GT ID 时提示无法判断。
检查全部返回候选，不限于 Top-10。空结果与请求失败分别提示。

尝试结果按 badcase 分别保存在当前页面内，翻页后回来仍可查看，刷新后清空。
查询过程中可继续标注和翻页，返回结果只属于发起查询的记录。
尝试检索不会修改原 query、原召回列表、标注或统计。

默认使用 `search.py` 的接口地址，超时为30秒。需要 `requests` 依赖。
如需指定其他接口，在原启动命令后增加：

```bash
--retrieval-url http://host:8276/run_case_retrieval --retrieval-timeout 60
```

## 一键导入 GT 标题检索失败结果

先停止正在运行的标注器，在项目根目录执行：

```bash
python label_badcase/import_gt_retrieval.py \
  --report results/gt_title_retrieval_check.json \
  --annotations label_badcase/annotations.json
```

脚本从现有标注文件的 `sources` 自动读取 badcases、模型分析和案例库路径，
复用标注器的数据关联、指纹校验、标注校验和统计逻辑。按 GT ID 匹配结果，
同一 GT 对应的多条 badcase 分别导入，只处理模型分析成功且 Top-10 非 GT
候选没有 `same=true` 的记录，与页面“模型认为没有其他正确案例”筛选一致。
只有 GT ID 在标题检索结果中完全没有出现才导入；如果报告指定了 cutoff，
仅排名超出 cutoff、实际上已召回的记录会跳过。缺少标题和请求失败也不会导入。

导入后标为已完成，错误类型为“检索端问题”，详细说明为
“即使用GT案例标题检索也检索不到”，分类负责人为“唐锦康”。
错误类型和负责人自动加入下拉选项，统计和版本号同步更新。

默认跳过所有已有标注；添加 `--overwrite` 可更新符合条件的已有标注，
保留其候选核验结果，仍跳过包含“是”“不确定”或未判断的人工核验记录。
重复执行不会再次更新已经导入且内容一致的标注。
添加 `--dry-run` 可预览导入数量而不写文件。
写入前自动在同目录生成 `annotations.json.*.bak` 原文件备份。
导入后使用原来的命令重新启动标注器。

如果没有现成落盘文件，或原始数据路径已经变化，显式指定三个源文件：

```bash
python label_badcase/import_gt_retrieval.py \
  --report results/gt_title_retrieval_check.json \
  --annotations label_badcase/annotations.json \
  --badcases results/badcases.json \
  --analysis results/badcase_analysis.json \
  --cases data/case_summary.json
```

## 数据来源

| 输入 | 用途 |
| --- | --- |
| `--badcases` | `collect_badcases.py` 输出；提供对话、query、queries、GT ID、召回排名和标题 |
| `--analysis` | `analyze_badcases.py` 输出；提供模型筛选依据、各候选判断和理由 |
| `--cases` | 案例库 `case_id -> {case_name, text}`；补齐 GT 和召回案例正文 |

按 `sample_index` / `record_key` 关联模型分析记录，按 `case_id` 关联正文。只展示 rank 1–10 内的召回案例；GT 自身不会算作“其他正确案例”。重复 `sample_index` 会拒绝启动，避免误关联。模型分析缺失或失败时会明确展示，可以手动核验召回案例。

## 标注流程

1. 左侧阅读对话、query / queries、GT 标题和内容，以及 top-10 召回案例。
2. 右侧核验模型 `same=true` 的案例，每个案例填写“是 / 否”和负责人。选“是”且模型提供了理由时，人工理由可留空，表示接受模型理由；选“否”必须填写理由。人工补充且模型未判为正确的案例仍需填写理由。
3. 如模型漏选，可在左侧召回列表点击“补充核验”，再填写判断。模型否决、未选中、判断不完整的案例都可以补充。
4. 至少一个案例被人工确认“是”时，记录归为“存在其他正确案例”。
5. 没有其他正确案例时，填写错误类型、详细说明和分类负责人。
6. 点击“完成标注”。未填写完整的记录会保留为草稿；后续修改已完成记录会自动重新变为待完成。

支持筛选、对话/标题/ID搜索、页码跳转、上一条/下一条，以及非输入状态下的左右方向键翻页。数据集展示顺序和原始 badcase 文件一致。若当前记录完成后不再满足筛选条件，页面仍保留当前记录供确认，点击下一条可继续处理筛选结果中的下一条。

底部“下一个未完成”会先保存当前草稿并同步最新进度，然后在当前筛选和搜索范围内跳过已完成记录，寻找当前记录之后的未完成 badcase；到末尾后从头继续，不会跳回当前记录。范围内没有其他未完成记录时会提示并停留。保存失败或存在未解决冲突时不会跳转。

错误类型初始只提供“其他”，支持在“管理错误类型”中添加或删除类型。“其他”不可删除。删除已经被使用的类型时，旧标注自动转入“其他”，原类型写入详细说明，历史结论和负责人保留。负责人填写姓名。

核验负责人和分类负责人使用姓名下拉框。在页面顶部“负责人设置”中添加姓名后，两处都可重复选择。负责人名单保存在输出文件的 `owners` 数组中，并随自动同步更新到其他页面。相同姓名不会重复添加；旧标注中已填写的姓名会在启动时自动加入名单。添加负责人不会清空正在编辑的标注草稿。

在同一个设置窗口选择“我的默认负责人”，之后空白的负责人字段自动显示该姓名，编辑或完成标注时写入结果，手动补充的核验案例也会使用默认姓名。已有负责人不覆盖，仍可在单条记录下拉框里选择其他人。仅浏览记录或设置默认姓名不会创建标注或影响统计。默认姓名保存在当前浏览器的 localStorage，不发送为服务端全局设置，不同电脑/浏览器可分别配置；同一浏览器同一站点共用此偏好。刷新后保留，清除浏览器存储或更换访问地址后需重新设置。选择“不设置默认负责人”即可取消。

案例核验还支持“不确定”（`verdict: "uncertain"`），无需理由，但仍需负责人。所有待核验案例都已选择判断后，可完成本轮标注。没有“是”但存在“不确定”的记录归为 `outcome: "uncertain"`，不强制填写错误分类、不计入真实 badcase 或错误类型分布；可使用“待进一步确认（不确定）”筛选继续复核。存在“是”的记录仍归为已有其他正确案例。

## 保存与统计

- 文本编辑后约 450 ms 自动保存，选择判断后同样自动保存。
- 翻页、筛选和导出前等待当前草稿保存；失败时停止跳转，保留页面草稿并显示重试按钮。
- 每次保存都会将整个标注文件写到同目录临时文件，flush/fsync 后原子替换；不会修改输入数据。
- 页面显示写盘成功/失败状态。关闭未保存页面时浏览器会提醒。刷新、重启服务后读取已有输出文件继续标注。
- 自动保存不是“完成标注”。完成比例只统计人工点击完成且信息完整的记录。
- “人工确认有其他正确案例”统计至少已有一个“是”的记录，包含草稿；同一 badcase 多个“是”只计一次。
- “真实 badcase”与错误类型分布只统计已完成且没有任何“是”或“不确定”的记录。
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

`outcome` 为 `alternative_found` / `genuine_badcase` / `uncertain` / `pending`。`verdict` 为 `yes` / `no` / `uncertain` / 空字符串。`owner` 是真实 badcase 分类负责人，单个候选的负责人保存在 `reviews` 内。

新保存的核验结果还包含 `reason_source` 和 `effective_reason`：留空并接受模型理由时，`reason` 保留为空，`reason_source` 为 `model`，`effective_reason` 保存被接受的模型理由；填写人工理由时，`reason_source` 为 `human`，`effective_reason` 为人工理由。模型无肯定判断或理由缺失时，不能通过留空理由完成核验。

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
