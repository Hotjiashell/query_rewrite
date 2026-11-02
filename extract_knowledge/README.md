# 从 badcase 提取 query 增强知识

先编辑本目录的 [config.json](config.json)，填写 `llm.base_url`、`llm.model_name`，并把 `extract_knowledge.badcases` 改为实际 badcase 路径。密钥默认读取 `OPENAI_API_KEY`，也可以在 `llm.api_key` 中填写，或通过 `llm.api_key_env` 指定其他环境变量。base_url / model_name 留空时分别读取 OPENAI_BASE_URL / OPENAI_MODEL。

随后在项目根目录运行，无需传参数：

```bash
python3 -m extract_knowledge
```

也可以直接运行脚本（脚本会自动定位默认配置）：

```bash
python3 extract_knowledge/pipeline.py
```

配置的 `extract_knowledge` 节集中设置所有流程参数：

| 配置项 | 说明 |
| --- | --- |
| `badcases` | 输入路径数组，支持 `results/badcases*.json` 等通配符 |
| `prompts_file` | 三个阶段共用的提示词文件，默认 `extract_knowledge/prompts.md` |
| `knowledge` | 只包含已有知识条目的独立文本文件，默认 `extract_knowledge/existing_knowledge.txt` |
| `cases` / `dialogues` | 可选的标题和对话补齐文件，不需要时为 null |
| `output_dir` | 各阶段结果和检查点输出目录 |
| `concurrency` | 每阶段模型调用并发数，默认 16 |
| `timeout` / `retries` | 单次请求超时秒数、失败后重试次数 |
| `max_tokens` | 单次模型输出上限，默认 4096 |
| `keyword_batch_size` | GT 组提取关键词时每批对话数，默认 10 |
| `max_examples` | 最终抽取时每簇样本数，1～10 |
| `resume` | 是否复用成功调用并重试失败调用，示例已开启 |

**配置文件中的相对数据路径统一相对于项目根目录**，不受配置文件位置影响。默认配置为 `extract_knowledge/config.json`。可以保存多份配置，通过 `--config 路径` 切换：

```bash
python3 -m extract_knowledge --config extract_knowledge/config.production.json
```

命令行参数仍可覆盖配置，例如 `--concurrency 32`、`--no-resume`。优先级为：显式命令行参数 > 配置文件 > 默认值（模型连接参数再回退到环境变量）。命令行显式传入的相对路径仍相对于当前工作目录。

模型配置兼容 evaluate.py 的 `llm` 配置节；仅含 llm 的旧配置仍可配合命令行输入使用。依赖根目录 requirements.txt 的 openai 包，与 evaluate.py 一样发送 `chat_template_kwargs.enable_thinking=false`。密钥不写入过程文件。

输入必须是 collect_badcases.py 输出的 `badcase_report`。不会修改输入或原提示词，也不会调用检索服务。

## 已有知识独立维护

[existing_knowledge.txt](existing_knowledge.txt) 只保存已有知识条目，已从原 query 提示词复制 WebG、Outlook 两条。原 query 提示词保持不变；后续请在此文件维护过滤阶段要使用的已有知识，程序不会自动同步原提示词。

每条知识单独一行，可带编号。这里不放生成 query 的指令、JSON 示例或对话占位符。没有已有知识时可留空文件。配置项 `extract_knowledge.knowledge` 指向该文件，过滤模型收到的 `existing_knowledge` 就是此文件全文。

输出目录里的 `knowledge.txt` 只含本次新提取的知识，`knowledge.json` 保存对应依据与计数；已有知识文件不会被自动覆盖或追加。确认新条目后可手动合入已有知识文件，后续运行会据此过滤。

## 人工调整提示词

所有模型提示词集中在 [prompts.md](prompts.md) 中，可以直接编辑，无需修改 Python：

- `## keywords`：从 GT 标题提取遗漏关键词。
- `## filter`：判断已有知识是否覆盖关键词。
- `## extract`：生成知识条目和触发条件。

请保留这三个二级标题，每段保留唯一的 `{input}` 占位符。程序只替换该占位符，JSON 示例中的其他大括号不用转义。输出字段名称与类型由程序校验。extract 阶段现在仅需输出 `{"knowledge":"完整知识条目","reason":"抽取依据"}`，不再要求 applicable、keyword、explanation、trigger。可调整措辞、判断规则和示例。

通过配置的 `extract_knowledge.prompts_file` 可以切换不同版本的提示词文件，也支持 `--prompts-file` 覆盖。每次启动会加载一次提示词，修改后重新运行即可生效。即使开启 `resume`，修改过提示词的阶段也会自动使用新缓存键重新调用；其他阶段输入和提示词未变时继续复用。当前模板快照保存在输出目录的 `prompt_templates.json`，各调用的完整渲染提示词仍保存在 `calls/` 中。

## 流程与计数

1. 合并输入，以 **GT ID + 完整对话文本** 去重。重复记录合并 query 和来源，避免同一对话在多轮实验中反复失败导致计数膨胀。坏文件/坏记录单独记入 `input_issues.json`，不影响其他文件或记录。缺失对话或标题时可传 `--dialogues data/dialog_example.json`（按 call_sno 匹配）、`--cases data/case_example.json` 补齐。
2. 按 GT ID 聚类。同一 GT 的对话按 `--keyword-batch-size`（默认 10）拆批，所有批次并行提取标题中的软件名和内部术语，不丢弃大 GT 组后面的对话。一个 GT 可贡献多个关键词；模型可输出空数组。代码检查关键词确实出现在对应 GT 标题中。
3. 按关键词聚类，匹配时做 Unicode NFKC、大小写和空白归一化；**不自动合并别名或语义相近的不同词**。一个关键词在某个 GT 批次被发现后，聚入该 GT 所有标题含该词的去重对话。一条对话可以属于多个关键词簇。
4. 每个关键词并行与全部已有知识比较，由模型判断别名、缩写等覆盖关系。已覆盖的不进入抽取；过滤失败的单独保存，不能当成“未覆盖”。
5. 对保留簇并行提炼知识。每簇最多 `--max-examples` 条（1～10，默认 10），按 GT 轮询优先覆盖不同 GT。extract 返回完整 knowledge 条目及 reason 依据，缺失或为空时按失败重试。
6. 最终按每个关键词簇的**去重对话数**降序排列，相同数量按关键词稳定排序。JSON 同时保留 `gt_case_count`（不同 GT 案例数量）。计数用完整簇，不受最多十条抽样影响。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `01_records.json` / `input_issues.json` | 归一化对话、合并后的 query、全部来源，以及输入问题 |
| `existing_knowledge.json` | 本次已有知识快照 |
| `02_gt_clusters.json` | GT 聚类及全部记录 |
| `03_keyword_results.json` | 关键词阶段调用结果 |
| `04_keyword_clusters.json` / `rejected_keywords.json` | 关键词簇、计数、依据及未通过标题验证的词 |
| `05_filter_results.json` | 过滤判断、覆盖条目引用和理由 |
| `06_retained_clusters.json` | 保留簇 |
| `covered_clusters.json` / `filter_failed_clusters.json` | 已覆盖簇、过滤失败待重试簇 |
| `07_extraction_results.json` | 抽取输入样本、完整知识条目和抽取依据 |
| `calls/{stage}/{hash}.json` | 每次逻辑调用的完整 prompt、输入、历次原始响应、失败类型、模型配置和成功状态 |
| `knowledge.json` | 排序后的完整 knowledge、reason、关键词、簇大小和样本引用 |
| `knowledge.txt` | 同顺序的编号知识条目，直接使用模型返回的 knowledge，不拼接或重写正文 |
| `summary.json` | 各阶段统计、失败任务数和 partial 标记 |

过程文件包含原始客服对话，应按原始对话数据的权限管理。

## 并行、恢复与失败处理

每阶段内使用线程池，`--concurrency` 默认 16，阶段间有依赖屏障。终端显示已完成/总量、成功、失败和缓存数量。`--timeout` 默认每次调用 120 秒，`--retries` 默认失败后重试两次，指数退避加随机抖动；SDK 自带重试关闭，避免叠加。无效 JSON、字段类型错误和截断响应也会重试，`--max-tokens` 默认 4096。

每次调用完成或失败后原子保存检查点。相同命令加 `--resume` 会复用成功检查点并重试失败任务；缓存键包含输入、提示词、endpoint、模型、temperature 和 max_tokens。修改知识文件只会使依赖它的过滤调用失效；抽取输入未变时可继续复用抽取。不要在同一输出目录同时运行多个进程；更换数据集建议使用新目录，旧调用缓存会保留。

部分输入或模型任务失败时仍输出成功部分，返回码 **1** 并设置 `partial=true`，不能把这些结果当成全部完成。配置/无法读写等启动或保存错误返回 **2**；完整成功返回 **0**。中断后可通过检查点恢复（线程池中的已发出请求可能需要等待超时）。未发现新关键词或全部已覆盖时，结果为空是合法结果，相关过程文件可区分原因。

离线测试（不发真实模型请求）：

```bash
python3 -m unittest extract_knowledge.test_pipeline
python3 -m extract_knowledge --help
```
