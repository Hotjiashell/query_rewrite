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
| `prompts_file` | 抽取与反馈阶段共用的提示词文件，默认 `extract_knowledge/prompts.md` |
| `knowledge` | 只包含已有知识条目的独立文本文件，默认 `extract_knowledge/existing_knowledge.txt` |
| `cases` / `dialogues` | cases 为案例映射文件，按 GT ID 读取 case_name 和 text 正文；dialogues 为可选的对话补齐文件 |
| `output_dir` | 各阶段结果和检查点输出目录 |
| `concurrency` | 每阶段模型调用并发数，默认 16 |
| `timeout` / `retries` | 单次请求超时秒数、失败后重试次数 |
| `max_tokens` | 单次模型输出上限，默认 4096 |
| `keyword_batch_size` | 每个 GT 案例抽词时最多传入的对话数，默认 10；超出时随机抽样，相同输入可复现 |
| `precluster_similarity_threshold` | embedding 余弦相似度阈值，默认 0.85，范围 -1～1 |
| `min_cluster_size` | 模型融合后的去重 badcase 数阈值，默认 1；小于等于该值的簇不进入已有知识过滤和知识抽取，0 表示不丢弃 |
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

- `## keywords`：传 GT 案例标题、对话和 queries，不传案例正文，直接以 JSON 代码块返回单个关键词及 reason；复合短语也可作为关键词。无符合项时 keywords 可为空字符串。
- `## describe`：只传 GT 标题、正文、关键词，返回非空 desc 描述解释。
- `## merge`：只根据关键词判断同义分组。
- `## filter`：判断已有知识是否覆盖关键词。
- `## extract`：生成知识条目和触发条件。

请保留这五个抽取阶段的二级标题；开启反馈时还必须包含 `feedback_review`、`feedback_revise` 两个二级标题，每段保留唯一的 `{input}` 占位符。程序只替换该占位符，JSON 示例中的其他大括号不用转义。输出字段名称与类型由程序校验。extract 阶段现在仅需输出 `{"knowledge":"完整知识条目","reason":"抽取依据"}`，不再要求 applicable、keyword、explanation、trigger。可调整措辞、判断规则和示例。

通过配置的 `extract_knowledge.prompts_file` 可以切换不同版本的提示词文件，也支持 `--prompts-file` 覆盖。每次启动会加载一次提示词，修改后重新运行即可生效。即使开启 `resume`，修改过提示词的阶段也会自动使用新缓存键重新调用；其他阶段输入和提示词未变时继续复用。当前模板快照保存在输出目录的 `prompt_templates.json`，各调用的完整渲染提示词仍保存在 `calls/` 中。

## 流程与计数

1. 合并输入，以 **GT ID + 完整对话文本** 去重。重复记录合并 query 和来源，避免同一对话在多轮实验中反复失败导致计数膨胀。坏文件/坏记录单独记入 `input_issues.json`，不影响其他文件或记录。缺失对话或标题时可传 `--dialogues data/dialog_example.json`（按 call_sno 匹配）、`--cases data/case_example.json` 补齐。
2. 按 GT ID 聚类。每个 GT 只生成一个关键词抽取任务，各案例任务并行执行。关键词抽取传入标题、对话及 queries；对话超过 keyword_batch_size 时随机抽取指定数量，同样输入可复现。非空关键词进入独立 describe 阶段，仅传案例标题、正文、关键词生成描述；解释失败记录 partial，跳过该案例，不让无解释词条进入聚类。每个 GT 最多抽取一个关键词；无符合项时输出空字符串。多关键词响应校验失败并重试。程序不做 GT 标题字符串匹配校验，允许模型换一种说法。
3. 按关键词聚类，匹配时做 Unicode NFKC、大小写和空白归一化；**不自动合并别名或语义相近的不同词**。一个关键词在某个 GT 案例中被发现后，聚入该 GT 的全部去重对话。一条对话可以属于多个关键词簇。
4. embedding 预聚类和模型融合后，按记录并集重新去重计数，再丢弃去重 badcase 对话数 `count <= min_cluster_size` 的关键词簇（默认阈值 1，即至少 2 条才保留）。这里不是按不同 GT 数 `gt_case_count` 判断。通过数量阈值的簇再并行与全部已有知识比较，由模型判断别名、缩写等覆盖关系。已覆盖的不进入抽取；过滤失败的单独保存，不能当成“未覆盖”。
5. 对保留簇并行提炼知识。每簇最多 `--max-examples` 条（1～10，默认 10），按 GT 轮询优先覆盖不同 GT。extract 允许先分析再返回 JSON 代码块，必须包含非空 knowledge；reason 为可选字段，兼容旧输出。
6. 最终按每个关键词簇的**去重对话数**降序排列，相同数量按关键词稳定排序。JSON 同时保留 `gt_case_count`（不同 GT 案例数量）。计数用完整簇，不受最多十条抽样影响。

## 相似关键词融合

流程顺序为：GT 分组抽词 → 独立生成解释 → 按词聚类 → embedding 语义预聚类 → 模型同义分组 → 合并记录并去重计数 → 数量阈值过滤 → 已有知识过滤 → 抽取知识。

预聚类现在只使用 embedding 语义相似度，不再使用编辑距离、Dice、短词阈值或字符串长度过滤。将“关键词：描述解释”完整文本发送到独立的 OpenAI-compatible embedding 服务（base_url 为 API 根地址，SDK 请求其 /embeddings 接口）。向量按余弦相似度比较，达到 `extract_knowledge.precluster_similarity_threshold`（默认 0.85）时连边，连通分量形成候选组，再交给 merge 模型判断。阈值越低候选越宽；不同 embedding 模型的分数分布不同，建议查看过程文件后调整。

在 config.json 中配置独立的顶层 embedding 节：

```json
"embedding": {
  "base_url": "https://your-embedding-service.example/v1",
  "model_name": "your-embedding-model",
  "api_key_env": "EMBEDDING_API_KEY",
  "batch_size": 32,
  "concurrency": 4,
  "timeout": 120,
  "retries": 2
}
```

可直接配置 `api_key`，否则读取 api_key_env 指定的环境变量，未提供时回退至 llm 密钥；无需鉴权的本地服务可填写任意非空 api_key。base_url/model_name 未填写时读取 EMBEDDING_BASE_URL / EMBEDDING_MODEL。配置必须提供 embedding 地址及模型名。旧的三个 precluster 字符串阈值配置已删除，不再接受。

embedding 分批并行、显示批次进度，检查返回索引、维度和有限非零向量，失败退避重试。开启 resume 复用成功批次。`calls/embedding/` 保存关键词、无密钥配置、向量及各次失败类型，`04_embeddings.json` 汇总向量和失败批次。失败批次的词不参与语义预聚类，保留原簇进入后续处理，并将 summary 的 partial 设为 true、退出码设为 1；恢复运行可重试。相似度边和候选组仍写入 04_merge_candidates.json。向量两两比较时间复杂度 O(N²D)，适合中小规模关键词集合。

模型只收到 `{"keywords":["华为wifi：企业无线网络","华为wi-fi：内部无线接入服务"]}`，不会收到预聚类来源、相似度或“这些可能是一组”的提示。提示词可在 prompts.md 的 merge 段编辑。分组必须精确覆盖全部输入词，每词一次，代表词选自组内；模型可把整个列表拆成多个组。校验失败会重试，最终调用失败则保留该候选组的所有原簇，summary 标记 partial，支持恢复重试。

融合结果保留 aliases，已有知识过滤和知识抽取都能看到这些名称。记录按 ID 取并集，多个别名共享的对话只计一次，合并后重新排序。默认数量阈值 1 在模型融合后执行：单样本原始簇可参与融合，融合后去重对话数仍不超过 1 的簇才丢弃。

新增过程文件：`04_merge_candidates.json` 保存候选组及相似度边，`04_merge_results.json` 保存模型判断，`04_merged_keyword_clusters.json` 保存融合后的簇和别名。融合调用并行运行、显示进度、重试并保存检查点。keywords 和 merge 提示词允许先输出分析再给出 JSON 代码块，解析器读取最后一个 JSON 代码块作为最终分组；原始完整响应保存在 calls 中。接口 enable_thinking 仍为 false，与通过提示词先输出分析的方式独立。

## 模型输入字段

模型调用前统一按白名单构造输入，避免将过程管理字段传入模型：

| 阶段 | 实际传入内容 |
| --- | --- |
| keywords | 顶层 gt_case_title；records 每条只有 chat_content、queries，不传案例正文 |
| describe | 仅 gt_case_title、gt_case_content、keyword |
| merge | keywords 词条列表，每项是“关键词：描述解释”，无案例、对话、ID、相似度或候选分组说明 |
| filter | keyword（关键词：描述解释）、desc、aliases、concepts、existing_knowledge |
| extract | keyword、desc、aliases、concepts、examples 数组，每条只有 chat_content、gt_case_title、queries |

embedding 输入只含“关键词：描述解释”。记录 ID、GT ID、call_sno、cluster_id、sources、文件路径、记录序号等仅保留在本地记录和聚类文件，用于去重、计数和追溯。calls 中的 input 和 prompt 记录的是实际发给模型的精简内容。缓存键也基于精简后的提示词，来源路径等元数据变化不再导致重复调用。旧的含元数据请求与新的精简请求具有不同缓存键，首次运行会重新调用相关任务。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `01_records.json` / `input_issues.json` | 归一化对话、合并后的 query、全部来源，以及输入问题 |
| `existing_knowledge.json` | 本次已有知识快照 |
| `02_gt_clusters.json` | GT 聚类及全部记录 |
| `02_keyword_tasks.json` | 每个 GT 的关键词抽取任务及抽中的对话 |
| `03_keyword_results.json` | 关键词阶段调用结果 |
| `03_description_tasks.json` / `03_description_results.json` | 独立描述解释任务和结果，调用详情保存在 calls/describe/ |
| `04_keyword_clusters.json` / `rejected_keywords.json` | 关键词簇、计数、依据；rejected_keywords.json 为兼容旧产物保留，目前为空列表 |
| `low_frequency_clusters.json` / `04_eligible_keyword_clusters.json` | 模型融合后的低频丢弃簇及原因、通过数量阈值的融合簇；04_keyword_clusters.json 和 04_merged_keyword_clusters.json 分别保留完整原始簇和融合簇 |
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

## 人工知识标注界面

本地界面无需安装前端依赖，也不调用模型。指定一个已有 run 输出目录：

```bash
python3 -m extract_knowledge.annotate extract_knowledge/outputs/run_001
```

省略目录时，默认使用 `extract_knowledge/config.json` 的 `extract_knowledge.output_dir`：

```bash
python3 -m extract_knowledge.annotate
```

在浏览器打开 **http://127.0.0.1:8765**。端口冲突时可加 `--port 8766`。服务只监听本机，Ctrl+C 停止。也可直接运行 `python3 extract_knowledge/annotate.py <run目录>`。显式传入的相对目录按当前工作目录解释。

- 左侧“确定知识”以该 run 的 `existing_knowledge.json` 快照为初始知识，每条可直接编辑、移除，也可手动新增；支持搜索。
- 右侧每页展示一个模型生成的知识条目，包含关键词、别名、聚类数量、抽取依据及模型原文。待加入编辑框默认使用模型输出，人工修改后点击“加入确定知识”。同一模型条目重复点击会更新原有确定条目，不重复插入。
- 下方展示该聚类全部来源案例标题、query 和完整对话，并标记最终模型总结使用的样本。可展开/折叠对话。来源从 `06_retained_clusters.json` 关联读取，并兼容其他聚类文件；缺失来源文件时会提示。
- 支持上一条、下一条、页码跳转、下一条未处理；不需要的模型条目可标记“暂不加入”。编辑草稿、确定知识及当前页码自动保存到 run 目录的 `annotation_state.json`，刷新或重启可继续。
- 点击“导出确定知识”，将保存当前编辑内容，并在 run 目录输出 **final_knowledge.txt**（全部确定知识的编号文本）和 **final_knowledge.json**（含条目来源及标注版本）。文本可直接用于下一轮配置的 knowledge 文件。

导出包含初始知识及人工加入/新增的知识，只导出当前确定列表，未加入的草稿不会混入；不会修改原 `knowledge.json` 或全局 existing_knowledge.txt。导出可重复点击，会更新同名结果文件。保存失败会显示错误，可点击顶部保存状态重试；有未保存修改时关闭页面会提醒。多个页面同时修改会检查版本并拒绝覆盖，提示刷新。若重新跑抽取导致 knowledge.json 或已有知识快照改变，旧标注状态会拒绝加载，请使用新的 run 目录，或先备份移走旧 annotation_state.json。

离线验证：

```bash
python3 -m unittest extract_knowledge.test_annotation extract_knowledge.test_pipeline
node extract_knowledge/annotation/test_ui.cjs
```

前端逻辑测试使用最小 DOM 验证编辑、加入、防重复、翻页、跳过和导出；它不替代真实浏览器的视觉检查。


### 案例正文与描述解释

关键词抽取输入新增 gt_case_content，优先按 GT ID 从 cases 文件的 text 字段读取，也可使用 badcase 自带的 gt_case_content。正文缺失的记录写入 input_issues.json 并跳过，避免无正文推断；请把配置中的 cases 改成实际完整案例库，示例库只用于展示格式。正文不截断，仅用于关键词抽取，后续 extract 样本仍保留原先的对话、标题和 query。

关键词阶段结果为 `{"keywords":"关键词","reason":"依据"}`；独立 describe 阶段结果为 `{"desc":"描述解释"}`，desc 缺失或为空会重试。两个阶段各自保存检查点并可恢复。同名关键词仍先聚合计数，描述选择关联有效去重对话数最多的 GT 案例所产出的那一份（按抽样前的完整关联对话数比较）；并列最多时随机选一份，使用固定种子保证相同输入可复现，不再拼接。各份描述及其对话数仍保存在 keyword_evidence 中，description_source 记录最终选用的案例及对话数，得到 concept=`关键词：描述解释`。同名多义词目前仍共享一个初始簇，不按含义拆分。

embedding、候选边、merge 输入及分组校验均使用完整 concept；模型必须原样返回完整词条，不可只返回关键词。融合后 keyword / desc 来自模型选择的代表词条，aliases 保留纯关键词，concepts 保存全部参与融合的描述词条。filter 和 extract 可看到 desc 与 concepts，knowledge.json 及人工审核页面也保留并展示描述。来源案例正文可在审核页面展开查看。数量过滤位置和去重计数规则保持不变。

## 根据知识生成评估提示词

```bash
python3 -m extract_knowledge.build_eval_prompt \
  extract_knowledge/outputs/run_001 \
  extract_knowledge/eval_prompt_template.txt \
  20
```

三个参数依次为 run 输出目录、模板路径、新知识数量 k。读取 run 中 `existing_knowledge.json` 的 content 快照作为全部已有知识，再按 `knowledge.json` 的原有顺序追加前 k 条 knowledge，统一编号成每行 `1. 内容`，替换模板中的 `{knowledge}`，写入 run 目录的 `final_prompt.txt`。不会读取人工标注的 final_knowledge，也不会读取当前可能已修改的 existing_knowledge.txt。原有编号会去除后重排；已有知识每个非空行视为一条。

k 必须是非负整数；k=0 时只放已有知识，超过新知识总数时使用全部新知识。模板中的其他大括号保持原样。输入无效时输出错误并返回退出码 2，不覆盖已有 final_prompt.txt；成功时原子替换该文件。

## 抽取后双重反馈验证

当前示例配置开启顶层 `feedback.enabled=true`。旧配置未提供该节时关闭。开启后会实际调用检索服务，需要安装根目录 requirements.txt 中的依赖并确保检索服务可访问。

每条参与反馈的知识使用最终保留簇的全部去重对话测试，不受 max_examples 限制。query 模型只看到已有知识、本条候选知识和当前对话，不看到 GT 标题、正文、历史 query 或任何记录编号。query 响应复用 gen_query.extract_queries 解析（兼容 query 字符串或数组）；query 生成执行复用 evaluate.QueryGenerationRunner，检索复用 evaluate.RetrievalEvaluator，支持 round_robin 和 score 融合，检查 GT 是否命中融合后的 Top K。GT ID 只在本地匹配。

`feedback.max_entries` 限制参与反馈的知识数量：默认 `0` 表示全部，正整数 N 表示只处理 `knowledge_initial.json` 中排名前 N 条。当前顺序按最终簇的去重对话数从多到少排列，同数时沿用原顺序；这是成功抽取的知识排名。限制同时作用于评审、query 生成、检索和反馈修改，前 N 条未通过时不从后面递补。它与控制检索结果排名的 `feedback.top_k` 是两个独立参数。可在配置中设置，例如：

```json
"feedback": {
  "enabled": true,
  "max_entries": 20
}
```

也可用 `python -m extract_knowledge --feedback-max-entries 20` 覆盖配置。超过实际条目数时处理全部；负数、布尔值和非整数无效。未选中的知识保留原文，写入最终 `knowledge.json/txt`，并在 JSON 中标记 `validation.skipped=true`、`validation.passed=null`，不能视为已通过验证。`feedback_skipped_knowledge.json` 单独保存这些条目；`summary.json` 的 `feedback_selected_count` 和 `feedback_skipped_count` 记录参与与跳过数量。主动限制范围本身不会令 `summary.partial=true`；选中条目未通过或调用失败仍按原规则标记 partial。

独立 review_llm 仅接收候选知识、聚类对话和 GT 案例标题，不传案例正文；评审适用条件是否归纳共同语义，明确拒绝枚举样本原话、现象、操作或设备。修改模型收到关键词、描述解释、当前知识、已有知识，以及最多10条对话的标题和 query；评审拒绝时附评审理由，检索未达标时附检索未命中的语义反馈（对话、目标标题、生成 query 和检索标题），不传案例 ID、记录 ID、来源路径、排名等管理字段。修改后重新执行两道验证，不能直接接受未验证的最后一次修改。

顶层 feedback 参数：enabled（开关）、max_rounds（最多验证轮数，默认3，含首次验证）、top_k（轮询融合后的检索截断数，默认10）、min_recall（通过所需的簇内对话命中比例，默认1.0，即全部命中）、retrieval_url（默认 search.py 地址）、timeout（单次检索超时秒数，默认30）、concurrency（簇内测试并发，默认继承主流程）。服务错误停止本条迭代，标记未完成，可恢复重试；检索每次恢复时重新执行，模型成功调用沿用原有缓存机制。

顶层 review_llm 可配置 base_url、model_name、api_key/api_key_env、temperature，未指定的模型连接字段继承 llm。要使用不同的评审模型，请显式填写 model_name，默认配置中的空字段会回退主模型。反馈提示词统一在 prompts.md 的 feedback_review 和 feedback_revise 段，可通过 prompts_file 切换；保留每段唯一的 {input} 占位符。

新增产物：knowledge_initial.json 保存原始抽取；08_feedback_results.json 保存参与反馈条目的每轮知识、评审、query、实际检索和命中结果；feedback_rejected_knowledge.json 保存未通过条目；feedback_prompt_templates.json 保存模板快照。最终 knowledge.json/txt 包含双重验证通过的知识，以及因 max_entries 限制跳过反馈的原始知识；参与验证但未通过或服务失败会令 summary.partial=true 并返回1。全部通过只代表在这些聚类样本上的验证结果，仍需独立样本评估泛化效果。

离线回归：`python3 -m unittest extract_knowledge.test_feedback extract_knowledge.test_pipeline`。


### 指定反馈评测提示词与已有知识

feedback.prompt_file 指向 query 生成提示词模板，必须有 `{knowledge}`，可包含 `{dialogue}`。缺少对话占位符时会追加“当前对话”。仅替换这两个占位符，JSON 示例的大括号不需要转义，也不会额外添加 query 指令覆盖模板。feedback.knowledge_file 指向已有知识文本；每轮将它与当前候选知识合并填入模板。该文件可独立于抽取前过滤使用的 extract_knowledge.knowledge；不填写时使用过滤阶段的知识路径。路径在配置中相对项目根目录，命令行相对当前目录。

```bash
python3 -m extract_knowledge \
  --feedback-prompt-file extract_knowledge/eval_prompt_template.txt \
  --feedback-knowledge-file extract_knowledge/existing_knowledge.txt
```

这两个参数在 feedback.enabled=true 时生效。feedback.fusion_method 可设为 round_robin（默认）或 score，与 evaluate.py 一致。feedback_evaluation_settings.json 保存模板、已有知识和融合配置快照，每轮 tests.evaluation 保留 evaluate.py 的 matched_rank、union_top_k、per_query_traces 和检索状态等字段。通过条件使用融合排名不超过 top_k，部分检索失败保留评测明细但不算本条验证通过。


反馈修改阶段与当前 feedback_revise 提示词对齐：输入字段为 keyword、desc、knowledge、existing_knowledge、examples、retrieval_failures，以及仅在评审拒绝时提供的 review。examples 每项仅含 chat_content、gt_case_title、queries；不传案例正文或管理编号。模型可先分析再输出 JSON 代码块，仅返回非空 knowledge 即可，reason 可选。修改后未返回 reason 时清除旧抽取依据，避免将旧依据误关联到新知识。所有抽取及反馈模板统一保存于 prompt_templates.json，反馈模板另外保留 feedback_prompt_templates.json 快照；模板变化自动改变对应阶段的缓存键。query 验证的外部模板仍通过 feedback.prompt_file 配置。


### 反馈执行顺序

每轮先调用 feedback_review。评审拒绝时跳过 query 生成和检索，直接将评审意见交给 feedback_revise；修改后从评审重新开始。只有评审通过才测试整个聚类，检索命中比例不足时带着检索失败反馈修改，再重新评审。评审调用失败停止本条迭代，不当成质量拒绝。max_rounds 仍包含所有评审轮次，到达上限不再生成无法验证的新版本。

跳过检索的轮次在 08_feedback_results.json 中记录 retrieval_status=skipped、tests=[]、recall=null，避免误认为命中率为零。此时修改输入的 examples.queries 使用原 badcase 的历史 query，retrieval_failures 为空；执行检索后则使用本轮实际生成的 query。

终端通过 stderr 实时打印 `[feedback]` 总进度，以及当前知识的评审轮次、评审通过/拒绝、query 生成与检索完成数、命中数、错误数、命中比例和修改状态。簇内检索按实际完成顺序更新进度，过程文件中的测试顺序仍与原对话顺序一致。


feedback_revise 的对话总预算为10条，examples 与 retrieval_failures 共享预算，不重复传入同一条对话。优先从本轮未命中的对话中按 GT 轮询选择，不足10条时再从剩余对话按 GT 轮询补齐。评审拒绝而未执行检索时，从全簇按 GT 轮询选最多10条，携带历史 query。existing_knowledge 为反馈评测使用的已有知识文件全文，knowledge 为当前待修改条目。评审通过后若仅检索不达标，输入中完全省略 review 字段。检索验证仍使用全簇对话，评审样本范围不受本次修改采样影响。


## 可选术语调研与关联聚类

顶层 research.enabled 默认 false。流程严格为：所有 GT 关键词抽取完成 → 汇总同名关键词的去重对话数 → 达到门槛的词进行调研生成定义 → 其他词按原 describe 生成定义 → 同名聚类 → embedding 与同义融合 → 一跳关联聚类 → 数量阈值/已有知识过滤 → 抽取 → 可选反馈。

```json
"research": {
  "enabled": true,
  "top_k": 5,
  "max_searches": 3,
  "min_dialogues": 10,
  "concurrency": 4,
  "timeout": 30
}
```

同名词按原有 NFKC、大小写、空白规则归一化，跨 GT 关联的记录 ID 取并集；仅当对话数严格大于 min_dialogues 时调研。例如10表示至少11条，0表示所有非空关键词都可调研。相同关键词只调研一次，结果共享给相关 GT。research 并发数与主流程独立，防止昂贵检索任务占满连接。

终端通过 stderr 实时打印 `[research]` 的待调研/跳过数量、关键词完成数和成功/失败数；每个关键词另外打印当前检索次数及 query、累计证据数、模型分析与继续检索/总结状态，失败时提示回退普通定义生成。并发关键词的日志带关键词标识。

第一次固定使用关键词检索，每次取 top_k 个案例的标题和正文。max_searches 是包括首次检索在内的总请求预算，至少1次。模型查看源 GT 标题/正文和累计去重的检索证据，可返回 search 动作生成新 query，或 summarize 动作给出 desc 与 relations；预算耗尽必须 summarize，违反约束会按原模型调用规则重试，最终失败不会越过预算。每次模型重试不增加检索请求。调研失败保留原 describe 路径产出可用定义，同时 summary.partial=true，不把缺少证据的调研当成功。恢复运行复用成功模型调用，但重新请求检索以获取当前证据。

所有调研模型提示词位于 prompts.md 的 research 段。输入只含当前关键词、筛选后的候选词名列表 known_keywords、案例 title/content、已经查询的文本和是否还可查询；候选词的 GT 片段、相似度、案例 ID、排名和来源路径不传入 research 模型。relations 每项为 source、target、description；source 必须为当前关键词，target 必须为本轮 known_keywords 中的词，description 为一句有证据支持的自然语言关联。相关不等于同义，调研不会自动新增关键词或伪造对话。

候选词来自本次运行所有成功抽取的去重关键词（包括未达到调研门槛的词），排除当前词。每轮检索后，优先选累计案例标题或正文中直接出现的词（NFKC、大小写、空白归一化，英文名称加字母数字边界匹配），再用 embedding 按相关性补充；直接匹配之间按输入顺序选择，embedding 按余弦相似度降序、同分按归一化词排序。每轮最多新增20个，同一关键词的整个调研累计最多40个，已选词持续保留、不替换；达到40个后不再新增。因此各轮传入的列表可能为20、40、40个，而不是每轮都重置为20个。没有足够候选时不强制凑满。

embedding 内部使用“候选关键词 + 对应源 GT 标题/正文”的上下文，与当前累计案例比较。正文按1000字符分段并保留标题，以最高片段相似度排序，覆盖长案例尾部及同词多 GT；候选 GT 片段只供 embedding 排序，research 模型仍只收到候选词名。复用顶层 embedding 配置、批处理、重试和缓存，调用文件位于 calls/research_embedding；单个关键词内已处理的证据向量跨轮复用。embedding 失败时保留已有候选并继续直接匹配，在调研结果中标记 candidate_embedding_failed，最终 summary.partial=true。

research 提示词采用 ReAct：每轮以累计 cases 和 previous_queries 为 Observation，模型先输出简短的 Thought 说明证据、缺口或行动依据，再在 Action 下输出唯一的 JSON 代码块（search 或 summarize）。程序解析最后一个 JSON 代码块执行动作；search 的检索结果累计到下一轮 cases，summarize 结束调研。模型不自行生成 Observation。这里的 Thought 是普通响应正文中的分析说明，沿用现有调用设置，不依赖服务端原生 thinking 模式；原始 Thought 和动作响应保存在 calls/research 的 attempts 中。证据充分可提前总结，预算耗尽时只能总结有依据的内容并说明关键限制。

语义同义融合完成后，将关联实体通过 aliases 映射回语义簇。无需双向确认，A→B 即可组成 A、B 的关联簇；每个源簇仅扩展它直接指向的邻居，不沿邻居继续搜索。因此 A→B、B→C 形成 AB 与 BC，不合并为传递闭包 ABC。多个直接关系 A→B、A→C 则形成 ABC。去除重复及被包含的冗余候选集合；关联簇可能重叠。每簇对话按 ID 取并集计数，不跨簇累加成唯一总数。

关联簇包含 relations、relation_members（各语义簇代表词及定义）、aliases、concepts。过滤模型必须判断整个关联簇是否已被覆盖，不能只检查一个代表词；抽取和反馈修改模型使用关联定义，总结一条包含多个必要术语及分别适用条件的知识，允许多句“当……时可考虑某关键词”。关闭模块时原有行为保持不变。

新增文件：03_research_results.json 保存各词检索 query、模型动作及结果，其中 candidate_selections 记录每轮新增词、入选方式、embedding 分数和累计候选列表；03_keyword_relations.json 保存结构化关系；03_ordinary_description_results.json 保存原 describe 调用；03_description_results.json 汇总最终定义；04_relation_clusters.json 保存一跳簇，04_merged_keyword_clusters.json 仍保存同义融合结果。calls/research 保存实际精简输入、证据和模型响应。knowledge.json 同时保留最终关系与成员定义。

验证命令：`python3 -m unittest extract_knowledge.test_research extract_knowledge.test_feedback extract_knowledge.test_pipeline tests.test_evaluate tests.test_gen_query`。
