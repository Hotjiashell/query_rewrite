# 同义词增强评估

在项目根目录运行，复用根目录 evaluate.py 的配置、模型、检索、指标以及所有 CLI 参数。新增代码仅位于本目录。

## 1. 生成后的 query 替换，再检索

已有生成结果时，无需再次调用模型：

```bash
python test_syonyms/evaluate_synonyms.py retrieve --synonym-mode replace --config config.json --input path/to/queries.json --output test_syonyms/replace_results.json
```

也可以一次完成生成与检索：

```bash
python test_syonyms/evaluate_synonyms.py all --synonym-mode replace --config config.json --query-output test_syonyms/replace_queries.json --retrieval-output test_syonyms/replace_results.json
```

生成产物保留模型原始 query；检索产物记录实际替换后的 query。对单 query 和多 query 均生效。所有匹配位置都替换，区分大小写，长词优先，单次处理，不级联替换。此模式单独运行 generate 时只生成原始 query。

## 2. 在生成提示词末尾追加对话命中的同义词

```bash
python test_syonyms/evaluate_synonyms.py all --synonym-mode prompt --config config.json --query-output test_syonyms/prompt_queries.json --retrieval-output test_syonyms/prompt_results.json
```

也可先运行 generate，再使用原 evaluate.py retrieve 检索生成产物。仅按原始对话匹配原词，每个原词列一次，按词表顺序组织为 `- 原词——同义词`。列表填入 prompt_template.txt 的 `{syonyms_list}`，再追加到完整 query 生成提示词末尾。无命中时不追加。此模式单独运行 retrieve 不改变已有 query。

`--synonyms PATH` 可指定词表，默认本目录 syonyms.txt；每行 `原词, 同义词`，支持中文逗号或制表符，忽略空行和 # 注释，同一原词出现多次时保留首次出现的同义词，忽略后续映射。`--synonym-prompt PATH` 可指定增强模板（需包含 `{syonyms_list}`，也兼容 `{}`）。`--prompt-file` 仍表示原有的完整自定义生成提示词，两者互不混用。

## 检索错误重跑与自动重试

支持原 evaluate.py 的 `--retry-errors RESULT_JSON` 参数（仅限 retrieve）。输入 query 必须与上轮使用的文件相同，增强模式和词表也应保持一致。

```bash
python test_syonyms/evaluate_synonyms.py retrieve --synonym-mode replace --config config.json --input test_syonyms/replace_queries.json --retry-errors test_syonyms/replace_results.json --output test_syonyms/replace_results.json
```

在 retrieve 或 all 命令中添加 `--internal-time 30`，本轮结束后如果还有检索错误，等待 30 秒再仅重跑错误样本，合并并保存结果，直到没有检索错误。不设置该参数时只执行一轮。多 query 部分失败也会重跑该样本。每轮复用生成的 query，不再调用模型。间隔必须为大于零的有限秒数，可用 Ctrl+C 停止；已完成轮次的结果保留在输出文件中。

```bash
python test_syonyms/evaluate_synonyms.py all --synonym-mode prompt --config config.json --query-output test_syonyms/prompt_queries.json --retrieval-output test_syonyms/prompt_results.json --internal-time 30
```
