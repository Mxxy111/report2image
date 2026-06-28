# 回顾性配对批量实验使用手册

**适用版本：** 开发迭代阶段  
**更新日期：** 2026-06-28  
**当前内置比较：** 同一病例 `UNGATED` 与 `GATED`

本手册说明如何配置 API、准备实验、进行小规模试跑、断点续跑、完成人工评价、导出数据和执行统计分析。当前项目尚未投稿或最终锁库；可以继续调整提示词、模型、参数和研究比较。这里的“锁定”只表示一个具体实验 ID 内部保持一致，不限制创建后续版本。

## 0. 先读这四条

1. 第一次配置或换模型时，先用开发实验 ID，例如 `dev-gptimage2-r1`，不要直接使用正式批次名称。
2. `--prepare-only` 会创建或校验实验锁，但不会调用模型 API；这是每个批次的第一步。
3. 同一个 `experiment-id` 只用于断点续跑同一套设置。模型、提示词、参考图、参数、代码或数据有意改变时，必须换新的 `experiment-id`。
4. 批量图片实验默认不生成患者理解度题目；门诊网页默认会生成题目，可在网页中关闭。

## 1. 运行结构

正式默认队列为：

- 肾癌 100 例；
- 前列腺癌 100 例；
- 尿路上皮癌 100 例；
- 每例运行 `UNGATED` 和 `GATED` 两种策略；
- 合计 300 例、600 个工作项。

策略定义：

- `UNGATED`：生成一次，不进行 AI 图片审查；
- `GATED`：生成后进行 AI 图片审查；失败时根据反馈修图，直至通过或达到 `--max-revisions`。

批量实验不生成患者理解度题目，因为题目不属于本轮图片准确性主要终点。门诊网页仍可生成题目。

## 2. 环境准备

在项目根目录打开 PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

如果项目自带虚拟环境缺少依赖，也可以使用当前 Python：

```powershell
python -m pip install -r requirements-dev.txt
```

查看批量入口全部参数：

```powershell
python scripts/experiments/run_paired_experiment.py --help
```

## 3. API 配置

批量入口与门诊网页使用同一套 provider/pipeline 配置：

- 项目模板：`settings/providers.json`
- 本机私密配置：`settings/local_providers.json`
- 环境变量：`.env`

`settings/local_providers.json` 和 `.env` 已被 `.gitignore` 排除，不应提交 API Key。

选择配置方式时可按下面判断：

| 场景 | 推荐方式 | 原因 |
|---|---|---|
| 只在本机网页和批量脚本中使用 | 网页“API 设置” | 最直观，会写入 `settings/local_providers.json` |
| 想用环境变量管理密钥 | `.env` | 适合固定 provider 和简单部署 |
| 私密配置不想放在项目目录 | `--local-provider-config` | 可把 Key 保存在 `D:\private\...` 等外部路径 |

兼容 OpenAI 的 Base URL 通常填写 provider 根地址，例如 `https://example.com/v1`，不要填到 `/images/generations`、`/chat/completions` 或 `/responses` 这类具体接口路径。

### 3.1 方法一：通过网页配置

启动网页：

```powershell
python run_web.py
```

打开 `http://127.0.0.1:8000`，点击右上角“API 设置”，填写：

- Provider 名称；
- API 类型：`openai` 或 `openai_compatible`；
- Base URL；
- API Key；
- 超时时间；
- 默认生图模型；
- 默认审查模型；
- 默认出题模型。

保存后写入 `settings/local_providers.json`。批量脚本会自动读取该文件。

### 3.2 方法二：通过 `.env` 配置

复制模板：

```powershell
Copy-Item .env.example .env
```

三任务使用独立 API 时：

```dotenv
PETCT_IMAGE_API_KEY=...
PETCT_IMAGE_BASE_URL=https://example-image-api/v1

PETCT_REVIEW_API_KEY=...
PETCT_REVIEW_BASE_URL=https://example-review-api/v1

PETCT_QUESTION_API_KEY=...
PETCT_QUESTION_BASE_URL=https://example-question-api/v1
```

全部使用 OpenAI 单账号时：

```dotenv
OPENAI_API_KEY=...
```

全部使用同一个兼容网关时：

```dotenv
CUSTOM_OPENAI_API_KEY=...
CUSTOM_OPENAI_BASE_URL=https://example-gateway/v1
```

批量图片研究只要求生图和审查配置可用，不要求出题 API。

### 3.3 方法三：指定其他配置文件

```powershell
python scripts/experiments/run_paired_experiment.py `
  --experiment-id dev-model-a `
  --seed 20260621 `
  --provider-config settings/providers.json `
  --local-provider-config D:\private\petct-providers.json `
  --prepare-only
```

不要把 API Key 直接作为命令行参数。命令可能保存在 PowerShell 历史、任务日志或系统进程列表中。

### 3.4 Provider 配置示例

兼容 OpenAI API 的本地配置可以使用：

```json
{
  "version": 1,
  "providers": [
    {
      "id": "my_gateway",
      "label": "我的兼容网关",
      "apiStyle": "openai_compatible",
      "apiKey": "仅保存在本地的密钥",
      "baseUrl": "https://example.com/v1",
      "timeoutSeconds": 300,
      "taskModels": {
        "image": "my-image-model",
        "review": "my-vision-model"
      }
    }
  ]
}
```

该 provider 可通过以下参数用于批量运行：

```powershell
--image-provider my_gateway --image-model my-image-model `
--review-provider my_gateway --review-model my-vision-model
```

## 4. 可配置参数

### 4.1 实验与数据参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--experiment-id` | 本批次唯一名称，必填 | 无 |
| `--dataset-root` | 三癌种评测数据根目录 | `data/derived/evaluation_dataset` |
| `--output-root` | 实验锁和审计日志根目录 | `runtime/experiments` |
| `--seed` | 病例顺序及病例内策略顺序随机种子，必填 | 无 |
| `--expected-per-cancer` | 每癌种要求的病例数 | `100` |
| `--max-items` | 本次进程最多运行多少个待处理工作项 | 不限制 |
| `--prepare-only` | 只检查队列并建立实验锁，不调用 API | 关闭 |

随机种子不控制生成图片像素。当前图像 API 不接受 seed，系统会明确记录 `providerSeedApplied=false`。

### 4.2 Pipeline 与 API 参数

| 参数 | 含义 |
|---|---|
| `--pipeline` | 使用 `settings/providers.json` 中的 pipeline ID |
| `--provider-config` | 指定公开 provider/pipeline 配置文件 |
| `--local-provider-config` | 指定含本地 Key 和 URL 的私密配置文件 |
| `--image-provider` | 临时覆盖生图 provider |
| `--image-model` | 临时覆盖生图模型 |
| `--review-provider` | 临时覆盖审查 provider |
| `--review-model` | 临时覆盖审查模型 |

项目内置 pipeline：

- `split_default`：生图、审查和出题分别使用独立 provider；
- `openai_single_key`：三任务共用 OpenAI Key；
- `custom_gateway_all`：三任务共用一个兼容网关。

### 4.3 生图参数

| 参数 | 示例 | 说明 |
|---|---|---|
| `--image-size` | `1536x1024` | 输出尺寸 |
| `--image-quality` | `high` | 图像质量 |
| `--image-output-format` | `png`、`jpeg`、`webp` | 输出格式 |
| `--image-background` | `opaque`、`transparent` | 背景方式，取决于模型支持 |
| `--image-compression` | `90` | JPEG/WebP 压缩质量 |
| `--image-input-fidelity` | `high` | 参考图/修图保真参数，取决于模型支持 |
| `--image-moderation` | `auto`、`low` | 审核强度，取决于模型支持 |
| `--reference-image` | `reference.png` | 固定的参考样式图 |
| `--max-revisions` | `1` | 门控失败后的最大修图次数 |

不同 provider 对参数的支持可能不同。兼容网关返回 400/404 时，应先用单病例或 `--max-items 1` 验证接口与模型参数。

## 5. 推荐运行流程

### 5.1 第一步：只准备，不调用 API

```powershell
python scripts/experiments/run_paired_experiment.py `
  --experiment-id dev-gptimage2-r1 `
  --seed 20260621 `
  --pipeline split_default `
  --max-revisions 1 `
  --reference-image D:\references\petct-style.png `
  --prepare-only
```

成功时应显示：

```text
300 cases, 600 paired runs
Preparation completed; no model APIs were called.
```

生成：

```text
runtime/experiments/dev-gptimage2-r1/
├─ experiment_lock.json
├─ work_items.csv
└─ events.jsonl
```

### 5.2 第二步：小规模 API 试跑

先运行 2 个工作项，可覆盖一个病例的一对策略：

```powershell
python scripts/experiments/run_paired_experiment.py `
  --experiment-id dev-gptimage2-r1 `
  --seed 20260621 `
  --pipeline split_default `
  --max-revisions 1 `
  --reference-image D:\references\petct-style.png `
  --max-items 2
```

恢复同一实验时，命令中的队列、seed、pipeline、参数、参考图和当前代码必须与锁文件一致。

试跑后建议立即检查：

- `runtime/cases/<日期>/<病例号>/<run_id>/attempt_*.png` 是否能正常打开；
- `manifest.json` 中的 `strategy`、`pipeline`、`gate_outcome`、`laterality_plan` 是否符合预期；
- `runtime/experiments/<experiment-id>/events.jsonl` 是否出现 `completed` 或明确的 `failed` 事件；
- 图片是否存在明显的侧别、中文、病灶位置、性质颜色或 SUVmax 错误。

如果试跑暴露出需要改提示词、模型、参考图、代码或数据的问题，不要在同一个 `experiment-id` 下继续；修正后换新实验 ID 重新 `--prepare-only`。

### 5.3 使用命令行覆盖模型和参数

```powershell
python scripts/experiments/run_paired_experiment.py `
  --experiment-id dev-gptimage2-r1 `
  --seed 20260621 `
  --pipeline split_default `
  --image-provider image_api `
  --image-model gpt-image-2 `
  --review-provider review_api `
  --review-model gpt-5.4 `
  --image-size 1536x1024 `
  --image-quality high `
  --image-output-format png `
  --image-background opaque `
  --max-revisions 1 `
  --max-items 2
```

### 5.4 继续运行与断点续跑

去掉 `--max-items` 即可运行全部剩余工作项：

```powershell
python scripts/experiments/run_paired_experiment.py `
  --experiment-id dev-gptimage2-r1 `
  --seed 20260621 `
  --pipeline split_default `
  --max-revisions 1 `
  --reference-image D:\references\petct-style.png
```

进程中断后重新执行完全相同的命令即可：

- 已完成工作项自动跳过；
- 失败工作项会再次尝试；
- 如果图片和 manifest 已保存、但进程在写审计事件前中断，恢复时会自动识别并补记；
- 不删除既往成功结果。

正式完整批次前，建议确认：

- API Key、Base URL、模型名和图像参数已经通过小样本试跑；
- `settings/local_providers.json`、`.env`、`runtime/` 不会被提交；
- 本次命令已经保存到实验记录或团队文档，便于完全相同地续跑；
- 团队已确认 SAP、人工评价方案和错误分类；
- 如果代码刚改过，已经运行 `node --check webapp/static/app.js` 和 `python -B -m pytest ...`。

## 6. 修改模型、提示词或增加比较

当前尚处开发阶段，可以继续修改。原则是：

- 同一个 `experiment-id`：代表同一套不可变设置，用于断点续跑；
- 修改模型、提示词、参考图、参数、代码或数据后：使用新的 `experiment-id`；
- 不同实验 ID 可以在后续作为模型、提示词或门控策略比较批次。

示例：

```text
dev-gptimage2-r1
dev-gptimage2-r2
dev-model-b-r1
dev-no-reference-r1
```

如果要加入第三种策略或其他正式比较，需要扩展 `petct/experiment.py` 中的策略定义和对应 SAP；不要把新策略伪装成现有 `GATED/UNGATED`。

## 7. 结果保存在哪里

### 7.1 每次图片生成结果

```text
runtime/cases/<日期>/<病例号>/<run_id>/
├─ attempt_1.png
├─ attempt_2.png              # 发生修订时存在
├─ reference.png              # 使用参考图时存在
└─ manifest.json
```

`manifest.json` 包含：

- 病例号、配对键、实验 ID、工作项 ID、癌种；
- 门控/无门控策略；
- provider、模型、图像参数和最大修订次数；
- 每次生成图片及审查意见；
- 最终门控结果、耗时和人工评价；
- 提示词版本与 SHA-256；
- 代码源树 SHA-256、Git 状态；
- 随机种子和参考图 SHA-256。

### 7.2 实验级记录

```text
runtime/experiments/<experiment-id>/
├─ experiment_lock.json
├─ work_items.csv
└─ events.jsonl
```

- `experiment_lock.json`：本批次不可变快照，包括 SAP 和人工评价方案文件哈希；
- `work_items.csv`：600 个计划工作项；
- `events.jsonl`：追加式运行审计日志。

### 7.3 人工评价

医生在门诊网页中打开当前生成结果后保存 `PASS/FAIL`、错误类型、评价者和备注。评价写回对应 `manifest.json`。

当前网页主要面向生成后的即时评价，尚没有独立的“600 例待评价队列”页面。批量生成完成后仍需逐例打开或后续开发专用评价工作台。

当前可用操作路径：

1. 启动网页：`python run_web.py`；
2. 打开 `http://127.0.0.1:8000`；
3. 点击“历史记录”，按病例号或最近记录找到对应运行；
4. 打开运行结果，查看最终图片和 AI 门控记录；
5. 填写医生 `PASS/FAIL`、错误类型、评价者和备注并保存；
6. 保存后重新导出统计输入。

导出表会保留缺失项并提示数据尚未可用于正式分析；真正的统计脚本会在发现 `MISSING` 或缺少人工 `PASS/FAIL` 时直接报错退出。

## 8. 导出统计输入

```powershell
python scripts/experiments/export_paired_results.py `
  --experiment-id dev-gptimage2-r1 `
  --output data/analysis/paired_runs.csv
```

输出一行对应一个计划工作项，共应为 600 行。它不会导出姓名或报告全文，但会保留病例号、配对键、运行路径、模型、提示词版本和人工评价。

如果仍有 `MISSING` 或未完成人工评价，导出命令会提示数据尚不能用于正式分析。

## 9. 执行统计分析

```powershell
python scripts/analysis/run_statistical_analysis.py `
  --input data/analysis/paired_runs.csv `
  --output-dir data/analysis/results
```

分析前强制检查：

- 600 个运行全部完成；
- 300 个病例均具有一对策略；
- 三癌种各 100 例；
- 全部具备人工 `PASS/FAIL`；
- 模型、参数、提示词、代码哈希和 seed 在批次内一致。

任一条件不满足时直接报错，不生成模拟结果。

输出：

```text
data/analysis/results/
├─ analysis_results.json
├─ pass_rates.csv
├─ mcnemar.csv
└─ error_types.csv
```

## 10. 常见问题

### 锁文件不一致

提示当前队列、设置、提示词、seed 或代码与锁文件不同。若是故意修改，换一个新的 `experiment-id`；若只是想续跑，应恢复原命令和原代码版本。

### API Key 缺失

检查 `.env`、网页保存的 `settings/local_providers.json`，或通过 `--local-provider-config` 指定正确文件。

### Base URL 错误

通常填写 provider 根地址，例如 `https://example.com/v1`，不要填到 `/images/generations` 或 `/chat/completions`。

### 单个病例失败

失败事件保留在 `events.jsonl`。修复临时网络或 API 状态后，重新执行相同命令；失败项会再次运行。

### 想更换模型继续同一个批次

不应在原实验 ID 中更换。创建新实验 ID，保留旧批次用于开发比较和追溯。

### 想先跑极小样本

使用 `--max-items 2` 或 `--max-items 6`。这不会改变锁定的 600 个计划工作项，只限制本次进程处理多少个待运行项。
