# PET/CT 患者友好可视化研究系统

将泌尿系统肿瘤 PET/CT 报告转换为患者容易理解的医学信息图，并支持 AI 图片审查、反馈修订、医生人工评价、患者理解度出题、可审计批量实验和真实数据统计分析。

当前项目仍处于开发和研究方案迭代阶段，尚未提交或完成正式 300 例研究。每个实验批次可独立锁定当时的数据、模型、提示词和参数，因此项目可以继续修改并增加新的比较。

## 该从哪里开始

按当前任务选择路径：

| 任务 | 建议阅读顺序 | 主要命令 |
|---|---|---|
| 门诊单例生成图片 | 快速安装 → API 与模型配置 → 门诊网页 | `python run_web.py` |
| 回顾性 300 例配对实验 | 快速安装 → API 与模型配置 → 回顾性配对批量实验 → 导出统计数据 → 统计分析 | `python scripts/experiments/run_paired_experiment.py ...` |
| 只做代码或文档修改 | 快速安装 → 运行测试 → 项目结构 | `node --check webapp/static/app.js` 和 `python -B -m pytest ...` |

正式批次前先使用开发实验 ID 完成 `--prepare-only` 和少量 `--max-items` 试跑。任何有意更换模型、提示词、参考图、参数、代码或数据的情况，都使用新的 `experiment-id`，不要在旧实验 ID 中继续混跑。

## 当前研究

当前内置回顾性研究为同病例配对设计：

- 肾癌、前列腺癌、尿路上皮癌各 100 例；
- 每例运行 `UNGATED` 和 `GATED`；
- `UNGATED`：只生成一次；
- `GATED`：视觉模型结合原报告审查图片，失败时反馈修图；
- 医生整图 `PASS/FAIL` 是金标准；
- 主要终点：无门控首次图片医生通过率；
- 配对比较：双侧精确 McNemar 检验。

旧版“2100→1500、三阶段评价”已不再作为当前主要方案。统一后的工作草案见：

- [课题计划书](docs/research_proposal.md)
- [统计分析计划（SAP）](docs/statistical_analysis_plan.md)
- [人工评价方案](docs/image_evaluation_plan.md)
- [项目计划与状态](docs/project_plan_and_status.md)
- [系统技术架构与论文写作说明](docs/technical_architecture_for_manuscript.md)

SAP 是 Statistical Analysis Plan，即统计分析计划。它在查看正式结果前预先规定终点、分析人群、统计方法和缺失数据规则。

## 快速安装

要求 Python `3.11–3.13`。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

运行测试：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP=(Resolve-Path .tmp).Path
$env:TMP=$env:TEMP
$base=".tmp\pytest-$((Get-Date).ToString('yyyyMMddHHmmss'))"
python -B -m pytest --basetemp $base -p no:cacheprovider -q
```

Windows 下固定复用 `.tmp\pytest` 偶尔会因旧进程或安全软件占用而清理失败；上面的命令每次使用新的临时目录，更适合日常验证。

## 门诊网页

启动：

```powershell
python run_web.py
```

Windows can also start the local 8001 service by double-clicking:

```text
NanoBananaPETCT.exe
```

或指定端口：

```powershell
.\scripts\start_web.ps1 -Port 8001
```

浏览器访问：

```text
http://127.0.0.1:8000
http://127.0.0.1:8001
```

网页支持：

1. 输入病例号和 PET/CT 报告；
2. 选择 pipeline、provider、模型和图像参数；
3. 上传参考样式图；
4. 开启或关闭 AI 质量门控；
5. 生成 2–3 道患者理解度题目；
6. 查看图片和每轮门控结果；
7. 保存医生 `PASS/FAIL`、错误类型、评价者和备注；
8. 打印患者友好报告；
9. 查看历史生成记录，并重新打开既往运行。

网页运行结果按以下结构保存：

```text
runtime/cases/<YYYY-MM-DD>/<病例号>/<run_id>/
├─ attempt_1.png
├─ attempt_2.png
├─ reference.png
└─ manifest.json
```

病例号中的 Windows 非法路径字符会替换为 `_`。最内层保留 `run_id`，同一天重复生成同一病例时不会覆盖。

## API 与模型配置

有三种配置方式：

### 网页配置

点击网页右上角“API 设置”，填写 API Key、Base URL、API 类型、超时和默认模型。配置保存在：

```text
settings/local_providers.json
```

该文件默认不提交 Git。

门诊网页默认会同时尝试生图、AI 审查和患者理解度出题，因此默认需要生图、审查和出题三个任务的配置。若只想快速生成图片，可以在网页中关闭“生成理解度题目”；批量图片实验本身只要求生图和审查配置可用。

### `.env`

三任务独立 API：

```dotenv
PETCT_IMAGE_API_KEY=
PETCT_IMAGE_BASE_URL=
PETCT_REVIEW_API_KEY=
PETCT_REVIEW_BASE_URL=
PETCT_QUESTION_API_KEY=
PETCT_QUESTION_BASE_URL=
```

OpenAI 单账号：

```dotenv
OPENAI_API_KEY=
```

统一兼容网关：

```dotenv
CUSTOM_OPENAI_API_KEY=
CUSTOM_OPENAI_BASE_URL=
```

### 自定义配置文件

批量命令可显式指定：

```powershell
--provider-config settings/providers.json `
--local-provider-config D:\private\petct-providers.json
```

不要把 API Key 直接写在命令行参数中。

项目内置 pipeline：

- `split_default`：生图、审查、出题使用独立 provider；
- `openai_single_key`：共用 OpenAI Key；
- `custom_gateway_all`：共用 OpenAI-compatible 网关。

## 回顾性配对批量实验

完整说明见 [回顾性配对批量实验使用手册](docs/retrospective_batch_manual.md)。

推荐操作顺序：

1. 用 `--prepare-only` 检查队列和配置，不调用模型 API；
2. 用 `--max-items 2` 先跑一个病例的一对策略；
3. 人工查看生成图片、`manifest.json` 和 `events.jsonl`；
4. 如果要调整模型、提示词、参考图或代码，换新的 `experiment-id` 后重新从第 1 步开始；
5. 确认可用后，去掉 `--max-items` 继续完整批次。

查看参数：

```powershell
python scripts/experiments/run_paired_experiment.py --help
```

### 只检查队列和配置

不会调用模型 API：

```powershell
python scripts/experiments/run_paired_experiment.py `
  --experiment-id dev-gptimage2-r1 `
  --seed 20260621 `
  --pipeline split_default `
  --max-revisions 1 `
  --prepare-only
```

默认验证三癌种各 100 例，并生成 300 例、600 个配对工作项。

### 小规模试跑

```powershell
python scripts/experiments/run_paired_experiment.py `
  --experiment-id dev-gptimage2-r1 `
  --seed 20260621 `
  --pipeline split_default `
  --image-model gpt-image-2 `
  --review-model gpt-5.4 `
  --image-size 1536x1024 `
  --image-quality high `
  --max-revisions 1 `
  --max-items 2
```

### 使用参考图

```powershell
--reference-image D:\references\petct-style.png
```

### 断点续跑

重新执行完全相同的命令：

- 已完成工作项自动跳过；
- 失败项可重新运行；
- 已保存 manifest 但尚未写入审计事件的结果会自动恢复；
- 不会覆盖已完成图片。

### 修改版本或增加比较

实验锁只约束同一个 `experiment-id`。模型、提示词、参考图、参数、代码或数据有意变化时，换一个新的实验 ID：

```text
dev-model-a-r1
dev-model-a-r2
dev-model-b-r1
dev-no-reference-r1
```

这样可以继续开发并保留不同版本用于后续比较。若增加第三种正式策略，需要同时扩展工作项定义、导出、SAP 和统计方法。

## 批量实验记录

```text
runtime/experiments/<experiment-id>/
├─ experiment_lock.json
├─ work_items.csv
└─ events.jsonl
```

- `experiment_lock.json`：队列、数据哈希、模型、参数、提示词、SAP/评价方案、参考图、随机种子和代码版本快照；
- `work_items.csv`：全部计划运行及固定顺序；
- `events.jsonl`：追加式开始、完成、失败和恢复记录。

每次生成的 `manifest.json` 还记录：

- 实验 ID、工作项 ID、癌种和配对键；
- provider ID、API 类型、Base URL、模型和超时；
- 生图参数及最大修订次数；
- 提示词版本和 SHA-256；
- 参考图和代码源树 SHA-256；
- Git commit/dirty 状态；
- 随机种子及 `providerSeedApplied`。

当前图像 API 不支持 seed，所以随机种子只控制病例与策略执行顺序，不能保证图像像素级复现。

## 导出统计数据

```powershell
python scripts/experiments/export_paired_results.py `
  --experiment-id dev-gptimage2-r1 `
  --output data/analysis/paired_runs.csv
```

导出表不包含姓名或报告全文。正式 300 例批次应有 600 行。

## 统计分析

```powershell
python scripts/analysis/run_statistical_analysis.py `
  --input data/analysis/paired_runs.csv `
  --output-dir data/analysis/results
```

统计脚本没有模拟数据回退。以下任一情况都会报错退出：

- 输入文件不存在；
- 不是 600 个已完成运行；
- 病例没有严格配对；
- 三癌种不是各 100 例；
- 缺少医生 `PASS/FAIL`；
- 冻结的模型、参数、提示词、代码哈希或 seed 不一致。

输出：

```text
data/analysis/results/
├─ analysis_results.json
├─ pass_rates.csv
├─ mcnemar.csv
└─ error_types.csv
```

## 旧批量生图入口

早期历史批量生图仍可使用：

```powershell
python cli.py "data/raw/08-24年肾癌患者核医学报告.csv" --limit 5
```

该入口使用旧的 `NANOBANANA_*` 配置，不具备新版配对实验的实验锁、审计日志和严格统计导出。正式配对研究应使用 `scripts/experiments/run_paired_experiment.py`。

## 项目结构

```text
petct/                         核心领域逻辑、provider、生成、评价和实验契约
webapp/                        FastAPI 门诊网页
settings/                      provider 与 pipeline 配置
scripts/experiments/           配对批量运行和统计导出
scripts/analysis/              严格真实数据统计分析
scripts/datasets/              历史抽样与数据集构建
scripts/questions/             理解度题目工具
data/raw/                      原始报告
data/derived/evaluation_dataset/ 300 例当前评测队列
data/analysis/                 统计输入和结果
runtime/cases/                 单次运行图片与 manifest
runtime/experiments/           实验锁、工作项和审计日志
docs/                          SAP、评价方案、使用手册和决策记录
archive/                       历史入口和旧测试
```

## 数据与安全

原始 CSV 和运行 manifest 可能含病例号、姓名或报告全文。真实门诊使用前仍需完善：

- 患者标识脱敏；
- 访问控制；
- 审计与备份；
- 数据保存期限；
- 伦理审批和知情同意；
- 多评价者原始评分与仲裁记录。

不要提交 `.env`、`settings/local_providers.json`、`runtime/` 或 `outputs/`。
