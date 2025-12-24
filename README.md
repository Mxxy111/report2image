# NanoBanana PET-CT Visualization Project

本项目用于批量调用 NanoBanana API（或兼容的生图 API），根据 CSV 中的医学报告内容生成 PET-CT 可视化图像。


## 1. 快速开始

### 安装依赖
```powershell
pip install -r requirements.txt
```

### 准备数据
准备一个 CSV 或 Excel 文件（例如 `data.csv`），需包含以下列：
- **ID 列**：如 `门诊号/住院号`、`ID`
- **文本列**：如 `检查所见`、`report`

**提示**：本项目已支持 **智能编码识别**，您可以直接使用 Windows 下生成的中文 CSV（GBK 编码）或 UTF-8 文件，无需手动转换格式。

### 配置 API
在项目根目录创建 `.env` 文件（推荐），或者直接在 `config.py` 中修改。

**`.env` 文件示例：**
```env
# === 关键配置：请根据您的 Chat 接口文档填写 ===
NANOBANANA_API_KEY=sk-your-api-key-here
# 注意：这里使用 Chat 接口地址
NANOBANANA_API_URL=https://api.sydney-ai.com/v1/chat/completions
NANOBANANA_MODEL=gemini-2.5-flash-image

# === (可选) 参考图 URL ===
# 如果您希望所有生成任务都参考同一张底图，请在此填入 URL
# REFERENCE_IMAGE_URL=https://filesystem.site/cdn/.../image.png

# === 性能配置 ===
RPM=60              # 每分钟最大请求数 (根据您的套餐限制)
CONCURRENCY=5       # 最大并发数 (同时进行的请求数)
```

## 2. 工作模式与配置说明

本项目支持 **Chat 模式** 和 **Image 模式** 两种接口协议，并各自支持“文生图”和“垫图生图”。请根据您的 API 文档选择正确的模式。

### 模式选择指南
- 如果您的 API 地址以 `/v1/chat/completions` 结尾，请选择 **Chat 模式**（默认）。
- 如果您的 API 地址以 `/v1/images/generations` 结尾，请选择 **Image 模式**。

---

### 模式 1：Chat 接口模式 (默认)
**接口地址**：`.../v1/chat/completions`
**特点**：使用 `messages` 数组通信。

- **A. 纯文本生图**：
  - **配置**：不设置 `REFERENCE_IMAGE_URL`。
  - **逻辑**：发送 `messages: [{"content": "prompt"}]`。

- **B. 垫图生图 (多模态)**：
  - **配置**：设置 `REFERENCE_IMAGE_URL`。
  - **逻辑**：发送包含 `image_url` 对象的多模态消息。

---

### 模式 2：Image 接口模式 (OpenAI 变体)
**接口地址**：`.../v1/images/generations`
**特点**：使用 `prompt`, `size` 等扁平字段。

- **A. 纯文本生图**：
  - **配置**：`api_mode="image"`，不设置 `REFERENCE_IMAGE_URL`。
  - **逻辑**：发送 `{"prompt": "prompt...", "size": "1024x1024"}`。

- **B. 垫图生图 (URL 拼接)**：
  - **配置**：`api_mode="image"`，设置 `REFERENCE_IMAGE_URL`。
  - **逻辑**：发送 `{"prompt": "prompt...\nhttps://image.url", "size": "..."}`。脚本会自动将图片 URL 拼接到 Prompt 后。

## 3. 详细参数列表

所有参数均支持通过 **命令行** 或 **环境变量 (.env)** 设置。

| 参数名 | 命令行参数 | 环境变量 | 说明 | 默认值 |
|--------|------------|----------|------|--------|
| **API 密钥** | `--api-key` | `NANOBANANA_API_KEY` | 必需。API 认证密钥。 | - |
| **API 模式** | `--api-mode` | `NANOBANANA_API_MODE` | `chat` 或 `image` | chat |
| **API 地址** | `--api-url` | `NANOBANANA_API_URL` | 完整 URL | - |
| **模型名称** | `--model` | `NANOBANANA_MODEL` | 如 `gemini-2.5-flash-image` | - |
| **参考图 URL** | `--reference-image` | `REFERENCE_IMAGE_URL` | 可选。支持 **HTTP链接** 或 **本地文件路径** (自动转 Base64)。 | None |
| **图片尺寸** | `--size` | `NANOBANANA_SIZE` | 仅 Image 模式有效，如 `1024x1024` | 1024x1024 |
| **RPM 限制** | `--rpm` | `RPM` | 每分钟最大请求数。 | 60 |
| **并发数** | `--concurrency` | `CONCURRENCY` | 同时发起的请求数。 | 5 |
| **超时时间** | `--timeout` | `TIMEOUT` | 单次 API 请求的超时时间（秒）。 | 300 |
| **测试限制** | `--limit` | - | **新增**：仅处理前 N 条数据（方便测试）。 | - |
| **历史文件** | `--history-file` | - | **正式模式核心**：指定一个 CSV 路径记录已处理任务。启用后会自动跳过已做过的任务，并统一图片存储路径。 | None |
| **同步校验** | `--sync` | - | **新增**：运行前检查历史文件与图片文件夹的一致性，自动清理无效记录。 | False |
| **历史合并** | `--merge-history` | - | **新增**：检测到“前片/对比”关键词时，自动合并同一患者的历史报告。 | False |
| **随机抽样** | `--random-sample` | - | **新增**：唯一性随机抽样。每个患者 ID 只保留 1 条记录，然后随机打乱。常与 `--limit` 配合使用。 | False |
| **报告文本列** | `--text-cols` | - | **新增**：指定 CSV 中作为报告内容的列名，多列用逗号分隔。 | 自动检测 |
| **ID 列** | `--id-col` | - | **新增**：指定 CSV 中的唯一 ID 列名。 | 自动检测 |

## 4. 输出结果说明
运行完成后，默认会在 `outputs/` 目录下生成一个带时间戳的文件夹。

**如果指定了 `--history-file`**（正式模式）：
```
outputs/
└── {history_filename}_images/  # 与历史记录文件同名的图片文件夹
    ├── 1001_abcdef.png
    └── ...
```

**默认模式**（测试/单次运行）：
```
outputs/
└── batch_20251203_160000/
    ├── images/                 # 存放所有生成的图片
    │   ├── 1001_abcdef.png     # 命名格式：{ID}_{Hash前6位}.png
    │   └── 1002_123456.png
    ├── call_logs/              # (新增) API调用详细日志
    │   └── api_calls_xxx.jsonl # 包含完整Prompt、Token消耗等
    └── results_summary.csv     # 结果汇总表
```

**`results_summary.csv` 包含以下关键列：**
*   `id`: 数据的唯一标识。
*   `original_text`: **原文内容**（用于与生成的图片进行对照）。
*   `local_image_path`: 图片在本地的保存路径。
*   `image_url`: 原始下载链接。
*   `elapsed_time`: 生成耗时。

## 5. 运行命令示例

### 场景 1：基础文生图 (快速测试前 5 条)
```powershell
python cli.py "data.csv" --limit 5 --api-url "..."
```

### 场景 2：使用本地图片垫图 (推荐)
为了获得风格高度一致的“医学简笔画”，**强烈建议**始终提供一张标准的参考图。脚本会自动将其转换为 Base64。
```powershell
python cli.py "data.csv" \
  --api-url "..." \
  --reference-image "C:\images\standard_template.png"
```

### 场景 3：进阶用法 - 历史合并与随机抽样
适用于复查患者较多的数据集。脚本会自动识别“同前”、“较前”等描述，将历史报告合并给 AI；并随机抽取 50 个不同患者进行测试。
```powershell
python cli.py "data.csv" \
  --limit 50 \
  --merge-history \
  --random-sample \
  --text-cols "检查结论" \
  --api-url "..." \
  --reference-image "C:\images\ref.png"
```

### 场景 4：自定义报告内容列 (拼接多列)
假设您的 CSV 中有“检查所见”和“诊断结论”两列，您希望将它们合并作为提示词的输入：
```powershell
python cli.py "data.csv" \
  --text-cols "检查所见,诊断结论" \
  --api-url "..."
```

### 场景 3：使用 Image 接口 (垫图模式)
```powershell
python cli.py "data.csv" \
  --api-mode "image" \
  --api-url "https://api.sydney-ai.com/v1/images/generations" \
  --size "1024x1024" \
  --reference-image "https://filesystem.site/cdn/demo.png"
```

## 6. 增量处理与正式运行 (Formal Mode)

当您需要处理大规模数据（如数千条记录），并且需要分批次运行、避免重复处理时，请使用 **历史文件 (`--history-file`)**。

### 工作原理
1. 脚本读取指定的 History CSV 文件（如 `processed_history.csv`）。
2. 在处理新数据时，会自动跳过 History 中已存在的记录（根据 ID + 内容哈希判断）。
3. 生成的图片将统一保存在 `{history_filename}_images` 文件夹中，而不是分散在每次运行的临时文件夹里。
4. 任务成功后，会自动将记录追加到 History CSV 中。

### 命令示例
```powershell
python cli.py "new_patient_data.csv" \
  --history-file "processed_history.csv" \
  --api-url "..." \
  --concurrency 10
```
**注意**：首次运行时，脚本会自动创建空的 History CSV。

## 7. 实用工具 (Tools)

### 批量删除图片 (`delete_images.py`)
**场景**：当您有一批特定的 ID（例如在 Excel 中筛选出的“不合格样本”），需要从图片文件夹中批量删除对应的图片时使用。

```powershell
python tools/delete_images.py "bad_cases.csv" "outputs/images"
```

**常用参数**：
- `--dry-run`：**试运行**。仅打印要删除的文件列表，不执行删除。强烈建议首次运行使用。
- `--no-header`：如果您的 CSV 文件没有表头（第一行就是数据），请加上此参数。
- `--id-col`：指定 ID 列名（默认自动识别包含 "id"/"号" 的列）。

### 8. 维护与修复 (Sync/Update)

此功能用于保持 **CSV 记录** 与 **实际图片文件夹** 的高度一致。
通常在以下情况使用：
1.  **清理无效记录**：您手动删除了某些不满意的图片，希望 CSV 中对应的“已完成”记录也自动删除，以便重新生成。
2.  **重建/恢复记录**：您的 CSV 记录意外丢失或损坏，但图片还在，希望根据文件名重新找回 CSV 记录。

#### 场景 A：自动清理无效记录
这是最常用的功能。脚本会检查 History CSV，如果发现某条记录对应的图片不存在，就会从 CSV 中删除该行。

**方法 1：在 CLI 中直接启用**
```powershell
python cli.py "data.csv" --history-file "processed.csv" --sync
```

**方法 2：使用独立工具**
```powershell
python tools/sync_history.py \
  --history "processed.csv" \
  --images "outputs/processed_images"
```

#### 场景 B：根据图片恢复 CSV (反向更新)
如果您有图片文件，但 CSV 中缺失了对应的记录（例如手动复制了图片进来），可以使用 `--source` 参数从原始数据表中找回这些信息并补全到 History CSV 中。

```powershell
python tools/sync_history.py \
  --history "processed.csv" \
  --images "outputs/processed_images" \
  --source "original_data.csv"
```
*   `--history`: 待修复的进度文件。
*   `--images`: 现有的图片文件夹。
*   `--source`: 原始完整数据源（用于查找 ID 对应的文本信息）。

## 9. 常见问题

**Q: 报错 `API Error 404: Not Found`**
A: 通常是 `NANOBANANA_API_URL` 填写错误。请检查是否多写或少写了 `/v1/...` 路径后缀。

**Q: 报错 `API Error 400: Invalid Parameter`**
A: 可能是模型名称 (`model`) 不正确，或者 API 不支持 `size` 参数。请检查 `client.py` 中的 `payload` 字典。

**Q: 图片无法下载**
A: 请检查 API 返回的 URL 是否可以公开访问。有些内网 API 返回的 URL 无法在外部下载。
