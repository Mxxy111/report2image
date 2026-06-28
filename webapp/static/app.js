const byId = (id) => document.getElementById(id);

const ERROR_TYPES = [
  ["LATERALITY", "左右相反"],
  ["CHINESE_TEXT", "中文错误"],
  ["LESION_LOCATION", "病灶位置错误"],
  ["OMISSION", "病灶或关键信息遗漏"],
  ["ANATOMICAL_DISTORTION", "解剖结构错误"],
  ["BENIGN_MALIGNANT", "良恶性倾向错误"],
  ["SUVMAX", "SUVmax 错误"],
  ["STYLE", "风格不一致"],
  ["HALLUCINATION", "虚构信息"],
  ["OTHER", "其他"],
];

const MODEL_FIELDS = {
  image: {
    providerSelect: "imageProviderSelect",
    modelInput: "imageModelInput",
    datalist: "imageModelOptions",
    taskModelKey: "image",
  },
  review: {
    providerSelect: "reviewProviderSelect",
    modelInput: "reviewModelInput",
    datalist: "reviewModelOptions",
    taskModelKey: "review",
  },
  question: {
    providerSelect: "questionProviderSelect",
    modelInput: "questionModelInput",
    datalist: "questionModelOptions",
    taskModelKey: "questions",
  },
};

const PROGRESS_STAGE_LABELS = {
  configuration: "配置检查",
  report_parsing: "信息筛选",
  laterality: "左右规划",
  image: "生成图片",
  review: "AI 图片审查",
  questions: "生成理解度题目",
  saving: "保存结果",
};

let providerRegistry = null;
let currentRunId = null;
let activeProviderId = null;
let providerFormModels = [];
let referenceImageDataUrl = null;
let progressTimerId = null;
let progressStartedAt = null;

function providerById(providerId) {
  return providerRegistry?.providers?.find((provider) => provider.id === providerId) || null;
}

function createOption(value, label, selectedValue) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  option.selected = value === selectedValue;
  return option;
}

function providerOptionLabel(provider) {
  const status = provider.isConfigured ? "" : "（未配置密钥）";
  const source = provider.source === "local" ? " · 本地" : "";
  return `${provider.label}${status}${source}`;
}

function fillProviderSelect(selectId, selectedId = "") {
  const select = byId(selectId);
  select.innerHTML = "";
  (providerRegistry?.providers || []).forEach((provider) => {
    select.appendChild(createOption(provider.id, providerOptionLabel(provider), selectedId));
  });
}

function providerTaskModel(providerId, taskModelKey) {
  return providerById(providerId)?.taskModels?.[taskModelKey] || "";
}

function selectedPipeline() {
  const id = byId("pipelineSelect").value;
  return providerRegistry?.pipelines?.find((pipeline) => pipeline.id === id);
}

function renderModelStatus(pipeline) {
  const container = byId("modelStatus");
  container.innerHTML = "";
  const effective = {
    image: byId("imageModelInput")?.value.trim() || pipeline.image.model,
    review: byId("reviewModelInput")?.value.trim() || pipeline.review.model,
    questions: byId("questionModelInput")?.value.trim() || pipeline.questions.model,
  };
  [
    ["生图", effective.image],
    ["审查", effective.review],
    ["左右规划", effective.questions],
    ["出题", effective.questions],
  ].forEach(([label, model]) => {
    const chip = document.createElement("span");
    chip.className = "status-token";
    chip.textContent = `${label} ${model || "未设置"}`;
    container.appendChild(chip);
  });
}

function fillModelDatalist(datalistId, providerId) {
  const datalist = byId(datalistId);
  datalist.innerHTML = "";
  const provider = providerById(providerId);
  (provider?.cachedModels || []).forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    datalist.appendChild(option);
  });
}

function refreshTaskModelOptions() {
  Object.values(MODEL_FIELDS).forEach((field) => {
    fillModelDatalist(field.datalist, byId(field.providerSelect).value);
  });
}

function updateOverrideDefaults() {
  const pipeline = selectedPipeline();
  if (!pipeline) return;
  fillProviderSelect("imageProviderSelect", pipeline.image.providerId);
  fillProviderSelect("reviewProviderSelect", pipeline.review.providerId);
  fillProviderSelect("questionProviderSelect", pipeline.questions.providerId);
  byId("imageModelInput").value = providerTaskModel(pipeline.image.providerId, "image");
  byId("imageModelInput").placeholder = pipeline.image.model;
  byId("imageSizeInput").placeholder = pipeline.image.options?.size || "1536x1024";
  byId("imageQualityInput").placeholder = pipeline.image.options?.quality || "medium";
  byId("imageOutputFormatInput").value = "";
  byId("imageBackgroundInput").value = "";
  byId("imageModerationInput").value = "";
  byId("imageCompressionInput").value = "";
  byId("imageInputFidelityInput").value = "";
  byId("reviewModelInput").value = providerTaskModel(pipeline.review.providerId, "review");
  byId("reviewModelInput").placeholder = pipeline.review.model;
  byId("questionModelInput").value = providerTaskModel(pipeline.questions.providerId, "questions");
  byId("questionModelInput").placeholder = pipeline.questions.model;
  renderModelStatus(pipeline);
  refreshTaskModelOptions();
}

function renderPipelineSelect(preserveSelection = false) {
  const select = byId("pipelineSelect");
  const previous = preserveSelection ? select.value : "";
  select.innerHTML = "";
  (providerRegistry?.pipelines || []).forEach((pipeline) => {
    const selected = previous || providerRegistry.defaultPipelineId;
    select.appendChild(createOption(pipeline.id, pipeline.label, selected));
  });
}

async function loadConfig(options = {}) {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (!response.ok) throw new Error(config.detail || "模型配置不可用");
    providerRegistry = config.providerRegistry;
    renderPipelineSelect(options.preservePipeline);
    updateOverrideDefaults();
    if (!byId("providerDialog").hidden) renderProviderManager();
  } catch (exception) {
    byId("modelStatus").textContent = exception.message || "模型配置不可用";
  }
}

function setLoading(loading) {
  const button = byId("generateButton");
  button.disabled = loading;
  button.querySelector("span").textContent = loading ? "正在生成，请稍等..." : "生成图片与题目";
}

function progressStep(stage) {
  if (!Object.hasOwn(PROGRESS_STAGE_LABELS, stage)) return null;
  return byId("generationProgress").querySelector(`[data-stage="${stage}"]`);
}

function updateElapsedTime() {
  if (!progressStartedAt) return;
  const elapsedSeconds = Math.floor((Date.now() - progressStartedAt) / 1000);
  const minutes = Math.floor(elapsedSeconds / 60);
  const seconds = String(elapsedSeconds % 60).padStart(2, "0");
  byId("progressElapsed").textContent = `${minutes}:${seconds}`;
}

function formatDuration(duration_seconds) {
  const totalSeconds = Number(duration_seconds);
  if (!Number.isFinite(totalSeconds)) return "生成耗时未记录";
  if (totalSeconds < 1) return "生成耗时 <1 秒";
  const rounded = Math.round(totalSeconds);
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;
  return minutes > 0
    ? `生成耗时 ${minutes} 分 ${seconds} 秒`
    : `生成耗时 ${seconds} 秒`;
}

function formatGeneratedAt(generatedAt) {
  if (!generatedAt) return "";
  const date = new Date(generatedAt);
  if (Number.isNaN(date.getTime())) return "";
  return `生成时间 ${date.toLocaleString("zh-CN", { hour12: false })}`;
}

function resetGenerationProgress({ gateEnabled, questionsEnabled }) {
  if (progressTimerId) window.clearInterval(progressTimerId);
  const container = byId("generationProgress");
  container.hidden = false;
  container.setAttribute("aria-busy", "true");
  byId("progressTitle").textContent = "正在准备生成";
  byId("progressElapsed").textContent = "0:00";
  byId("generationErrorDetail").hidden = true;
  byId("generationErrorDetail").textContent = "";
  container.querySelectorAll(".progress-step").forEach((step) => {
    const detail = step.querySelector("small");
    if (!step.dataset.defaultMessage) step.dataset.defaultMessage = detail.textContent;
    detail.textContent = step.dataset.defaultMessage;
    step.dataset.status = "pending";
  });
  if (!gateEnabled) progressStep("review").dataset.status = "skipped";
  if (!questionsEnabled) progressStep("questions").dataset.status = "skipped";
  progressStartedAt = Date.now();
  progressTimerId = window.setInterval(updateElapsedTime, 1000);
}

function applyProgressEvent(event) {
  const step = progressStep(event.stage);
  if (!step) return;
  const allowedStatuses = new Set(["pending", "running", "completed", "skipped", "error"]);
  const status = allowedStatuses.has(event.status) ? event.status : "running";
  step.dataset.status = status;
  const details = [event.message];
  if (event.providerLabel || event.model) {
    details.push([event.providerLabel, event.model].filter(Boolean).join(" / "));
  }
  step.querySelector("small").textContent = details.filter(Boolean).join(" · ");
  if (status === "running") {
    byId("progressTitle").textContent = `${PROGRESS_STAGE_LABELS[event.stage]}中`;
  }
}

function finishGenerationProgress(success) {
  if (progressTimerId) window.clearInterval(progressTimerId);
  progressTimerId = null;
  updateElapsedTime();
  byId("generationProgress").setAttribute("aria-busy", "false");
  byId("progressTitle").textContent = success ? "全部处理完成" : "处理未完成";
}

function generationError(detail, fallbackMessage = "生成失败") {
  const exception = new Error(detail?.message || fallbackMessage);
  exception.detail = detail;
  return exception;
}

function renderGenerationError(detail) {
  const stage = Object.hasOwn(PROGRESS_STAGE_LABELS, detail?.stage) ? detail.stage : null;
  const stageLabel = stage ? PROGRESS_STAGE_LABELS[stage] : "处理过程";
  const step = stage ? progressStep(stage) : null;
  if (step) {
    step.dataset.status = "error";
    step.querySelector("small").textContent = detail?.message || "未能完成请求";
  }

  const error = byId("errorMessage");
  error.textContent = `${stageLabel}失败：${detail?.message || "未能完成请求"}`;
  error.hidden = false;

  const lines = [];
  if (detail?.providerLabel || detail?.model) {
    lines.push(`调用：${[detail.providerLabel, detail.model].filter(Boolean).join(" / ")}`);
  }
  if (detail?.upstreamStatus) lines.push(`上游状态：HTTP ${detail.upstreamStatus}`);
  if (detail?.suggestion) lines.push(`建议：${detail.suggestion}`);
  const box = byId("generationErrorDetail");
  box.textContent = lines.join("\n");
  box.hidden = lines.length === 0;
}

async function readGenerationStream(response) {
  if (!response.ok) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      throw generationError(null, `服务返回 HTTP ${response.status}`);
    }
    const detail = payload?.detail;
    throw generationError(
      detail && !Array.isArray(detail) ? detail : null,
      `请求校验失败（HTTP ${response.status}）`,
    );
  }
  if (!response.body) throw generationError(null, "浏览器不支持流式进度读取");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  const handleLine = (line) => {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === "progress") applyProgressEvent(event);
    if (event.type === "error") throw generationError(event.error);
    if (event.type === "result") result = event.data;
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    lines.forEach(handleLine);
    if (done) break;
  }
  if (buffer.trim()) handleLine(buffer);
  if (!result) throw generationError(null, "服务未返回最终生成结果");
  return result;
}

function setReferenceStatus(text) {
  byId("referenceImageStatus").textContent = text;
}

function clearReferenceImage() {
  referenceImageDataUrl = null;
  byId("referenceImageInput").value = "";
  setReferenceStatus("未选择参考图");
}

function selectedCheckboxValues(groupId) {
  return [...byId(groupId).querySelectorAll("input:checked")].map((item) => item.value);
}

function currentReviewStrength() {
  return byId("reviewStrengthSelect").value;
}

function currentInputReportText() {
  const conclusionText = byId("conclusionText").value.trim();
  const findingsText = byId("findingsText").value.trim();
  return [
    conclusionText ? `检查结论：\n${conclusionText}` : "",
    findingsText ? `检查所见：\n${findingsText}` : "",
  ].filter(Boolean).join("\n\n");
}

function handleReferenceImageChange(event) {
  const file = event.target.files?.[0];
  const error = byId("errorMessage");
  error.hidden = true;
  if (!file) {
    clearReferenceImage();
    return;
  }
  const acceptedTypes = new Set(["image/png", "image/jpeg", "image/webp"]);
  if (!acceptedTypes.has(file.type)) {
    clearReferenceImage();
    error.textContent = "参考图仅支持 PNG、JPEG 或 WebP。";
    error.hidden = false;
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    clearReferenceImage();
    error.textContent = "参考图不能超过 15 MB。";
    error.hidden = false;
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    referenceImageDataUrl = String(reader.result || "");
    const sizeMb = (file.size / 1024 / 1024).toFixed(2);
    setReferenceStatus(`${file.name} · ${sizeMb} MB`);
  };
  reader.onerror = () => {
    clearReferenceImage();
    error.textContent = "参考图读取失败，请重新选择。";
    error.hidden = false;
  };
  reader.readAsDataURL(file);
}

function buildOverrides() {
  const imageOptions = {};
  if (byId("imageSizeInput").value.trim()) imageOptions.size = byId("imageSizeInput").value.trim();
  if (byId("imageQualityInput").value.trim()) imageOptions.quality = byId("imageQualityInput").value.trim();
  if (byId("imageOutputFormatInput").value) imageOptions.output_format = byId("imageOutputFormatInput").value;
  if (byId("imageBackgroundInput").value) imageOptions.background = byId("imageBackgroundInput").value;
  if (byId("imageModerationInput").value) imageOptions.moderation = byId("imageModerationInput").value;
  if (byId("imageInputFidelityInput").value) imageOptions.input_fidelity = byId("imageInputFidelityInput").value;
  const compression = byId("imageCompressionInput").value.trim();
  if (compression) imageOptions.output_compression = Number(compression);

  return {
    imageOverride: {
      providerId: byId("imageProviderSelect").value,
      model: byId("imageModelInput").value.trim() || null,
      options: imageOptions,
    },
    reviewOverride: {
      providerId: byId("reviewProviderSelect").value,
      model: byId("reviewModelInput").value.trim() || null,
      options: {},
    },
    questionOverride: {
      providerId: byId("questionProviderSelect").value,
      model: byId("questionModelInput").value.trim() || null,
      options: {},
    },
  };
}

function renderQuestions(questions) {
  const section = byId("questionSection");
  const list = byId("questionList");
  list.innerHTML = "";
  section.hidden = questions.length === 0;
  questions.forEach((item, index) => {
    const block = document.createElement("div");
    block.className = "question";
    const title = document.createElement("p");
    title.textContent = `${index + 1}. ${item.question || ""}`;
    const answers = document.createElement("div");
    answers.className = "answers";
    (item.options || []).forEach((option, optionIndex) => {
      const answer = document.createElement("div");
      answer.className = "answer";
      const cleanOption = String(option || "")
        .replace(/^(?:\s*[（(]?[A-Da-d][)）.、:：]\s*)+/, "")
        .trim();
      answer.textContent = `${String.fromCharCode(65 + optionIndex)}. ${cleanOption}`;
      answers.appendChild(answer);
    });
    block.append(title, answers);
    list.appendChild(block);
  });
}

function renderGateOutcome(data) {
  const outcome = data.gate_outcome || { status: "NOT_REVIEWED", finalErrorTypes: [], allErrorTypes: [] };
  const badge = byId("gateBadge");
  if (outcome.status === "PASS") {
    badge.textContent = "AI 门控通过";
    badge.className = "status-badge pass";
  } else if (outcome.status === "FAIL") {
    badge.textContent = "AI 门控未通过";
    badge.className = "status-badge fail";
  } else {
    badge.textContent = "未启用 AI 门控";
    badge.className = "status-badge neutral";
  }
  let attemptSummary = "";
  if (outcome.revisionCount > 0) {
    attemptSummary = outcome.status === "FAIL"
      ? ` 已审查 ${outcome.attemptCount} 版并重修 ${outcome.revisionCount} 次；当前展示第 ${outcome.returnedAttempt} 版（重修后的最终图片），仍未通过。`
      : ` 已审查 ${outcome.attemptCount} 版并重修 ${outcome.revisionCount} 次；当前展示第 ${outcome.returnedAttempt} 版（重修后的最终图片）。`;
  } else if (outcome.attemptCount > 0) {
    attemptSummary = " 当前展示第 1 版图片。";
  }
  byId("gateSummary").textContent =
    `AI 门控：${outcome.status === "PASS" ? "通过" : outcome.status === "FAIL" ? "不通过" : "未审查"}。`
    + attemptSummary
    + (outcome.reason ? ` 理由：${outcome.reason}` : "");
  const issueList = byId("gateIssueList");
  issueList.innerHTML = "";
}

function renderErrorTypeCheckboxes() {
  const container = byId("humanErrorTypes");
  container.innerHTML = "";
  ERROR_TYPES.forEach(([value, label]) => {
    const item = document.createElement("label");
    item.className = "checkbox-pill";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    const text = document.createElement("span");
    text.textContent = label;
    item.append(input, text);
    container.appendChild(item);
  });
}

function selectedHumanErrorTypes() {
  return [...byId("humanErrorTypes").querySelectorAll("input:checked")].map((item) => item.value);
}

function applyHumanEvaluation(evaluation) {
  const errorTypes = new Set(evaluation?.errorTypes || []);
  byId("evaluationMessage").textContent = "";
  byId("humanDecision").value = evaluation?.overallDecision || "";
  byId("reviewerInput").value = evaluation?.reviewer || "";
  byId("humanNotes").value = evaluation?.notes || "";
  byId("humanErrorTypes").querySelectorAll("input").forEach((item) => {
    item.checked = errorTypes.has(item.value);
  });
}

function renderRunResult(data) {
  currentRunId = data.run_id;
  const caseId = data.case_id || byId("caseId").value.trim();
  const reportText = data.report_text || currentInputReportText();
  byId("emptyState").hidden = true;
  byId("printSheet").hidden = false;
  byId("printButton").disabled = false;
  byId("resultTitle").textContent = `病例 ${caseId || currentRunId || ""}`;
  byId("printCaseId").textContent = caseId;
  byId("printGeneratedAt").textContent = formatGeneratedAt(data.generated_at);
  byId("printDuration").textContent = formatDuration(data.duration_seconds);
  byId("originalReport").textContent = reportText;
  byId("resultImage").src = data.image_data_url || data.image_url || "";
  renderGateOutcome(data);
  renderQuestions(data.questions || []);
  applyHumanEvaluation(data.human_evaluation);
}

function historyStatusText(item) {
  const status = item?.gate_outcome?.status || "NOT_REVIEWED";
  if (status === "PASS") return "通过";
  if (status === "FAIL") return "不通过";
  return "未审查";
}

function historyStatusClass(item) {
  const status = item?.gate_outcome?.status || "NOT_REVIEWED";
  if (status === "PASS") return "history-status pass";
  if (status === "FAIL") return "history-status fail";
  return "history-status";
}

function renderHistoryList(items) {
  const list = byId("historyList");
  list.innerHTML = "";
  if (items.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted-text";
    empty.textContent = "暂无历史记录。";
    list.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    const text = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `病例 ${item.case_id || "未命名"}`;
    const meta = document.createElement("span");
    meta.textContent = [
      formatGeneratedAt(item.generated_at).replace(/^生成时间\s*/, ""),
      formatDuration(item.duration_seconds).replace(/^生成耗时\s*/, "耗时 "),
      item.pipeline_label || "",
    ].filter(Boolean).join(" · ");
    const detail = document.createElement("small");
    detail.textContent = `运行 ${item.run_id} · ${item.attempt_count || 0} 次生成`;
    const status = document.createElement("span");
    status.className = historyStatusClass(item);
    status.textContent = historyStatusText(item);
    text.append(title, meta, detail);
    button.append(text, status);
    button.addEventListener("click", () => loadHistoryRun(item.run_id));
    list.appendChild(button);
  });
}

async function loadHistory() {
  const message = byId("historyMessage");
  message.textContent = "正在读取历史记录...";
  const params = new URLSearchParams({ limit: "80" });
  const caseFilter = byId("historyCaseFilter").value.trim();
  if (caseFilter) params.set("case_id", caseFilter);
  try {
    const response = await fetch(`/api/history?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "历史记录读取失败。");
    renderHistoryList(data.items || []);
    message.textContent = `已读取 ${data.count || 0} 条历史记录。`;
  } catch (exception) {
    renderHistoryList([]);
    message.textContent = exception.message || "历史记录读取失败。";
  }
}

async function loadHistoryRun(runId) {
  const message = byId("historyMessage");
  message.textContent = "正在打开历史记录...";
  try {
    const response = await fetch(`/api/runs/${runId}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "历史记录打开失败。");
    renderRunResult(data);
    closeHistoryDialog();
  } catch (exception) {
    message.textContent = exception.message || "历史记录打开失败。";
  }
}

function openHistoryDialog() {
  byId("historyDialog").hidden = false;
  byId("historyCaseFilter").value = byId("caseId").value.trim();
  loadHistory();
}

function closeHistoryDialog() {
  byId("historyDialog").hidden = true;
}

async function generate() {
  const caseId = byId("caseId").value.trim();
  const conclusionText = byId("conclusionText").value.trim();
  const findingsText = byId("findingsText").value.trim();
  const reviewStrength = currentReviewStrength();
  const error = byId("errorMessage");
  error.hidden = true;
  if (!caseId || conclusionText.length < 6) {
    error.textContent = "请填写病例编号和检查结论。";
    error.hidden = false;
    return;
  }
  if (selectedCheckboxValues("displayFindingTypeGroup").length === 0) {
    error.textContent = "请至少选择一种图中展示内容。";
    error.hidden = false;
    return;
  }

  setLoading(true);
  resetGenerationProgress({
    gateEnabled: reviewStrength !== "OFF",
    questionsEnabled: byId("questionsEnabled").checked,
  });
  try {
    const { imageOverride, reviewOverride, questionOverride } = buildOverrides();
    const response = await fetch("/api/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_id: caseId,
        conclusion_text: conclusionText,
        findings_text: findingsText || null,
        display_finding_types: selectedCheckboxValues("displayFindingTypeGroup"),
        display_detail_fields: selectedCheckboxValues("displayDetailFieldGroup"),
        gate_enabled: byId("gateEnabled").checked,
        review_strength: reviewStrength,
        generate_questions: byId("questionsEnabled").checked,
        pipeline_id: byId("pipelineSelect").value,
        image_override: imageOverride,
        review_override: reviewOverride,
        question_override: questionOverride,
        reference_image_data_url: referenceImageDataUrl,
      }),
    });
    const data = await readGenerationStream(response);

    renderRunResult(data);
    finishGenerationProgress(true);
  } catch (exception) {
    if (exception.detail) {
      renderGenerationError(exception.detail);
    } else {
      renderGenerationError({ message: exception.message || "生成失败" });
    }
    finishGenerationProgress(false);
  } finally {
    setLoading(false);
  }
}

async function saveEvaluation() {
  const decision = byId("humanDecision").value;
  const message = byId("evaluationMessage");
  message.textContent = "";
  if (!currentRunId) {
    message.textContent = "请先生成图片。";
    return;
  }
  if (!decision) {
    message.textContent = "请选择整图通过或不通过。";
    return;
  }

  const payload = {
    overallDecision: decision,
    errorTypes: selectedHumanErrorTypes(),
    reviewer: byId("reviewerInput").value.trim(),
    notes: byId("humanNotes").value.trim(),
  };
  const response = await fetch(`/api/runs/${currentRunId}/evaluation`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (response.ok) {
    message.textContent = "评价已保存。";
  } else {
    const data = await response.json();
    message.textContent = data.detail || "评价保存失败。";
  }
}

function openProviderDialog() {
  byId("providerDialog").hidden = false;
  renderProviderManager();
}

function closeProviderDialog() {
  byId("providerDialog").hidden = true;
}

function providerStatusText(provider) {
  const configured = provider.isConfigured ? "已配置" : "未配置";
  const source = provider.source === "local" ? "本地" : "内置";
  return `${configured} · ${source}`;
}

function renderProviderManager() {
  const providers = providerRegistry?.providers || [];
  if (!activeProviderId || !providerById(activeProviderId)) {
    activeProviderId = providers[0]?.id || null;
  }
  renderProviderList(providers);
  fillProviderForm(providerById(activeProviderId));
}

function renderProviderList(providers) {
  const list = byId("providerList");
  list.innerHTML = "";
  providers.forEach((provider) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `provider-item${provider.id === activeProviderId ? " active" : ""}`;
    const title = document.createElement("strong");
    title.textContent = provider.label;
    const meta = document.createElement("span");
    meta.textContent = provider.baseUrl || provider.apiKeyEnv || "未设置请求地址";
    const status = document.createElement("small");
    status.textContent = providerStatusText(provider);
    button.append(title, meta, status);
    button.addEventListener("click", () => {
      activeProviderId = provider.id;
      renderProviderManager();
    });
    list.appendChild(button);
  });
}

function renderProviderModels(models) {
  providerFormModels = models || [];
  byId("modelCountLabel").textContent = `${providerFormModels.length} 个`;
  const datalist = byId("providerModelOptions");
  datalist.innerHTML = "";
  providerFormModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    datalist.appendChild(option);
  });
  const container = byId("providerModels");
  container.innerHTML = "";
  if (providerFormModels.length === 0) {
    const empty = document.createElement("span");
    empty.className = "muted-text";
    empty.textContent = "尚未缓存模型。保存 API Key 后可以点击获取模型列表。";
    container.appendChild(empty);
    return;
  }
  providerFormModels.forEach((model) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "model-chip";
    chip.textContent = model;
    chip.title = "点击复制模型名";
    chip.addEventListener("click", async () => {
      await navigator.clipboard?.writeText(model);
      byId("providerMessage").textContent = `已复制：${model}`;
    });
    container.appendChild(chip);
  });
}

function updateProviderModelsInMemory(providerId, models) {
  const provider = providerById(providerId);
  if (!provider) return;
  provider.cachedModels = models;
  refreshTaskModelOptions();
  renderProviderList(providerRegistry.providers);
}

function fillProviderForm(provider) {
  byId("providerMessage").textContent = "";
  byId("providerIdInput").value = provider?.id || "";
  byId("providerLabelInput").value = provider?.label || "";
  byId("providerApiStyleSelect").value = provider?.apiStyle || "openai_compatible";
  byId("providerBaseUrlInput").value = provider?.baseUrl || "";
  byId("providerApiKeyInput").value = "";
  byId("providerApiKeyInput").placeholder = provider?.isConfigured
    ? "已保存或已从环境变量读取，留空继续沿用"
    : "填写 API Key";
  byId("providerHomepageInput").value = provider?.homepage || "";
  byId("providerTimeoutInput").value = provider?.timeoutSeconds || 300;
  byId("providerNotesInput").value = provider?.notes || "";
  byId("providerImageModelInput").value = provider?.taskModels?.image || "";
  byId("providerReviewModelInput").value = provider?.taskModels?.review || "";
  byId("providerQuestionModelInput").value = provider?.taskModels?.questions || "";
  byId("deleteProviderButton").disabled = !provider || provider.source !== "local";
  renderProviderModels(provider?.cachedModels || []);
}

function newProviderForm() {
  activeProviderId = null;
  fillProviderForm(null);
  byId("providerLabelInput").focus();
}

function providerFormPayload() {
  return {
    id: byId("providerIdInput").value || null,
    label: byId("providerLabelInput").value.trim(),
    apiStyle: byId("providerApiStyleSelect").value,
    apiKey: byId("providerApiKeyInput").value.trim() || null,
    baseUrl: byId("providerBaseUrlInput").value.trim() || null,
    homepage: byId("providerHomepageInput").value.trim() || null,
    notes: byId("providerNotesInput").value.trim() || null,
    timeoutSeconds: Number(byId("providerTimeoutInput").value || 300),
    cachedModels: providerFormModels,
    taskModels: {
      image: byId("providerImageModelInput").value.trim(),
      review: byId("providerReviewModelInput").value.trim(),
      questions: byId("providerQuestionModelInput").value.trim(),
    },
  };
}

function syncReviewStrengthFromGate() {
  if (!byId("gateEnabled").checked) {
    byId("reviewStrengthSelect").value = "OFF";
  } else if (byId("reviewStrengthSelect").value === "OFF") {
    byId("reviewStrengthSelect").value = "STANDARD";
  }
}

function syncGateFromReviewStrength() {
  byId("gateEnabled").checked = byId("reviewStrengthSelect").value !== "OFF";
}

async function saveProvider(event) {
  event.preventDefault();
  const message = byId("providerMessage");
  const payload = providerFormPayload();
  if (!payload.label) {
    message.textContent = "请填写供应商名称。";
    return;
  }
  const response = await fetch("/api/providers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    message.textContent = data.detail || "保存失败。";
    return;
  }
  activeProviderId = data.provider.id;
  await loadConfig({ preservePipeline: true });
  activeProviderId = data.provider.id;
  renderProviderManager();
  byId("providerMessage").textContent = "供应商已保存。本地配置会在下次打开时继续可用。";
}

async function fetchModelsForForm() {
  const message = byId("providerMessage");
  message.textContent = "正在获取模型列表...";
  const providerId = byId("providerIdInput").value || null;
  const response = await fetch("/api/providers/fetch-models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      providerId,
      apiKey: byId("providerApiKeyInput").value.trim() || null,
      baseUrl: byId("providerBaseUrlInput").value.trim() || null,
      timeoutSeconds: Number(byId("providerTimeoutInput").value || 60),
      cacheProviderId: providerId,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    message.textContent = data.detail || "获取模型列表失败。";
    return;
  }
  renderProviderModels(data.models || []);
  if (providerId) updateProviderModelsInMemory(providerId, data.models || []);
  message.textContent = `已获取 ${data.count} 个模型。表单内容已保留，点击保存后会缓存到本地配置。`;
}

async function deleteActiveProvider() {
  const providerId = byId("providerIdInput").value;
  if (!providerId) return;
  const response = await fetch(`/api/providers/${encodeURIComponent(providerId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const data = await response.json();
    byId("providerMessage").textContent = data.detail || "删除失败。";
    return;
  }
  activeProviderId = null;
  await loadConfig({ preservePipeline: true });
}

byId("pipelineSelect").addEventListener("change", updateOverrideDefaults);
Object.values(MODEL_FIELDS).forEach((field) => {
  byId(field.providerSelect).addEventListener("change", () => {
    fillModelDatalist(field.datalist, byId(field.providerSelect).value);
    byId(field.modelInput).value = providerTaskModel(
      byId(field.providerSelect).value,
      field.taskModelKey,
    );
    const pipeline = selectedPipeline();
    if (pipeline) renderModelStatus(pipeline);
  });
  byId(field.modelInput).addEventListener("input", () => {
    const pipeline = selectedPipeline();
    if (pipeline) renderModelStatus(pipeline);
  });
});
byId("generateButton").addEventListener("click", generate);
byId("printButton").addEventListener("click", () => window.print());
byId("saveEvaluationButton").addEventListener("click", saveEvaluation);
byId("clearButton").addEventListener("click", () => {
  byId("caseId").value = "";
  byId("conclusionText").value = "";
  byId("findingsText").value = "";
  clearReferenceImage();
  byId("caseId").focus();
});
byId("gateEnabled").addEventListener("change", syncReviewStrengthFromGate);
byId("reviewStrengthSelect").addEventListener("change", syncGateFromReviewStrength);
byId("referenceImageInput").addEventListener("change", handleReferenceImageChange);
byId("clearReferenceImageButton").addEventListener("click", clearReferenceImage);
byId("historyButton").addEventListener("click", openHistoryDialog);
byId("closeHistoryDialogButton").addEventListener("click", closeHistoryDialog);
byId("refreshHistoryButton").addEventListener("click", loadHistory);
byId("historyCaseFilter").addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadHistory();
});
byId("apiSettingsButton").addEventListener("click", openProviderDialog);
byId("quickApiSettingsButton").addEventListener("click", openProviderDialog);
byId("closeProviderDialogButton").addEventListener("click", closeProviderDialog);
byId("providerDialog").addEventListener("click", (event) => {
  if (event.target === byId("providerDialog")) closeProviderDialog();
});
byId("historyDialog").addEventListener("click", (event) => {
  if (event.target === byId("historyDialog")) closeHistoryDialog();
});
byId("addProviderButton").addEventListener("click", newProviderForm);
byId("providerForm").addEventListener("submit", saveProvider);
byId("fetchModelsButton").addEventListener("click", fetchModelsForForm);
byId("deleteProviderButton").addEventListener("click", deleteActiveProvider);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !byId("providerDialog").hidden) closeProviderDialog();
  if (event.key === "Escape" && !byId("historyDialog").hidden) closeHistoryDialog();
});

renderErrorTypeCheckboxes();
loadConfig();
