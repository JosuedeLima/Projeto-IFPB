/**
 * Validador web — captura câmera no navegador e envia frames ao backend YOLO.
 */

const API_BASE = window.location.origin;
const MAX_SEND_WIDTH = 640;
const TARGET_INTERVAL_MS = 150;

const COLORS = {
  pe_abacaxi: { stroke: "#22c55e", fill: "rgba(34, 197, 94, 0.15)" },
  olho_abacaxi: { stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.2)" },
};

const $ = (id) => document.getElementById(id);

const video = $("video");
const overlay = $("overlay");
const placeholder = $("placeholder");
const viewerFrame = document.querySelector(".viewer__frame");
const btnCamera = $("btnCamera");
const btnSwitch = $("btnSwitch");
const btnClearImage = $("btnClearImage");
const fileInput = $("fileInput");
const confRange = $("confRange");
const confValue = $("confValue");
const statusBadge = $("statusBadge");
const modelAlert = $("modelAlert");
const statFps = $("statFps");
const statInfer = $("statInfer");
const statCount = $("statCount");
const statMaxConf = $("statMaxConf");
const detectionList = $("detectionList");
const errorMsg = $("errorMsg");

const ctx = overlay.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let stream = null;
let running = false;
let imageMode = false;
let staticImage = null;
let inferBusy = false;
let facingMode = "environment";
let loopTimer = null;
let frameTimes = [];

/** @type {MediaDeviceInfo[]} */
let videoDevices = [];
let deviceIndex = 0;

function parseApiError(errBody, status) {
  if (!errBody) return `Erro HTTP ${status}`;
  if (typeof errBody.detail === "string") return errBody.detail;
  if (Array.isArray(errBody.detail)) {
    return errBody.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `Erro HTTP ${status}`;
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (!data.ok) throw new Error("Modelo indisponível");

    statusBadge.textContent = "Modelo pronto";
    statusBadge.className = "badge badge--ok";

    const diag = data.diagnostico || {};
    if (diag.alerta) {
      modelAlert.textContent = diag.alerta;
      modelAlert.hidden = false;
      if (diag.max_confianca != null && diag.max_confianca < 0.1) {
        confRange.value = "1";
        updateConfLabel();
      }
    } else if (diag.testado && diag.max_confianca != null) {
      modelAlert.textContent = `Teste em ${diag.imagem}: confiança máxima ${(diag.max_confianca * 100).toFixed(1)}%.`;
      modelAlert.hidden = false;
      modelAlert.style.background = "rgba(34, 197, 94, 0.12)";
      modelAlert.style.color = "#86efac";
      modelAlert.style.borderColor = "rgba(34, 197, 94, 0.35)";
    }
    return true;
  } catch {
    statusBadge.textContent = "Servidor offline";
    statusBadge.className = "badge badge--err";
    showError("Não foi possível conectar ao servidor. Execute: python api_server.py");
    return false;
  }
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = !msg;
}

function getConfidence() {
  return Number(confRange.value) / 100;
}

function updateConfLabel() {
  confValue.textContent = `${confRange.value}%`;
}

async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    videoDevices = devices.filter((d) => d.kind === "videoinput");
    btnSwitch.hidden = videoDevices.length < 2;
    btnSwitch.disabled = videoDevices.length < 2;
  } catch {
    btnSwitch.hidden = true;
  }
}

async function startCamera() {
  showError("");
  imageMode = false;
  staticImage = null;
  btnClearImage.hidden = true;
  fileInput.value = "";

  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Seu navegador não suporta acesso à câmera.");
    return;
  }

  stopCamera(false);

  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };

  if (videoDevices.length > 0 && videoDevices[deviceIndex]?.deviceId) {
    constraints.video = {
      deviceId: { exact: videoDevices[deviceIndex].deviceId },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    };
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    const hint =
      err.name === "NotAllowedError"
        ? "Permissão da câmera negada."
        : err.name === "NotFoundError"
          ? "Nenhuma câmera encontrada."
          : `Erro ao abrir câmera: ${err.message}`;
    showError(hint);
    return;
  }

  video.hidden = false;
  video.srcObject = stream;
  await video.play();

  placeholder.hidden = true;
  btnCamera.textContent = "Parar câmera";
  btnCamera.classList.remove("btn--primary");
  btnCamera.classList.add("btn--danger");

  await listCameras();
  resizeOverlay();
  running = true;
  scheduleLoop();
}

function loadStaticImage(file) {
  stopCamera(false);
  imageMode = true;
  btnClearImage.hidden = false;
  btnCamera.textContent = "Iniciar câmera";
  btnCamera.classList.remove("btn--danger");
  btnCamera.classList.add("btn--primary");

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    staticImage = img;
    video.hidden = true;
    placeholder.hidden = true;
    resizeOverlayForImage(img.width, img.height);
    running = true;
    tick();
    scheduleLoop();
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    showError("Não foi possível carregar a imagem.");
  };
  img.src = url;
}

function stopCamera(resetUi = true) {
  running = false;
  if (loopTimer) {
    clearTimeout(loopTimer);
    loopTimer = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;

  if (resetUi) {
    imageMode = false;
    staticImage = null;
    video.hidden = false;
    placeholder.hidden = false;
    btnClearImage.hidden = true;
    btnCamera.textContent = "Iniciar câmera";
    btnCamera.classList.add("btn--primary");
    btnCamera.classList.remove("btn--danger");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    statFps.textContent = "— FPS";
    statInfer.textContent = "— ms";
    statCount.textContent = "0 detecções";
    renderDetectionList([]);
  }
}

function resizeOverlayForImage(w, h) {
  const frame = viewerFrame.getBoundingClientRect();
  const scale = Math.min(frame.width / w, frame.height / h);
  overlay.width = Math.round(w * scale);
  overlay.height = Math.round(h * scale);
}

function resizeOverlay() {
  if (imageMode && staticImage) {
    resizeOverlayForImage(staticImage.width, staticImage.height);
    return;
  }
  const w = video.videoWidth || overlay.clientWidth;
  const h = video.videoHeight || overlay.clientHeight;
  if (!w || !h) return;
  overlay.width = w;
  overlay.height = h;
}

function scheduleLoop() {
  if (!running) return;
  loopTimer = setTimeout(async () => {
    await tick();
    scheduleLoop();
  }, TARGET_INTERVAL_MS);
}

async function tick() {
  if (!running || inferBusy) return;
  if (!imageMode && video.readyState < 2) return;

  resizeOverlay();
  const frameBlob = await captureFrameBlob();
  if (!frameBlob) return;

  inferBusy = true;
  viewerFrame.classList.add("viewer__frame--processing");
  try {
    const t0 = performance.now();
    const result = await sendFrame(frameBlob);
    const elapsed = performance.now() - t0;

    recordFps();
    statInfer.textContent = `${result.tempo_ms ?? Math.round(elapsed)} ms`;
    statCount.textContent = `${result.deteccoes.length} detecção(ões)`;

    const maxConf = result.deteccoes.length > 0 
      ? Math.max(...result.deteccoes.map(d => d.confianca)) 
      : 0;
    statMaxConf.textContent = maxConf > 0 
      ? `${(maxConf * 100).toFixed(1)}% Confiança` 
      : "— Confiança";

    if (imageMode && staticImage) {
      drawStaticImage();
    }

    drawDetections(result.deteccoes, result.largura, result.altura);
    renderDetectionList(result.deteccoes);
    showError("");
  } catch (err) {
    showError(err.message || "Falha na inferência.");
  } finally {
    inferBusy = false;
    viewerFrame.classList.remove("viewer__frame--processing");
  }
}

function drawStaticImage() {
  if (!staticImage) return;
  ctx.drawImage(staticImage, 0, 0, overlay.width, overlay.height);
}

function recordFps() {
  const now = performance.now();
  frameTimes.push(now);
  frameTimes = frameTimes.filter((t) => now - t < 1000);
  statFps.textContent = `${frameTimes.length} FPS`;
}

async function captureFrameBlob() {
  if (imageMode && staticImage) {
    captureCanvas.width = staticImage.width;
    captureCanvas.height = staticImage.height;
    captureCtx.drawImage(staticImage, 0, 0);
  } else {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;

    let w = vw;
    let h = vh;
    if (w > MAX_SEND_WIDTH) {
      const scale = MAX_SEND_WIDTH / w;
      w = MAX_SEND_WIDTH;
      h = Math.round(vh * scale);
    }
    captureCanvas.width = w;
    captureCanvas.height = h;
    captureCtx.drawImage(video, 0, 0, w, h);
  }

  return new Promise((resolve) => {
    captureCanvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8);
  });
}

async function sendFrame(blob) {
  const conf = getConfidence();
  const form = new FormData();
  form.append("imagem", blob, "frame.jpg");

  const res = await fetch(`${API_BASE}/api/detect?conf=${conf.toFixed(2)}`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(parseApiError(err, res.status));
  }
  return res.json();
}

function drawDetections(deteccoes, srcW, srcH) {
  const dw = overlay.width;
  const dh = overlay.height;
  if (!dw || !dh) return;

  if (!imageMode) {
    ctx.clearRect(0, 0, dw, dh);
  } else if (staticImage) {
    drawStaticImage();
  }

  const sx = dw / (srcW || dw);
  const sy = dh / (srcH || dh);

  for (const det of deteccoes) {
    const [x1, y1, x2, y2] = det.bbox;
    const px1 = x1 * sx;
    const py1 = y1 * sy;
    const pw = (x2 - x1) * sx;
    const ph = (y2 - y1) * sy;

    const style = COLORS[det.classe] || COLORS.pe_abacaxi;
    const label = `${det.classe} ${(det.confianca * 100).toFixed(0)}%`;

    ctx.fillStyle = style.fill;
    ctx.fillRect(px1, py1, pw, ph);

    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = Math.max(2, dw / 400);
    ctx.strokeRect(px1, py1, pw, ph);

    ctx.font = `600 ${Math.max(11, Math.round(dw / 45))}px system-ui, sans-serif`;
    const metrics = ctx.measureText(label);
    const pad = 4;
    const lh = Math.max(16, Math.round(dw / 40));
    const ly = Math.max(py1 - 2, lh);

    ctx.fillStyle = style.stroke;
    ctx.fillRect(px1, ly - lh, metrics.width + pad * 2, lh + pad);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, px1 + pad, ly - 4);
  }
}

function renderDetectionList(deteccoes) {
  detectionList.innerHTML = "";
  if (!deteccoes.length) {
    const li = document.createElement("li");
    li.className = "detection-list__empty";
    li.textContent = running
      ? "Nenhum objeto acima do limiar. Baixe a confiança mínima."
      : "Nenhuma detecção ainda.";
    detectionList.appendChild(li);
    return;
  }

  const sorted = [...deteccoes].sort((a, b) => b.confianca - a.confianca);
  for (const det of sorted) {
    const li = document.createElement("li");
    const isOlho = det.classe === "olho_abacaxi";
    li.className = `detection-list__item detection-list__item--${isOlho ? "olho" : "pe"}`;
    li.innerHTML = `
      <span>${det.classe}</span>
      <span class="detection-list__conf">${(det.confianca * 100).toFixed(1)}%</span>
    `;
    detectionList.appendChild(li);
  }
}

function switchCamera() {
  if (videoDevices.length > 1) {
    deviceIndex = (deviceIndex + 1) % videoDevices.length;
    if (running && !imageMode) {
      startCamera();
      return;
    }
  }
  facingMode = facingMode === "environment" ? "user" : "environment";
  if (running && !imageMode) startCamera();
}

btnCamera.addEventListener("click", () => {
  if (running && !imageMode) stopCamera();
  else if (!running) startCamera();
});

btnSwitch.addEventListener("click", switchCamera);
btnClearImage.addEventListener("click", () => stopCamera());
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) loadStaticImage(file);
});
confRange.addEventListener("input", () => {
  updateConfLabel();
  if (running) tick();
});

window.addEventListener("resize", resizeOverlay);
video.addEventListener("loadedmetadata", resizeOverlay);

updateConfLabel();
checkHealth();
