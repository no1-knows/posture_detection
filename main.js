import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// =============================================================================
// 定数定義
// =============================================================================

const CONFIG = {
  SMOOTHING_FRAMES: 5,
  DEVIATION_THRESHOLD: 0.08,
  MIN_VISIBILITY: 0.5,
  NG_SOUND_COOLDOWN_MS: 1000,
  NG_SOUND_FREQUENCY: 440,
  NG_SOUND_DURATION: 0.15
};

const LANDMARK_INDEX = {
  NOSE: 0,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28
};

// =============================================================================
// グローバル状態
// =============================================================================

let poseLandmarker = null;
let video = null;
let canvas = null;
let ctx = null;
let drawingUtils = null;
let lastVideoTime = -1;
let pointsBuffer = [];
let lastNGSoundTime = 0;
let audioContext = null;

// =============================================================================
// DOM要素
// =============================================================================

const elements = {
  video: () => document.getElementById("video"),
  canvas: () => document.getElementById("canvas"),
  loading: () => document.getElementById("loading"),
  result: () => document.getElementById("result"),
  deviation: () => document.getElementById("deviation"),
  guide: () => document.getElementById("guide")
};

// =============================================================================
// 初期化
// =============================================================================

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: 640, height: 480 }
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    return true;
  } catch (error) {
    console.error("カメラの起動に失敗:", error);
    updateGuide("カメラの起動に失敗しました。権限を確認してください。", "error");
    return false;
  }
}

async function initPoseLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1
    });
    drawingUtils = new DrawingUtils(ctx);
    return true;
  } catch (error) {
    console.error("PoseLandmarkerの初期化に失敗:", error);
    updateGuide("姿勢推定モデルの読み込みに失敗しました。", "error");
    return false;
  }
}

// =============================================================================
// 座標抽出
// =============================================================================

function getBestPoint(landmarks, leftIndex, rightIndex) {
  const left = landmarks[leftIndex];
  const right = landmarks[rightIndex];
  const leftVis = left?.visibility ?? 0;
  const rightVis = right?.visibility ?? 0;

  if (leftVis >= CONFIG.MIN_VISIBILITY && rightVis >= CONFIG.MIN_VISIBILITY) {
    return {
      x: (left.x + right.x) / 2,
      y: (left.y + right.y) / 2,
      visibility: (leftVis + rightVis) / 2
    };
  } else if (leftVis > rightVis) {
    return { x: left.x, y: left.y, visibility: leftVis };
  } else {
    return { x: right.x, y: right.y, visibility: rightVis };
  }
}

function extractKeyPoints(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const head = getBestPoint(
    landmarks,
    LANDMARK_INDEX.LEFT_EAR,
    LANDMARK_INDEX.RIGHT_EAR
  );
  const hip = getBestPoint(
    landmarks,
    LANDMARK_INDEX.LEFT_HIP,
    LANDMARK_INDEX.RIGHT_HIP
  );
  const ankle = getBestPoint(
    landmarks,
    LANDMARK_INDEX.LEFT_ANKLE,
    LANDMARK_INDEX.RIGHT_ANKLE
  );

  if (
    head.visibility < CONFIG.MIN_VISIBILITY ||
    hip.visibility < CONFIG.MIN_VISIBILITY ||
    ankle.visibility < CONFIG.MIN_VISIBILITY
  ) {
    return null;
  }

  return { head, hip, ankle };
}

// =============================================================================
// スムージング
// =============================================================================

function smoothPoints(newPoints) {
  pointsBuffer.push(newPoints);
  if (pointsBuffer.length > CONFIG.SMOOTHING_FRAMES) {
    pointsBuffer.shift();
  }

  const avgHead = { x: 0, y: 0 };
  const avgHip = { x: 0, y: 0 };
  const avgAnkle = { x: 0, y: 0 };

  for (const p of pointsBuffer) {
    avgHead.x += p.head.x;
    avgHead.y += p.head.y;
    avgHip.x += p.hip.x;
    avgHip.y += p.hip.y;
    avgAnkle.x += p.ankle.x;
    avgAnkle.y += p.ankle.y;
  }

  const n = pointsBuffer.length;
  avgHead.x /= n;
  avgHead.y /= n;
  avgHip.x /= n;
  avgHip.y /= n;
  avgAnkle.x /= n;
  avgAnkle.y /= n;

  return { head: avgHead, hip: avgHip, ankle: avgAnkle };
}

// =============================================================================
// 判定ロジック
// =============================================================================

function pointToLineDistance(point, lineStart, lineEnd) {
  const A = point.x - lineStart.x;
  const B = point.y - lineStart.y;
  const C = lineEnd.x - lineStart.x;
  const D = lineEnd.y - lineStart.y;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;

  if (lenSq === 0) {
    return Math.hypot(A, B);
  }

  const param = dot / lenSq;
  let xx, yy;

  if (param < 0) {
    xx = lineStart.x;
    yy = lineStart.y;
  } else if (param > 1) {
    xx = lineEnd.x;
    yy = lineEnd.y;
  } else {
    xx = lineStart.x + param * C;
    yy = lineStart.y + param * D;
  }

  return Math.hypot(point.x - xx, point.y - yy);
}

function computeDeviation(head, hip, ankle) {
  const distance = pointToLineDistance(hip, head, ankle);
  const lineLength = Math.hypot(ankle.x - head.x, ankle.y - head.y);

  if (lineLength === 0) return 1;

  return distance / lineLength;
}

function judgeAlignment(head, hip, ankle) {
  const deviation = computeDeviation(head, hip, ankle);
  const isOK = deviation <= CONFIG.DEVIATION_THRESHOLD;

  return {
    isOK,
    deviation,
    deviationPercent: (deviation * 100).toFixed(1)
  };
}

// =============================================================================
// ガイドメッセージ
// =============================================================================

function checkSideView(landmarks) {
  if (!landmarks || landmarks.length === 0) return false;

  const leftShoulder = landmarks[LANDMARK_INDEX.LEFT_SHOULDER];
  const rightShoulder = landmarks[LANDMARK_INDEX.RIGHT_SHOULDER];

  if (!leftShoulder || !rightShoulder) return true;

  const shoulderXDiff = Math.abs(leftShoulder.x - rightShoulder.x);
  return shoulderXDiff < 0.15;
}

function getGuideMessage(landmarks, keyPoints) {
  if (!landmarks || landmarks.length === 0) {
    return { message: "体全体が映るように立ってください", type: "warning" };
  }

  if (!checkSideView(landmarks)) {
    return { message: "カメラに対して横を向いてください", type: "warning" };
  }

  if (!keyPoints) {
    return { message: "頭・腰・足首が見えるように立ってください", type: "warning" };
  }

  if (pointsBuffer.length < CONFIG.SMOOTHING_FRAMES) {
    return { message: "姿勢を安定させてください...", type: "waiting" };
  }

  return null;
}

// =============================================================================
// 描画
// =============================================================================

function drawOverlay(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!landmarks || landmarks.length === 0) return;

  drawingUtils.drawLandmarks(landmarks, {
    radius: 4,
    color: "#00FF00",
    fillColor: "#00FF00"
  });

  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#00FF00",
    lineWidth: 2
  });

  const keyPoints = extractKeyPoints(landmarks);
  if (keyPoints) {
    const { head, hip, ankle } = keyPoints;
    const w = canvas.width;
    const h = canvas.height;

    ctx.beginPath();
    ctx.moveTo(head.x * w, head.y * h);
    ctx.lineTo(ankle.x * w, ankle.y * h);
    ctx.strokeStyle = "#FFFF00";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(head.x * w, head.y * h, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#FF00FF";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(hip.x * w, hip.y * h, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#FF00FF";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(ankle.x * w, ankle.y * h, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#FF00FF";
    ctx.fill();
  }
}

// =============================================================================
// UI更新
// =============================================================================

function updateResult(isOK, deviationPercent) {
  const resultEl = elements.result();
  resultEl.textContent = isOK ? "OK" : "NG";
  resultEl.className = `result ${isOK ? "result-ok" : "result-ng"}`;

  elements.deviation().textContent = `ずれ量: ${deviationPercent}%`;
}

function updateGuide(message, type = "warning") {
  const guideEl = elements.guide();
  guideEl.textContent = message;
  guideEl.className = `guide guide-${type}`;
}

function clearResult() {
  const resultEl = elements.result();
  resultEl.textContent = "---";
  resultEl.className = "result result-waiting";
  elements.deviation().textContent = "ずれ量: ---%";
}

// =============================================================================
// NG音
// =============================================================================

function playNGSound() {
  const now = Date.now();
  if (now - lastNGSoundTime < CONFIG.NG_SOUND_COOLDOWN_MS) return;
  lastNGSoundTime = now;

  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  const oscillator = audioContext.createOscillator();
  const gainNode = audioContext.createGain();

  oscillator.connect(gainNode);
  gainNode.connect(audioContext.destination);

  oscillator.frequency.value = CONFIG.NG_SOUND_FREQUENCY;
  oscillator.type = "square";

  gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(
    0.01,
    audioContext.currentTime + CONFIG.NG_SOUND_DURATION
  );

  oscillator.start(audioContext.currentTime);
  oscillator.stop(audioContext.currentTime + CONFIG.NG_SOUND_DURATION);
}

// =============================================================================
// メインループ
// =============================================================================

function detectPose() {
  if (!poseLandmarker || !video || video.readyState < 2) {
    requestAnimationFrame(detectPose);
    return;
  }

  if (video.currentTime === lastVideoTime) {
    requestAnimationFrame(detectPose);
    return;
  }

  lastVideoTime = video.currentTime;

  const result = poseLandmarker.detectForVideo(video, performance.now());
  const landmarks = result.landmarks[0] ?? null;

  drawOverlay(landmarks);

  const keyPoints = extractKeyPoints(landmarks);
  const guideInfo = getGuideMessage(landmarks, keyPoints);

  if (guideInfo) {
    updateGuide(guideInfo.message, guideInfo.type);
    clearResult();
    if (guideInfo.type === "warning") {
      pointsBuffer = [];
    }
  } else if (keyPoints) {
    const smoothed = smoothPoints(keyPoints);
    const judgment = judgeAlignment(smoothed.head, smoothed.hip, smoothed.ankle);

    updateResult(judgment.isOK, judgment.deviationPercent);

    if (judgment.isOK) {
      updateGuide("姿勢が整っています", "ok");
    } else {
      updateGuide("腰の位置を調整してください", "warning");
      playNGSound();
    }
  }

  requestAnimationFrame(detectPose);
}

// =============================================================================
// エントリーポイント
// =============================================================================

async function main() {
  video = elements.video();
  canvas = elements.canvas();
  ctx = canvas.getContext("2d");

  const cameraOK = await initCamera();
  if (!cameraOK) return;

  const landmarkerOK = await initPoseLandmarker();
  if (!landmarkerOK) return;

  elements.loading().classList.add("hidden");
  updateGuide("横を向いて全身が映るように立ってください");

  detectPose();
}

main();
