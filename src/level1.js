import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { getStream } from './devices.js';

const MODEL_URL = new URL('models/mobilenet/model.json', document.baseURI).href;
const CLASSES = [
  { emoji: '🖊️', name: 'Cây bút' },
  { emoji: '📘', name: 'Quyển sách' },
  { emoji: '🍶', name: 'Chai nước' },
];
const MAX_PER_CLASS = 200;

const video = document.getElementById('l1Video');
const hint = document.getElementById('l1Hint');
const resultBox = document.getElementById('l1Result');
const classBox = document.getElementById('l1Classes');
const trainBtn = document.getElementById('l1Train');
const resetBtn = document.getElementById('l1Reset');
const screenEl = document.getElementById('level1');

let featureModel = null;           // MobileNet đã cắt bớt, dùng để rút đặc trưng ảnh
const classifier = knnClassifier.create();
let stream = null;
let active = false;
let trained = false;
let captureTimer = null;

const rows = [];                   // các phần tử giao diện của từng lớp

// ---------- Giao diện các lớp ----------
CLASSES.forEach((c, i) => {
  const row = document.createElement('div');
  row.className = 'l1-class';
  row.innerHTML = `
    <span style="font-size:1.6rem">${c.emoji}</span>
    <input value="${c.name}" maxlength="20" />
    <span class="count">0 ảnh</span>
    <button class="hold">📸 Giữ để chụp</button>
    <div class="l1-bar"><div></div></div>`;
  classBox.appendChild(row);

  const btn = row.querySelector('.hold');
  btn.addEventListener('pointerdown', () => startCapture(i));
  ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) =>
    btn.addEventListener(ev, stopCapture)
  );

  rows.push({
    input: row.querySelector('input'),
    count: row.querySelector('.count'),
    bar: row.querySelector('.l1-bar div'),
  });
});

const countOf = (i) => classifier.getClassExampleCount()[i] || 0;
const totalCount = () => CLASSES.reduce((sum, _, i) => sum + countOf(i), 0);
const filledClasses = () => CLASSES.filter((_, i) => countOf(i) > 0).length;
const name = (i) => rows[i].input.value || CLASSES[i].name;

function refreshCounts() {
  rows.forEach((r, i) => (r.count.textContent = countOf(i) + ' ảnh'));
}

function setHint(text) { hint.textContent = text; }

// ---------- Rút đặc trưng từ khung hình ----------
function embed() {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(video);
    const [h, w] = img.shape;
    const s = Math.min(h, w);                       // cắt hình vuông ở giữa
    const square = img.slice([Math.floor((h - s) / 2), Math.floor((w - s) / 2), 0], [s, s, 3]);
    const x = tf.image.resizeBilinear(square, [224, 224]).toFloat().div(127.5).sub(1).expandDims(0);
    return featureModel.predict(x).mean([1, 2]);    // vector đặc trưng
  });
}

const videoReady = () => video.readyState >= 2 && video.videoWidth > 0;

// ---------- Thu thập ảnh ----------
function addOne(i) {
  if (!featureModel || !videoReady() || countOf(i) >= MAX_PER_CLASS) return;
  const feat = embed();
  classifier.addExample(feat, i);
  feat.dispose();
  refreshCounts();
}

function startCapture(i) {
  stopCapture();
  addOne(i);
  captureTimer = setInterval(() => addOne(i), 100);
}

function stopCapture() {
  clearInterval(captureTimer);
  captureTimer = null;
}

// ---------- Huấn luyện ----------
trainBtn.addEventListener('click', async () => {
  if (filledClasses() < 2) {
    setHint('⚠️ Hãy chụp ảnh ít nhất 2 loại đồ vật trước khi huấn luyện (mỗi loại khoảng 20-30 ảnh).');
    return;
  }
  trainBtn.disabled = true;
  setHint('🧠 AI đang học từ ' + totalCount() + ' bức ảnh...');
  await new Promise((r) => setTimeout(r, 1200));
  trained = true;
  trainBtn.disabled = false;
  trainBtn.textContent = '🧠 Huấn luyện lại';
  screenEl.classList.add('l1-trained');
  setHint('🔎 Hãy đưa đồ vật lên camera. Muốn AI giỏi hơn? Cứ chụp thêm ảnh rồi huấn luyện lại!');
});

resetBtn.addEventListener('click', () => {
  classifier.clearAllClasses();
  trained = false;
  screenEl.classList.remove('l1-trained');
  trainBtn.textContent = '🧠 Huấn luyện AI';
  resultBox.textContent = 'AI chưa được huấn luyện';
  rows.forEach((r) => (r.bar.style.width = '0%'));
  refreshCounts();
  setHint('Bước 1: Chọn tên đồ vật, đưa nó lên camera và GIỮ nút "Chụp" để dạy AI.');
});

// ---------- Vòng lặp dự đoán ----------
async function loop() {
  if (!active) return;
  if (trained && totalCount() > 0 && videoReady()) {
    const feat = embed();
    try {
      const res = await classifier.predictClass(feat, Math.min(10, totalCount()));
      showResult(res);
    } catch (e) {
      console.warn(e);
    }
    feat.dispose();
  }
  setTimeout(loop, 120);
}

function showResult(res) {
  CLASSES.forEach((_, i) => {
    rows[i].bar.style.width = Math.round((res.confidences[i] || 0) * 100) + '%';
  });
  const best = res.confidences[res.label] || 0;
  const pct = Math.round(best * 100);
  resultBox.textContent =
    best >= 0.6
      ? `AI đoán: ${name(+res.label)} (${pct}%)`
      : `🤔 AI chưa chắc lắm... (${name(+res.label)} ${pct}%)`;
}

// ---------- Vào / rời màn chơi ----------
async function loadModel() {
  if (featureModel) return;
  await tf.ready();
  const mobilenet = await tf.loadLayersModel(MODEL_URL);
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  featureModel = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  tf.tidy(() => featureModel.predict(tf.zeros([1, 224, 224, 3]))); // chạy thử cho nóng máy
}

export async function enter() {
  active = true;
  setHint('⏳ Đang tải "bộ não" AI và mở camera...');
  try {
    await loadModel();
    const s = await getStream('video');
    if (!active) { s.getTracks().forEach((t) => t.stop()); return; } // đã rời màn trong lúc chờ
    stream = s;
    video.srcObject = stream;
    await video.play();
    refreshCounts();
    setHint(trained
      ? '🔎 Hãy đưa đồ vật lên camera.'
      : 'Bước 1: Chọn tên đồ vật, đưa nó lên camera và GIỮ nút "Chụp" để dạy AI.');
    loop();
  } catch (e) {
    console.error(e);
    setHint('❌ Lỗi: ' + e.message + ' (Kiểm tra camera trong ⚙️ Cài đặt)');
  }
}

export function leave() {
  active = false;
  stopCapture();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
}