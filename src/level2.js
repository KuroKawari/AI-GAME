import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { getStream } from './devices.js';

const CLASSES = [
  { emoji: '🔇', name: 'Tiếng ồn nền', hero: '😴', anim: '' },
  { emoji: '👏', name: 'Vỗ tay', hero: '🦘', anim: 'clap' },
  { emoji: '🎶', name: 'Huýt sáo', hero: '💃', anim: 'whistle' },
];
const BANDS = 32;            // số dải tần dùng làm đặc trưng
const BINS_USED = 256;       // chỉ dùng khoảng 0 - 11 kHz, đủ cho vỗ tay và huýt sáo
const MAX_PER_CLASS = 300;
const SOUND_MARGIN_DB = 6;   // âm thanh phải to hơn nền ít nhất 6 dB mới được lưu

const hint = document.getElementById('l2Hint');
const heroEl = document.getElementById('l2Hero');
const resultBox = document.getElementById('l2Result');
const classBox = document.getElementById('l2Classes');
const trainBtn = document.getElementById('l2Train');
const resetBtn = document.getElementById('l2Reset');
const screenEl = document.getElementById('level2');
const canvas = document.getElementById('l2Spectrum');
const noiseBar = document.getElementById('l2NoiseBar');
const noiseLabel = document.getElementById('l2NoiseLabel');
const cctx = canvas.getContext('2d');

const classifier = knnClassifier.create();
let stream = null;
let audioCtx = null;
let analyser = null;
let freq = null;
let active = false;
let runId = 0;
let trained = false;
let capturing = null;        // chỉ số lớp đang thu, hoặc null
let floor = null;            // mức ồn nền ước lượng (dB)
let heroUntil = 0;
let shownLevel = 0;

const rows = [];

// ---------- Giao diện các lớp ----------
CLASSES.forEach((c, i) => {
  const row = document.createElement('div');
  row.className = 'l1-class';
  row.innerHTML = `
    <span style="font-size:1.6rem">${c.emoji}</span>
    <input value="${c.name}" maxlength="20" />
    <span class="count">0 mẫu</span>
    <button class="hold">🎙️ Giữ để thu</button>
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
const setHint = (t) => (hint.textContent = t);

function refreshCounts() {
  rows.forEach((r, i) => (r.count.textContent = countOf(i) + ' mẫu'));
}

// ---------- Đọc âm thanh ----------
// Trả về năng lượng (dB) của từng dải tần và mức ồn trung bình
function readSpectrum() {
  analyser.getFloatFrequencyData(freq);
  const per = BINS_USED / BANDS;
  const db = new Array(BANDS);
  let sum = 0;
  for (let b = 0; b < BANDS; b++) {
    let s = 0;
    for (let k = 0; k < per; k++) {
      const v = freq[b * per + k];
      s += Number.isFinite(v) ? Math.max(-100, v) : -100;
    }
    db[b] = s / per;
    sum += db[b];
  }
  return { db, level: sum / BANDS };
}

// Đổi dB thành số nằm khoảng -1..1 để đưa cho KNN
const toFeatures = (db) => db.map((v) => Math.max(-1, Math.min(1, (v + 65) / 35)));

function updateFloor(level) {
  if (floor === null) floor = level;
  else if (level < floor + 4) floor += (level - floor) * 0.05;   // nền yên tĩnh: bám theo nhanh
  else floor += (level - floor) * 0.003;                          // ồn kéo dài: nền tăng rất chậm
}

// ---------- Vẽ biểu đồ và đồng hồ ----------
function drawSpectrum(db) {
  const w = canvas.width, h = canvas.height, bw = w / BANDS;
  cctx.clearRect(0, 0, w, h);
  db.forEach((v, i) => {
    const n = Math.max(0, Math.min(1, (v + 100) / 70));
    cctx.fillStyle = `hsl(${200 + n * 120}, 80%, 60%)`;
    cctx.fillRect(i * bw + 1, h - n * h, bw - 2, n * h);
  });
}

function updateNoise(level) {
  const pct = Math.max(0, Math.min(100, ((level + 90) / 50) * 100));
  shownLevel += (pct - shownLevel) * 0.3;
  const [text, color] =
    shownLevel < 35 ? ['Yên tĩnh 🟢', '#4caf50']
    : shownLevel < 65 ? ['Hơi ồn 🟡', '#f5b301']
    : ['Rất ồn 🔴', '#e53935'];
  noiseBar.style.width = shownLevel + '%';
  noiseBar.style.background = color;
  noiseLabel.textContent = text;
}

// ---------- Thu thập mẫu ----------
function startCapture(i) {
  capturing = i;
  setHint(i === 0
    ? '🎙️ Đang thu tiếng ồn nền: Cứ để lớp học như bình thường (nói chuyện nhỏ, im lặng...).'
    : '🎙️ Hãy tạo âm thanh thật rõ khi đang giữ nút. AI chỉ lưu những khoảnh khắc có tiếng!');
}

function stopCapture() {
  if (capturing === null) return;
  capturing = null;
  setHint('Đã thu xong. Thu thêm cho đủ 3 loại, rồi bấm "Huấn luyện AI".');
}

function addExample(i, db, level) {
  if (countOf(i) >= MAX_PER_CLASS) return;
  if (i !== 0 && level < floor + SOUND_MARGIN_DB) return; // bỏ qua lúc im lặng giữa 2 tiếng vỗ
  const t = tf.tensor1d(toFeatures(db));
  classifier.addExample(t, i);
  t.dispose();
  refreshCounts();
}

// ---------- Huấn luyện ----------
trainBtn.addEventListener('click', async () => {
  if (filledClasses() < 2) {
    setHint('⚠️ Hãy thu ít nhất 2 loại âm thanh (nên có cả "Tiếng ồn nền") trước khi huấn luyện.');
    return;
  }
  trainBtn.disabled = true;
  setHint('🧠 AI đang học từ ' + totalCount() + ' mẫu âm thanh...');
  await new Promise((r) => setTimeout(r, 1200));
  trained = true;
  trainBtn.disabled = false;
  trainBtn.textContent = '🧠 Huấn luyện lại';
  screenEl.classList.add('l1-trained');
  setHint('🔎 Thử vỗ tay, huýt sáo hoặc nói chuyện xem nhân vật phản ứng thế nào!');
});

resetBtn.addEventListener('click', () => {
  classifier.clearAllClasses();
  trained = false;
  screenEl.classList.remove('l1-trained');
  trainBtn.textContent = '🧠 Huấn luyện AI';
  resultBox.textContent = 'AI chưa được huấn luyện';
  setHero(0);
  rows.forEach((r) => (r.bar.style.width = '0%'));
  refreshCounts();
  setHint('Bước 1: Giữ từng nút để thu mẫu âm thanh cho mỗi loại.');
});

// ---------- Dự đoán ----------
function setHero(i) {
  heroEl.textContent = CLASSES[i].hero;
  heroEl.className = 'hero ' + CLASSES[i].anim;
}

function showResult(res) {
  CLASSES.forEach((_, i) => {
    rows[i].bar.style.width = Math.round((res.confidences[i] || 0) * 100) + '%';
  });
  const label = +res.label;
  const conf = res.confidences[label] || 0;
  const pct = Math.round(conf * 100);
  const now = performance.now();

  if (label !== 0 && conf >= 0.6) {
    // Tiếng vỗ tay rất ngắn nên giữ phản ứng của nhân vật khoảng 0,7 giây
    heroUntil = now + 700;
    setHero(label);
    resultBox.textContent = `AI nghe thấy: ${name(label)} (${pct}%)`;
  } else if (now > heroUntil) {
    setHero(0);
    resultBox.textContent = conf >= 0.6
      ? `AI nghe thấy: ${name(label)} (${pct}%)`
      : `🤔 AI chưa chắc lắm... (${name(label)} ${pct}%)`;
  }
}

// ---------- Vòng lặp chính (20 lần/giây) ----------
async function loop(id) {
  if (!active || id !== runId) return;
  const { db, level } = readSpectrum();
  updateFloor(level);
  drawSpectrum(db);
  updateNoise(level);

  if (capturing !== null) {
    addExample(capturing, db, level);
  } else if (trained && totalCount() > 0) {
    const t = tf.tensor1d(toFeatures(db));
    try {
      const res = await classifier.predictClass(t, Math.min(5, totalCount()));
      if (active && id === runId) showResult(res);
    } catch (e) {
      console.warn(e);
    }
    t.dispose();
  }
  setTimeout(() => loop(id), 50);
}

// ---------- Vào / rời màn chơi ----------
function cleanup() {
  capturing = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  stream = audioCtx = analyser = freq = null;
}

export async function enter() {
  const id = ++runId;
  active = true;
  floor = null;
  setHint('⏳ Đang mở micro...');
  try {
    await tf.ready();
    const s = await getStream('audio');
    if (id !== runId) { s.getTracks().forEach((t) => t.stop()); return; }
    stream = s;
    audioCtx = new AudioContext();
    await audioCtx.resume();
    if (id !== runId) { cleanup(); return; }
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0;   // không làm mượt, để bắt được tiếng vỗ tay ngắn
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    freq = new Float32Array(analyser.frequencyBinCount);
    refreshCounts();
    setHint(trained
      ? '🔎 Thử vỗ tay, huýt sáo hoặc nói chuyện xem nhân vật phản ứng thế nào!'
      : 'Bước 1: Giữ từng nút để thu mẫu âm thanh cho mỗi loại (nhớ thu cả "Tiếng ồn nền").');
    loop(id);
  } catch (e) {
    console.error(e);
    setHint('❌ Lỗi: ' + e.message + ' (Kiểm tra micro trong ⚙️ Cài đặt)');
  }
}

export function leave() {
  active = false;
  runId++;
  cleanup();
}