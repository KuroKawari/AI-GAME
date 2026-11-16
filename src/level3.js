import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { getStream } from './devices.js';

const MODEL_URL = new URL('models/movenet/model.json', document.baseURI).href;

const MIN_SCORE = 0.3;     // điểm khớp phải được AI tin ít nhất 30% mới dùng
const WRIST_SCORE = 0.2;   // cổ tay hay bị khuất nên cho ngưỡng thấp hơn
const ROUNDS = 8;
const ROUND_MS = 7000;     // thời gian mỗi lượt
const HOLD_MS = 700;       // phải giữ tư thế bao lâu mới tính

const POSES = [
  { id: 'up',   emoji: '🙌', name: 'Giơ hai tay lên cao', rule: 'Cả hai cổ tay cao hơn mũi' },
  { id: 'one',  emoji: '🙋', name: 'Giơ một tay lên',     rule: 'Một cổ tay cao hơn mũi, cổ tay kia thấp hơn vai' },
  { id: 'wide', emoji: '🤸', name: 'Dang hai tay ra',     rule: 'Hai cổ tay ngang vai và cách xa người' },
  { id: 'bow',  emoji: '🙇', name: 'Cúi đầu xuống',       rule: 'Mũi hạ thấp gần ngang vai' },
];
const OTHERS = {
  idle: { emoji: '🧍', name: 'Đứng yên / tư thế khác' },
  none: { emoji: '❓', name: '' },
};
const info = (id) => POSES.find((p) => p.id === id) || OTHERS[id];

// Các đoạn nối giữa 17 điểm khớp của MoveNet
const EDGES = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

const video = document.getElementById('l3Video');
const canvas = document.getElementById('l3Canvas');
const ctx = canvas.getContext('2d');
const hint = document.getElementById('l3Hint');
const poseBox = document.getElementById('l3Pose');
const confBox = document.getElementById('l3Conf');
const commandEl = document.getElementById('l3Command');
const scoreEl = document.getElementById('l3Score');
const roundEl = document.getElementById('l3Round');
const timeEl = document.getElementById('l3Time');
const holdBar = document.getElementById('l3Hold');
const guideEl = document.getElementById('l3Guide');
const startBtn = document.getElementById('l3Start');

let detector = null;
let stream = null;
let active = false;
let runId = 0;
let history = [];
let stable = 'none';
let lastWhy = '';

const setHint = (t) => (hint.textContent = t);

// Bảng hướng dẫn: mỗi tư thế kèm "luật" AI dùng để nhận ra
guideEl.innerHTML = POSES.map(
  (p) => `<div>${p.emoji} <b>${p.name}</b><br><small>Luật: ${p.rule}</small></div>`
).join('');

// ---------- Từ điểm khớp -> tên tư thế ----------
function classify(pose) {
  if (!pose) return { id: 'none', why: 'AI chưa thấy ai trong khung hình' };
  const k = {};
  pose.keypoints.forEach((p) => (k[p.name] = p));
  const nose = k.nose, ls = k.left_shoulder, rs = k.right_shoulder;
  if (![nose, ls, rs].every((p) => p && p.score >= MIN_SCORE)) {
    return { id: 'none', why: 'AI chưa thấy rõ đầu và vai, hãy ngồi/đứng vào giữa khung hình' };
  }
  const sw = Math.hypot(ls.x - rs.x, ls.y - rs.y);   // bề ngang vai, dùng làm thước đo
  if (sw < video.videoWidth * 0.1) {
    return { id: 'none', why: 'Bạn đứng hơi xa, hãy lại gần camera một chút' };
  }
  const midX = (ls.x + rs.x) / 2;
  const midY = (ls.y + rs.y) / 2;

  // Đo mọi thứ theo bề ngang vai nên người cao thấp, đứng gần xa đều dùng được
  if (nose.y > midY - 0.3 * sw) return { id: 'bow' };

  const lw = k.left_wrist, rw = k.right_wrist;
  if (lw && rw && lw.score >= WRIST_SCORE && rw.score >= WRIST_SCORE) {
    const up = (w) => w.y < nose.y - 0.6 * sw;
    const low = (w) => w.y > midY;
    if (up(lw) && up(rw)) return { id: 'up' };
    if ((up(lw) && low(rw)) || (up(rw) && low(lw))) return { id: 'one' };
    const wide = (w) => Math.abs(w.y - midY) < 0.6 * sw && Math.abs(w.x - midX) > 1.2 * sw;
    if (wide(lw) && wide(rw)) return { id: 'wide' };
  }
  return { id: 'idle' };
}

// Lấy đa số trong 5 khung hình gần nhất để kết quả không nhấp nháy
function smooth(id) {
  history.push(id);
  if (history.length > 5) history.shift();
  const counts = {};
  history.forEach((h) => (counts[h] = (counts[h] || 0) + 1));
  const [top, n] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  if (n >= 3) stable = top;
  return stable;
}

// ---------- Vẽ hình và bộ xương ----------
function drawFrame(pose) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (canvas.width !== vw) { canvas.width = vw; canvas.height = vh; }
  ctx.save();
  ctx.translate(vw, 0);
  ctx.scale(-1, 1);                    // lật gương cho dễ làm theo
  ctx.drawImage(video, 0, 0, vw, vh);
  ctx.restore();
  if (!pose) return;

  const showConf = confBox.checked;
  const kp = pose.keypoints;
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#00e5ff';
  EDGES.forEach(([a, b]) => {
    const p = kp[a], q = kp[b];
    if (p.score < MIN_SCORE || q.score < MIN_SCORE) return;
    ctx.beginPath();
    ctx.moveTo(vw - p.x, p.y);
    ctx.lineTo(vw - q.x, q.y);
    ctx.stroke();
  });
  kp.forEach((p) => {
    if (p.score < MIN_SCORE && !showConf) return;
    ctx.fillStyle = p.score >= 0.5 ? '#4caf50' : p.score >= MIN_SCORE ? '#ffb300' : '#e53935';
    ctx.beginPath();
    ctx.arc(vw - p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
    if (showConf) {
      ctx.fillStyle = 'white';
      ctx.font = '11px sans-serif';
      ctx.fillText(Math.round(p.score * 100) + '%', vw - p.x + 7, p.y - 6);
    }
  });
}

// ---------- Trò chơi "Simon nói" ----------
const game = { state: 'idle', round: 0, score: 0, target: null, last: null,
               deadline: 0, holdStart: 0, phase: 'play', until: 0 };

function resetGameUI() {
  game.state = 'idle';
  commandEl.textContent = 'Bấm "Bắt đầu chơi" nhé!';
  commandEl.className = 'command';
  scoreEl.textContent = '0';
  roundEl.textContent = '0';
  timeEl.textContent = '-';
  holdBar.style.width = '0%';
  startBtn.textContent = '▶️ Bắt đầu chơi';
}

function startGame() {
  game.state = 'playing';
  game.round = 0;
  game.score = 0;
  game.last = null;
  scoreEl.textContent = '0';
  startBtn.textContent = '🔁 Chơi lại';
  nextRound(performance.now());
}

function nextRound(now) {
  if (game.round >= ROUNDS) return endGame();
  game.round++;
  const choices = POSES.filter((p) => p.id !== game.last);
  game.target = choices[Math.floor(Math.random() * choices.length)];
  game.last = game.target.id;
  game.deadline = now + ROUND_MS;
  game.holdStart = 0;
  game.phase = 'play';
  roundEl.textContent = game.round;
  commandEl.className = 'command';
  commandEl.textContent = `Simon nói: ${game.target.emoji} ${game.target.name}!`;
  holdBar.style.width = '0%';
}

function endGame() {
  game.state = 'done';
  const s = game.score;
  const praise = s >= 100 ? '🏆 Vũ công siêu sao!' : s >= 60 ? '👏 Rất tốt!' : '💪 Cố gắng lên nhé!';
  commandEl.className = 'command good';
  commandEl.textContent = `🏁 Xong rồi! ${s} điểm. ${praise}`;
  timeEl.textContent = '-';
  holdBar.style.width = '0%';
}

function tickGame(now, current) {
  if (game.state !== 'playing') return;

  if (game.phase === 'feedback') {           // đang hiện "Đúng rồi" hoặc "Hết giờ"
    if (now >= game.until) nextRound(now);
    return;
  }

  const left = game.deadline - now;
  timeEl.textContent = Math.max(0, Math.ceil(left / 1000)) + 's';

  if (current === game.target.id) {
    if (!game.holdStart) game.holdStart = now;
  } else {
    game.holdStart = 0;
  }
  const held = game.holdStart ? now - game.holdStart : 0;
  holdBar.style.width = Math.min(100, (held / HOLD_MS) * 100) + '%';

  if (held >= HOLD_MS) {
    game.score += 10 + Math.ceil(left / 1000);   // làm nhanh được thưởng thêm
    scoreEl.textContent = game.score;
    commandEl.className = 'command good';
    commandEl.textContent = '✅ Đúng rồi! Giỏi quá!';
    game.phase = 'feedback';
    game.until = now + 1000;
  } else if (left <= 0) {
    commandEl.className = 'command bad';
    commandEl.textContent = '⏰ Hết giờ rồi, thử lượt sau nhé!';
    game.phase = 'feedback';
    game.until = now + 1200;
  }
}

startBtn.addEventListener('click', () => {
  if (active && detector) startGame();
});

// ---------- Xử lý mỗi khung hình ----------
function onPose(pose) {
  drawFrame(pose);
  const res = classify(pose);
  if (res.why) lastWhy = res.why;
  const current = smooth(res.id);

  poseBox.textContent = current === 'none'
    ? '❓ ' + lastWhy
    : `AI thấy: ${info(current).emoji} ${info(current).name}`;

  tickGame(performance.now(), current);
}

async function loop(id) {
  if (!active || id !== runId) return;
  if (video.readyState >= 2 && video.videoWidth > 0) {
    try {
      const poses = await detector.estimatePoses(video, { flipHorizontal: false });
      if (active && id === runId) onPose(poses[0]);
    } catch (e) {
      console.warn(e);
    }
  }
  setTimeout(() => loop(id), 40);
}

// ---------- Vào / rời màn chơi ----------
function cleanup() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  if (detector) detector.dispose();
  detector = null;
}

export async function enter() {
  const id = ++runId;
  active = true;
  history = [];
  stable = 'none';
  lastWhy = '';
  resetGameUI();
  setHint('⏳ Đang tải mô hình nhận dạng tư thế và mở camera...');
  try {
    await tf.ready();
    const d = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      modelUrl: MODEL_URL,
      enableSmoothing: true,
    });
    if (id !== runId) { d.dispose(); return; }   // đã rời màn trong lúc chờ
    detector = d;

    const s = await getStream('video');
    if (id !== runId) { s.getTracks().forEach((t) => t.stop()); return; }
    stream = s;
    video.srcObject = stream;
    await video.play();
    setHint('Hãy đứng cách camera khoảng 1,5-2 m, sao cho thấy rõ đầu, vai và hai tay. Rồi bấm "Bắt đầu chơi"!');
    loop(id);
  } catch (e) {
    console.error(e);
    setHint('❌ Lỗi: ' + e.message + ' (kiểm tra camera trong ⚙️ Cài đặt)');
  }
}

export function leave() {
  active = false;
  runId++;
  cleanup();
}