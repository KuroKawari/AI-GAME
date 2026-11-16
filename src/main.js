import { listDevices, getStream, getSelected, setSelected } from './devices.js';
import { enter as enterLevel1, leave as leaveLevel1 } from './level1.js';
import { enter as enterLevel2, leave as leaveLevel2 } from './level2.js';
import { enter as enterLevel3, leave as leaveLevel3 } from './level3.js';

// ---- Chuyển màn hình ----
const levels = {
  level1: { enter: enterLevel1, leave: leaveLevel1 },
  level2: { enter: enterLevel2, leave: leaveLevel2 },
  level3: { enter: enterLevel3, leave: leaveLevel3 }
};
let currentScreen = 'menu';
function showScreen(id) {
  if (levels[currentScreen] && currentScreen !== id) levels[currentScreen].leave();
  document.querySelectorAll('.screen').forEach((s) => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  currentScreen = id;
  if (levels[id]) levels[id].enter();
}
document.querySelectorAll('[data-screen]').forEach((btn) => {
  btn.addEventListener('click', () => showScreen(btn.dataset.screen));
});

// ---- Popup Settings ----
const modal = document.getElementById('settingsModal');
const video = document.getElementById('video');
const camSelect = document.getElementById('camSelect');
const micSelect = document.getElementById('micSelect');
const meterBar = document.getElementById('meterBar');
const status = document.getElementById('status');

let camStream = null;
let micStream = null;
let audioCtx = null;
let meterRunning = false;

function fillSelect(select, devices, savedId, fallbackName) {
  select.innerHTML = '';
  devices.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `${fallbackName} ${i + 1}`;
    select.appendChild(opt);
  });
  if (devices.some((d) => d.deviceId === savedId)) select.value = savedId;
}

function stopAll() {
  meterRunning = false;
  if (camStream) camStream.getTracks().forEach((t) => t.stop());
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  camStream = micStream = audioCtx = null;
  video.srcObject = null;
  meterBar.style.width = '0%';
}

async function startCamPreview() {
  if (camStream) camStream.getTracks().forEach((t) => t.stop());
  try {
    setSelected('video', camSelect.value);
    camStream = await getStream('video');
    video.srcObject = camStream;
    await video.play();
  } catch (e) {
    status.textContent = '❌ Camera: ' + e.message;
  }
}

async function startMicPreview() {
  meterRunning = false;
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  try {
    setSelected('audio', micSelect.value);
    micStream = await getStream('audio');
    audioCtx = new AudioContext();
    await audioCtx.resume();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    audioCtx.createMediaStreamSource(micStream).connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    meterRunning = true;
    (function loop() {
      if (!meterRunning) return;
      analyser.getByteFrequencyData(data);
      const avg = data.reduce((a, b) => a + b, 0) / data.length;
      meterBar.style.width = Math.min(100, avg * 2) + '%';
      requestAnimationFrame(loop);
    })();
  } catch (e) {
    status.textContent = '❌ Micro: ' + e.message;
  }
}

async function openSettings() {
  modal.classList.add('open');
  status.textContent = '⏳ Đang tìm thiết bị...';
  const { cams, mics } = await listDevices();
  const saved = getSelected();
  fillSelect(camSelect, cams, saved.camId, 'Camera');
  fillSelect(micSelect, mics, saved.micId, 'Micro');
  status.textContent = cams.length || mics.length
    ? 'Chọn thiết bị, hình và thanh âm lượng phải hoạt động. Hãy vỗ tay thử!'
    : '⚠️ Không tìm thấy camera hoặc micro nào.';
  if (cams.length) startCamPreview();
  if (mics.length) startMicPreview();
}

function closeSettings() {
  stopAll();
  modal.classList.remove('open');
}

document.getElementById('btnSettings').addEventListener('click', openSettings);
document.getElementById('btnCloseSettings').addEventListener('click', closeSettings);
camSelect.addEventListener('change', startCamPreview);
micSelect.addEventListener('change', startMicPreview);