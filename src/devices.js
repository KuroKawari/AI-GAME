const KEY = 'aigame.devices';
const settings = (() => {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
})();
const save = () => localStorage.setItem(KEY, JSON.stringify(settings));

export const getSelected = () => ({ camId: settings.camId, micId: settings.micId });
export function setSelected(kind, id) {
  settings[kind === 'video' ? 'camId' : 'micId'] = id;
  save();
}

// Xin quyền từng loại riêng để thiếu cam hoặc thiếu mic vẫn không hỏng cái còn lại
async function askPermission(constraints) {
  try {
    const s = await navigator.mediaDevices.getUserMedia(constraints);
    s.getTracks().forEach((t) => t.stop());
  } catch { /* bỏ qua, sẽ hiện thiết bị không tên */ }
}

export async function listDevices() {
  let devices = await navigator.mediaDevices.enumerateDevices();
  if (devices.some((d) => (d.kind === 'videoinput' || d.kind === 'audioinput') && !d.label)) {
    if (devices.some((d) => d.kind === 'videoinput')) await askPermission({ video: true });
    if (devices.some((d) => d.kind === 'audioinput')) await askPermission({ audio: true });
    devices = await navigator.mediaDevices.enumerateDevices();
  }
  return {
    cams: devices.filter((d) => d.kind === 'videoinput'),
    mics: devices.filter((d) => d.kind === 'audioinput'),
  };
}

// Các màn chơi gọi hàm này: getStream('video') hoặc getStream('audio')
export async function getStream(kind) {
  const id = kind === 'video' ? settings.camId : settings.micId;
  const base = kind === 'video'
  ? { width: 320, height: 240 }
  : { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  try {
    return await navigator.mediaDevices.getUserMedia({
      [kind]: id ? { ...base, deviceId: { exact: id } } : base,
    });
  } catch (e) {
    // Thiết bị đã lưu không còn nữa -> dùng thiết bị mặc định
    if (id && (e.name === 'OverconstrainedError' || e.name === 'NotFoundError')) {
      return navigator.mediaDevices.getUserMedia({ [kind]: base });
    }
    throw e;
  }
}