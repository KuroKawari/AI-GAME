import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const BASE = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/';
const OUT = 'public/models/mobilenet';

await mkdir(OUT, { recursive: true });

console.log('Đang tải model.json ...');
const res = await fetch(BASE + 'model.json');
if (!res.ok) throw new Error('Không tải được model.json: ' + res.status);
const json = await res.json();
await writeFile(path.join(OUT, 'model.json'), JSON.stringify(json));

for (const file of json.weightsManifest.flatMap((g) => g.paths)) {
  console.log('Đang tải', file, '...');
  const r = await fetch(BASE + file);
  if (!r.ok) throw new Error('Không tải được ' + file + ': ' + r.status);
  await writeFile(path.join(OUT, file), Buffer.from(await r.arrayBuffer()));
}
console.log('✅ Xong! Model nằm trong', OUT);