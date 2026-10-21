// Renders pitch.html deterministically, frame by frame.
//   node render.mjs stills 1.5 3 12.4      -> stills/t-1.50.png ...
//   node render.mjs video [fps] [from] [to] -> rogue-pitch.mp4 (silent master)
import { chromium } from 'playwright-core';
import { spawn } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const TYPES = { '.html': 'text/html', '.css': 'text/css', '.woff2': 'font/woff2', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml', '.webp': 'image/webp' };
const server = http.createServer((req, res) => {
  const file = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/pitch.html`;

const [mode = 'stills', ...rest] = process.argv.slice(2);
let browser;
try { browser = await chromium.launch(); }
catch { browser = await chromium.launch({ channel: 'chrome' }); } // falls back to the installed Google Chrome
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on('pageerror', e => console.error('PAGE ERROR', e.message));
page.on('console', m => { if (m.type() === 'error') console.error('CONSOLE', m.text()); });
await page.goto(base);
await page.evaluate(() => window.ready);
const cdp = await page.context().newCDPSession(page);
const shot = async () => Buffer.from((await cdp.send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true })).data, 'base64');

if (mode === 'stills') {
  fs.mkdirSync(path.join(ROOT, 'stills'), { recursive: true });
  for (const s of rest) {
    const t = parseFloat(s);
    await page.evaluate(t => window.render(t), t);
    fs.writeFileSync(path.join(ROOT, 'stills', `t-${t.toFixed(2)}.png`), await shot());
  }
  console.log('stills done');
} else {
  const fps = parseInt(rest[0] || '60', 10);
  const dur = await page.evaluate(() => window.DURATION);
  const from = parseFloat(rest[1] || '0'), to = parseFloat(rest[2] || String(dur));
  const out = path.join(ROOT, rest[3] || 'rogue-pitch-silent.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const n0 = Math.round(from * fps), n1 = Math.round(to * fps);
  const t0 = Date.now();
  for (let f = n0; f < n1; f++) {
    await page.evaluate(t => window.render(t), f / fps);
    const buf = await shot();
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if ((f - n0) % (fps * 5) === 0) console.log(`frame ${f - n0}/${n1 - n0}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  console.log('video done', out, ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
await browser.close();
server.close();
