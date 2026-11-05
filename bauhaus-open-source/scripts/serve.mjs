import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const directory = path.resolve(root, process.argv[2] || 'dist');
const port = Number(process.argv[3] || process.env.PORT || 18769);
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };
if (!fs.existsSync(path.join(directory, 'index.html')) && !fs.existsSync(path.join(directory, 'standalone.html'))) throw new Error('缺少构建文件，请先运行 npm run build。');
const server = http.createServer((req, res) => {
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405); res.end(); return; }
  let relative;
  try { relative = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); }
  catch { res.writeHead(400); res.end(); return; }
  const file = path.resolve(directory, '.' + (relative.endsWith('/') ? relative + 'index.html' : relative));
  if (file !== directory && !file.startsWith(directory + path.sep)) { res.writeHead(403); res.end(); return; }
  fs.stat(file, (err, stat) => {
    if (err || !stat.isFile()) { res.writeHead(404); res.end('Not found'); return; }
    res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream', 'Content-Length': stat.size, 'Cache-Control': 'no-cache' });
    if (req.method === 'HEAD') res.end();
    else { const stream = fs.createReadStream(file); stream.on('error', () => res.destroy()); stream.pipe(res); }
  });
});
server.on('error', e => { console.error(`无法启动端口 ${port}：${e.message}`); process.exitCode = 1; });
server.listen(port, '127.0.0.1', () => console.log(`包豪斯样板间：http://127.0.0.1:${port}/\n目录：${directory}\n按 Ctrl+C 停止。`));
