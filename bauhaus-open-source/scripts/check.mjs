import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const d = JSON.parse(fs.readFileSync(path.join(root, 'public/data.json')));
const products = new Map();
for (const p of d.catalog) {
  assert(!products.has(p.id), `重复商品 ID: ${p.id}`);
  products.set(p.id, p);
  assert(p.dims.length === 3 && p.dims.every(v => Number.isFinite(v) && v > 0), `尺寸无效: ${p.id}`);
  for (const k of ['name','brand','kind','categoryKey','room','price','url','review','modelNote']) assert(typeof p[k] === 'string', `${p.id}: 缺少 ${k}`);
  const referenceOnly = p.url.startsWith('products/') && p.name.includes('同款待核') && fs.existsSync(path.join(root,'public',p.url));
  assert(/^https?:\/\//.test(p.url) || referenceOnly, `商品链接格式无效: ${p.id}`);
}
for (const p of [...d.catalog, ...d.finishes]) {
  assert(p.photo && !path.isAbsolute(p.photo) && !p.photo.includes('..'), `图片路径无效: ${p.id}`);
  assert(fs.existsSync(path.join(root, 'public', p.photo)), `缺少图片: ${p.id}: ${p.photo}`);
}
const instances = new Set();
for (const o of d.layout.placed) {
  assert(products.has(o.pid), `未知商品引用: ${o.id}`);
  assert(!instances.has(o.id), `重复实例: ${o.id}`); instances.add(o.id);
  assert([o.x,o.y,o.z,o.rotation].every(Number.isFinite), `非有限坐标: ${o.id}`);
}
const readme = fs.readFileSync(path.join(root,'README.md'), 'utf8');
const images = [...readme.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)].map(m=>m[1]);
assert(images.length >= 5, 'README 应包含至少 5 张图片');
for (const image of images) assert(fs.existsSync(path.join(root,image)), `README 图片丢失: ${image}`);
for (const skill of ['room-planning','product-research']) {
  const text = fs.readFileSync(path.join(root,'skills',skill,'SKILL.md'),'utf8');
  assert(text.startsWith('---\nname: '+skill+'\n'), `skill 名称无效: ${skill}`);
}
console.log(JSON.stringify({ catalog: products.size, instances: instances.size, finishes: d.finishes.length, readmeImages: images.length, referencedPhotos: 'all-present', status: 'PASS' },null,2));
