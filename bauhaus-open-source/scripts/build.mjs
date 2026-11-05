import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const publicMode = process.argv.includes('--public');
const destination = path.join(root, publicMode ? 'release-public' : 'dist');
const read = name => fs.readFile(path.join(root, name), 'utf8');
const safeJSON = value => JSON.stringify(value).replace(/</g, '\\u003c').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
await fs.mkdir(destination, { recursive: true });
for (const name of ['index.html', 'style.css', 'model-review.html', 'missing.svg']) {
  await fs.copyFile(path.join(root, 'public', name), path.join(destination, name));
}
await fs.cp(path.join(root, 'public/assets'), path.join(destination, 'assets'), { recursive: true });
const data = JSON.parse(await read('public/data.json'));
if (publicMode) {
  // This edition does not redistribute merchant photographs. Procurement links,
  // records, geometry and the CC0 material maps remain available.
  for (const p of [...data.catalog, ...data.finishes]) {
    p.photo = 'missing.svg';
    if (p.url && !/^https?:\/\//.test(p.url)) p.url = 'missing.svg';
    p.photoStatus = '公开构建未包含第三方商品照片；请查看原商品链接。';
  }
} else {
  await fs.cp(path.join(root, 'public/products'), path.join(destination, 'products'), { recursive: true });
}
await fs.writeFile(path.join(destination, 'data.json'), JSON.stringify(data, null, 2));
const options = { bundle: true, minify: true, format: 'iife', target: 'es2022', legalComments: 'inline', write: false, absWorkingDir: root };
const app = (await build({ ...options, entryPoints: ['src/app.js'] })).outputFiles[0].text;
// The review page has top-level await. Wrap its body, but retain its imports.
const reviewSource = await read('src/model-review.js');
const imports = reviewSource.split('\n').filter(line => line.startsWith('import ')).join('\n');
const body = reviewSource.split('\n').filter(line => !line.startsWith('import ')).join('\n');
const review = (await build({ ...options, stdin: { contents: `${imports}\n(async()=>{${body}\n})().catch(console.error);`, resolveDir: path.join(root, 'src'), loader: 'js' } })).outputFiles[0].text;
await fs.writeFile(path.join(destination, 'app.js'), app);
await fs.writeFile(path.join(destination, 'model-review.js'), review);

const assets = {};
const mime = { '.jpg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.svg': 'image/svg+xml', '.avif': 'image/avif' };
for (const folder of publicMode ? ['assets'] : ['assets', 'products']) {
  for (const name of await fs.readdir(path.join(destination, folder))) {
    const type = mime[path.extname(name)];
    if (type) assets[`${folder}/${name}`] = `data:${type};base64,${(await fs.readFile(path.join(destination, folder, name))).toString('base64')}`;
  }
}
assets['missing.svg'] = `data:image/svg+xml;base64,${Buffer.from(await read('public/missing.svg')).toString('base64')}`;
const bootstrap = `globalThis.__BAUHAUS_DATA__=${safeJSON(data)};globalThis.__BAUHAUS_ASSETS__=${safeJSON(assets)};`;
let reviewTemplate = (await read('public/model-review.html')).replace('<script src="model-review.js" type="module"></script>', () => `<script>${review.replace(/<\/script/gi, '<\\/script')}</script>`);
reviewTemplate = reviewTemplate.replace('href="./?v=5"', 'href="#" onclick="window.parent.document.getElementById(\'embedded-review\').close();return false"');
// Create the review document only when clicked, reusing the embedded assets.
const hook = `document.querySelector('a[href="model-review.html"]').addEventListener('click',e=>{e.preventDefault();let dialog=document.getElementById('embedded-review');if(!dialog){dialog=document.createElement('dialog');dialog.id='embedded-review';dialog.style.cssText='padding:0;width:94vw;max-width:1600px;height:92vh;max-height:96vh;border:1px solid #aaa;background:#eeebe2';const close=document.createElement('button');close.textContent='关闭商品对照';close.style.cssText='position:absolute;right:12px;top:12px;z-index:2';close.onclick=()=>dialog.close();const frame=document.createElement('iframe');frame.title='内嵌商品图与模型对照';frame.style.cssText='width:100%;height:100%;border:0';const boot='globalThis.__BAUHAUS_DATA__='+JSON.stringify(globalThis.__BAUHAUS_DATA__).replace(/</g,'\\\\u003c')+';globalThis.__BAUHAUS_ASSETS__='+JSON.stringify(globalThis.__BAUHAUS_ASSETS__).replace(/</g,'\\\\u003c')+';';frame.srcdoc=${safeJSON(reviewTemplate)}.replace('<header>','<script>'+boot+'<\\/script><header>');dialog.append(close,frame);document.body.append(dialog);dialog.addEventListener('close',()=>dialog.remove());}dialog.showModal();});`;
const html = (await read('public/index.html'))
  .replace(/<link rel="stylesheet" href="[^"]+">/, `<style>${await read('public/style.css')}</style>`)
  .replace(/<script defer src="[^"]+"><\/script>/, () => `<script>${bootstrap}\n${app.replace(/<\/script/gi, '<\\/script')}\n${hook}</script>`);
const single = publicMode ? path.join(destination, 'standalone.html') : path.join(root, 'standalone.html');
await fs.writeFile(single, html);
await fs.copyFile(path.join(root, 'THIRD_PARTY_NOTICES.md'), path.join(destination, 'THIRD_PARTY_NOTICES.md'));
console.log(JSON.stringify({ edition: publicMode ? 'public-without-product-photos' : 'complete-local', output: destination, standalone: single, embeddedAssets: Object.keys(assets).length, bytes: Buffer.byteLength(html) }, null, 2));
