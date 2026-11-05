import { loadData, assetURL } from "./runtime.js";
import * as T from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { materials } from "./materials.js";
import { house } from "./house.js";
import { decorateWalls, roomNames, defaults, roomAt } from "./finishes.js";
import { product, box, cylinder } from "./objects.js";
import { issue, allowed, obstacles, corners } from "./collision.js";
const $ = (s) => document.querySelector(s), $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const KEY = "bauhaus-night-studio-v5" + (new URLSearchParams(location.search).has("check") ? "-qa-" + new URLSearchParams(location.search).get("check") : "");
let data, M, scene, renderer, camera, orbit, gizmo, composer, ao, architecture, objects, grid, items = [], editItems = [], selected = null, isSample = true, walk = true, tool = "drag", kind = "N007", selectionBox = null, ready = false;
let finishes = { ...defaults }, editFinishes = { ...defaults }, doorHover = null;
const catalog = {}, instances = /* @__PURE__ */ new Map(), undo = [], redo = [], keys = /* @__PURE__ */ new Set();
let drag = null, looking = null, beforeTransform = null, toastTimer, last = 0, fpsTime = 0, frames = 0;
const views = { entry: [[1.65, 1.48, 0.65], [0.37, 0.7, 0.61]], front: [[2.8, 1.6, -1.25], [4.7, 1, 0.5]], dining: [[2.73, 1.56, 4.7], [0.5, 1.12, 2.65]], living: [[3.87, 1.55, 5.57], [5.02, 0.86, 7.02]], study: [[4.3, 1.55, 4.22], [4.4, 0.82, 2.5]], bedroom: [[2.92, 1.55, 5.68], [1.85, 0.72, 7.08]], bath: [[5.85, 1.58, 6.32], [6.95, 1.12, 7.45]], garden: [[3.8, 1.6, 8.78], [1.7, 0.9, 10.65]] };
function toast(t) {
  $("#toast").textContent = t;
  $("#toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").classList.remove("show"), 3200);
}
function snapshot() {
  return structuredClone({ items, finishes });
}
function restore(state) {
  items = structuredClone(state.items || state);
  finishes = { ...defaults, ...state.finishes };
  applyFinishes();
  buildItems();
}
function applyFinishes() {
  if (!architecture) return;
  for (const [key, color] of Object.entries(finishes)) architecture.userData.roomMaterials[key]?.color.set(color);
  renderer.shadowMap.needsUpdate = true;
}
function store() {
  if (isSample) return;
  editItems = structuredClone(items);
  editFinishes = { ...finishes };
  try {
    localStorage.setItem(KEY, JSON.stringify(snapshot()));
    $("#save-status").textContent = "\u5DF2\u4FDD\u5B58\u5230\u672C\u673A \xB7 " + items.length + " \u4EF6\u5BB6\u5177";
  } catch {
    $("#save-status").textContent = "\u672C\u673A\u5B58\u50A8\u4E0D\u8DB3\uFF0C\u8BF7\u5BFC\u51FA\u65B9\u6848";
  }
}
function history(before) {
  if (JSON.stringify(before) === JSON.stringify(snapshot())) return;
  undo.push(before);
  if (undo.length > 50) undo.shift();
  redo.length = 0;
  $("#undo").disabled = !undo.length;
  $("#redo").disabled = true;
  store();
}
function valid(arr) {
  return Array.isArray(arr) && arr.length <= 500 && arr.every((o) => catalog[o.pid] && typeof o.id === "string" && [o.x, o.y, o.z, o.rotation].every(Number.isFinite) && o.x > -1 && o.x < 9 && o.z > -2 && o.z < 13 && o.y >= 0 && o.y < 2.65) && new Set(arr.map((o) => o.id)).size === arr.length;
}
function problem(item) {
  const p = catalog[item.pid];
  if (!p) return "\u672A\u77E5\u7269\u54C1";
  if (p.mount === "wall" || p.mount === "ceiling") return item.y + p.dims[2] > 2.75 ? "\u8D85\u51FA\u5BA4\u5185\u9AD8\u5EA6" : null;
  if (p.mount === "surface") return null;
  const visible = items.filter((o) => !["wall", "ceiling", "surface"].includes(catalog[o.pid].mount) && catalog[o.pid].kind !== "roundrug");
  if (p.kind === "roundrug") return issue(item, [], catalog, data.layout, obstacles("B", data.layout));
  return issue(item, visible, catalog, data.layout, obstacles("B", data.layout));
}
function sync(o) {
  const g = instances.get(o.id);
  if (!g) return;
  g.position.set(o.x, o.y, o.z);
  g.rotation.y = T.MathUtils.degToRad(o.rotation);
  g.updateMatrixWorld(true);
  renderer.shadowMap.needsUpdate = true;
}
function clearObjects() {
  gizmo.detach();
  if (selectionBox) {
    scene.remove(selectionBox);
    selectionBox.geometry.dispose();
    selectionBox.material.dispose();
    selectionBox = null;
  }
  objects.traverse((o) => {
    if (o.isMesh) o.geometry.dispose();
  });
  objects.clear();
  instances.clear();
}
function buildItems() {
  clearObjects();
  for (const o of items) {
    const g = product(catalog[o.pid], M);
    g.userData.instanceId = o.id;
    objects.add(g);
    instances.set(o.id, g);
    sync(o);
  }
  select(null);
  updateCount();
}
function updateCount() {
  const problems = items.filter((o) => problem(o));
  $("#quality").textContent = problems.length ? problems.length + " \u5904\u4F4D\u7F6E\u9700\u68C0\u67E5" : "\u591C\u95F4\u6696\u5149 / 21:00";
  renderer.domElement.dataset.items = items.length;
  renderer.domElement.dataset.collisions = problems.length;
}
function panel() {
  const o = items.find((x) => x.id === selected), p = o && catalog[o.pid];
  $("#empty").hidden = !!o;
  $("#selected").hidden = !o;
  $("#workspace").classList.toggle("has-selection", !!o && !isSample);
  if (!o) return;
  $("#object-name").textContent = p.name;
  $("#object-photo").src = p.photo || assetURL("missing.svg");
  $("#object-brand").textContent = p.brandIntro || p.brand;
  $("#object-price").textContent = priceText(p);
  $("#object-model").textContent = p.modelNote;
  $("#object-size").textContent = "\u5BBD \xD7 \u6DF1 \xD7 \u9AD8\uFF1A" + p.dims.map((x) => Math.round(x * 1e3)).join(" \xD7 ") + " mm" + (p.estimated ? " \xB7 \u542B\u6682\u5B9A\u5C3A\u5BF8" : "");
  for (const [id, key] of [["px", "x"], ["py", "y"], ["pz", "z"], ["angle", "rotation"]]) $("#" + id).value = Number(o[key].toFixed(3));
  $("#object-warning").textContent = problem(o) || "\u4F4D\u7F6E\u53EF\u7528";
  $("#source").href = p.url;
  $("#source-note").textContent = p.sizeText || "\u6682\u65E0\u8865\u5145\u5C3A\u5BF8\u8BF4\u660E";
  $("#review").textContent = p.review || "\u6682\u65E0\u5DF2\u6838\u9A8C\u8BC4\u8BBA";
  renderer.domElement.dataset.selected = selected || "";
}
function select(id) {
  selected = isSample ? null : id;
  gizmo.detach();
  if (selectionBox) {
    scene.remove(selectionBox);
    selectionBox.geometry.dispose();
    selectionBox.material.dispose();
    selectionBox = null;
  }
  const g = instances.get(selected);
  if (g) {
    selectionBox = new T.BoxHelper(g, "#d9ae56");
    selectionBox.material.depthTest = false;
    selectionBox.renderOrder = 10;
    scene.add(selectionBox);
    setTool(tool);
  }
  panel();
}
function setTool(t) {
  tool = t;
  $$("[data-tool]").forEach((b) => b.classList.toggle("active", b.dataset.tool === t));
  if (!selected || isSample) return;
  if (t === "drag") gizmo.detach();
  else {
    gizmo.attach(instances.get(selected));
    gizmo.setMode(t === "rotate" ? "rotate" : "translate");
    gizmo.showX = t === "translate";
    gizmo.showZ = t === "translate";
    gizmo.showY = t === "rotate" || t === "height" || t === "translate";
    gizmo.setSpace("world");
  }
}
function change(fn) {
  if (isSample || !selected) return;
  const before = snapshot(), o = items.find((x) => x.id === selected);
  fn(o);
  o.y = T.MathUtils.clamp(o.y, 0, 2.6);
  sync(o);
  panel();
  updateCount();
  history(before);
}
function setMode(sample) {
  if (sample === isSample) return;
  if (!isSample) {
    store();
    editItems = structuredClone(items);
    editFinishes = { ...finishes };
  }
  isSample = sample;
  items = structuredClone(sample ? data.layout.placed : editItems);
  finishes = { ...sample ? defaults : editFinishes };
  applyFinishes();
  buildItems();
  $("#workspace").classList.toggle("sample", sample);
  $("#sample").classList.toggle("active", sample);
  $("#edit").classList.toggle("active", !sample);
  $("#save-status").textContent = sample ? "\u5F53\u524D\u4E3A\u6837\u677F\u95F4\u6D4F\u89C8" : "\u4F60\u7684\u5E03\u7F6E\u5DF2\u6062\u590D";
  $("#hint").textContent = sample ? "\u62D6\u52A8\u8F6C\u5934 \xB7 WASD \u81EA\u7531\u884C\u8D70 \xB7 \u6EDA\u8F6E\u524D\u540E\u79FB\u52A8" : "\u5DE6\u952E\u9009\u53D6 / \u62D6\u653E \xB7 \u53F3\u952E\u8F6C\u5934 \xB7 WASD \u884C\u8D70";
  resize();
  inventory();
}
function visit(name) {
  const v = views[name];
  walk = true;
  orbit.enabled = false;
  architecture.userData.ceiling.visible = true;
  camera.up.set(0, 1, 0);
  camera.fov = name === "bath" ? 72 : 66;
  camera.position.fromArray(v[0]);
  camera.lookAt(new T.Vector3(...v[1]));
  camera.updateProjectionMatrix();
  $$("[data-room]").forEach((b) => b.classList.toggle("active", b.dataset.room === name));
  $("#viewport").dataset.room = name;
  $("#hint").textContent = isSample ? "\u62D6\u52A8\u8F6C\u5934 \xB7 WASD \u81EA\u7531\u884C\u8D70 \xB7 \u6EDA\u8F6E\u524D\u540E\u79FB\u52A8" : "\u5DE6\u952E\u9009\u53D6 / \u62D6\u653E \xB7 \u53F3\u952E\u8F6C\u5934 \xB7 WASD \u884C\u8D70";
  renderer.shadowMap.needsUpdate = true;
}
function planView(top) {
  walk = false;
  orbit.enabled = true;
  architecture.userData.ceiling.visible = false;
  camera.fov = 45;
  camera.up.set(0, top ? 0 : 1, top ? -1 : 0);
  camera.position.set(top ? 3.7 : 13, top ? 20 : 13, top ? 5.1 : 16);
  orbit.target.set(3.7, 0, 5.1);
  orbit.update();
  camera.updateProjectionMatrix();
  renderer.shadowMap.needsUpdate = true;
  $("#hint").textContent = "\u62D6\u52A8\u7A7A\u767D\u5904\u73AF\u7ED5 \xB7 \u6EDA\u8F6E\u7F29\u653E \xB7 \u70B9\u51FB\u5BB6\u5177\u7F16\u8F91";
}
function priceText(p) {
  return (p.price || "\u6682\u65E0\u5DF2\u6838\u9A8C\u4EF7\u683C").replace(/累计评价/g, "").replace(/\s+([；。])/g, "$1");
}
function productInfo(p) {
  return `<dl class="product-info" aria-label="\u5546\u54C1\u4FE1\u606F\u56DB\u680F"><div><dt>\u4EF7\u683C</dt><dd>${esc(priceText(p))}</dd></div><div><dt>\u8DF3\u8F6C\u94FE\u63A5</dt><dd>${p.url ? `<a href="${esc(p.url)}" target="_blank" rel="noopener">\u67E5\u770B\u539F\u5546\u54C1 \u2197</a>` : "\u6682\u65E0\u5546\u54C1\u94FE\u63A5"}</dd></div><div><dt>\u4EA7\u54C1\u4ECB\u7ECD</dt><dd>${esc(p.brandIntro || p.brand || "")}<p>${esc(p.modelNote || p.note || "")}</p></dd></div><div><dt>\u8BC4\u8BBA\u60C5\u51B5</dt><dd>${esc(p.review || "\u6682\u65E0\u5DF2\u6838\u9A8C\u8BC4\u8BBA")}</dd></div></dl>`;
}
function inventory() {
  if (!data) return;
  const query = $("#search").value.toLowerCase(), room = $("#room-filter").value;
  const matches = data.catalog.filter((p) => (!room || p.room === room) && (!query || (p.name + p.category + p.brand + p.seller).toLowerCase().includes(query)));
  const cats = [...new Map(matches.map((p) => [p.categoryKey, p])).values()];
  if (!cats.some((p) => p.categoryKey === kind) && cats.length) kind = cats[0].categoryKey;
  $("#categories").innerHTML = cats.map((p) => `<button data-category="${p.categoryKey}" class="${p.categoryKey === kind ? "active" : ""}">${esc(p.category)}</button>`).join("");
  $$("[data-category]").forEach((b) => b.onclick = () => {
    kind = b.dataset.category;
    inventory();
  });
  const ps = matches.filter((p) => p.categoryKey === kind);
  $("#count").textContent = data.catalog.length + "\u4EF6";
  $("#category-name").textContent = ps[0]?.category || "\u6CA1\u6709\u5339\u914D\u5546\u54C1";
  $("#candidate-count").textContent = ps.length + " \u4E2A\u5019\u9009";
  $("#candidates").innerHTML = ps.map((p) => `<article class="candidate"><figure draggable="true" data-pid="${p.id}"><img loading="lazy" src="${p.photo || assetURL("missing.svg")}" alt="${esc(p.name)}" draggable="false"><span class="number">${p.sourceKey}</span></figure><small class="brand">${esc(p.brand)}</small><h3>${esc(p.name)}</h3><p>\u5BBD \xD7 \u6DF1 \xD7 \u9AD8\uFF1A${p.dims.map((x) => Math.round(x * 1e3)).join(" \xD7 ")} mm${p.estimated ? " \xB7 \u542B\u6682\u5B9A\u5C3A\u5BF8" : ""}</p>${productInfo(p)}<details><summary>\u5C3A\u5BF8\u8BF4\u660E</summary><p>${esc(p.sizeText)}</p></details><div class="actions">${items.some((o) => o.pid === p.id) ? `<button data-locate="${p.id}">\u67E5\u770B\u573A\u666F\u6A21\u578B</button>` : ""}<button class="add" data-add="${p.id}">\u653E\u5165\u623F\u95F4</button><button data-replace="${p.id}" ${!selected ? "disabled" : ""}>\u66FF\u6362\u9009\u4E2D</button></div></article>`).join("");
  $$("[data-locate]").forEach((b) => b.onclick = () => {
    const o = items.find((x) => x.pid === b.dataset.locate);
    if (o) {
      select(o.id);
      const a = o.rotation * Math.PI / 180;
      let pt = null;
      for (let radius of [1.15, 0.85, 1.5, 1.8]) {
        for (let offset of [0, 0.6, -0.6, 1.2, -1.2, 2, -2, Math.PI]) {
          let x = o.x + Math.sin(a + offset) * radius, z = o.z + Math.cos(a + offset) * radius;
          if (!blocked(x, z)) {
            pt = [x, z];
            break;
          }
        }
        if (pt) break;
      }
      if (pt) camera.position.set(pt[0], Math.min(1.65, Math.max(0.8, o.y + catalog[o.pid].dims[2] + 0.4)), pt[1]);
      camera.lookAt(o.x, o.y + catalog[o.pid].dims[2] / 2, o.z);
      walk = true;
      orbit.enabled = false;
    }
  });
  $$("[data-pid]").forEach((f) => f.ondragstart = (e) => e.dataTransfer.setData("text/plain", f.dataset.pid));
  $$("[data-add]").forEach((b) => b.onclick = () => add(b.dataset.add));
  $$("[data-replace]").forEach((b) => b.onclick = () => replace(b.dataset.replace));
}
function wallPosition(o) {
  let best = null;
  for (const [x1, z1, x2, z2, t] of data.layout.walls) {
    const dx = x2 - x1, dz = z2 - z1, l = Math.hypot(dx, dz), f = T.MathUtils.clamp(((o.x - x1) * dx + (o.z - z1) * dz) / (l * l), 0.06, 0.94), cx = x1 + f * dx, cz = z1 + f * dz;
    let nx = -dz / l, nz = dx / l;
    if ((camera.position.x - cx) * nx + (camera.position.z - cz) * nz < 0) {
      nx = -nx;
      nz = -nz;
    }
    const dist = Math.hypot(o.x - cx, o.z - cz);
    if (!best || dist < best.dist) best = { dist, x: cx + nx * (t / 2 + catalog[o.pid].dims[1] / 2 + 0.012), z: cz + nz * (t / 2 + catalog[o.pid].dims[1] / 2 + 0.012), rotation: Math.atan2(nx, nz) * 180 / Math.PI };
  }
  if (best) Object.assign(o, { x: best.x, z: best.z, rotation: best.rotation });
}
function newId() {
  return "B-" + crypto.randomUUID();
}
function add(pid, point) {
  if (isSample) setMode(false);
  const p = catalog[pid];
  if (!p) return;
  const before = snapshot();
  let pt = point;
  if (!pt) {
    const dir = new T.Vector3();
    camera.getWorldDirection(dir);
    dir.y = 0;
    pt = camera.position.clone().add(dir.normalize().multiplyScalar(1.3));
  }
  const o = { id: newId(), pid, x: pt.x, z: pt.z, y: p.mount === "wall" ? Math.max(0.3, 1.9 - p.dims[2]) : p.mount === "ceiling" ? 2.12 : p.mount === "surface" ? point?.y || 0.75 : 0, rotation: 0 };
  if (p.mount === "wall") wallPosition(o);
  if (!point && problem(o)) {
    let found = false;
    for (let radius = 0; radius < 2 && !found; radius += 0.2) for (let a = 0; a < Math.PI * 2; a += 0.4) {
      o.x = pt.x + Math.cos(a) * radius;
      o.z = pt.z + Math.sin(a) * radius;
      if (!problem(o)) {
        found = true;
        break;
      }
    }
  }
  items.push(o);
  const g = product(p, M);
  g.userData.instanceId = o.id;
  objects.add(g);
  instances.set(o.id, g);
  sync(o);
  select(o.id);
  history(before);
  updateCount();
  inventory();
  $("#workspace").classList.remove("library-open");
  toast(problem(o) || "\u5DF2\u653E\u5165 " + p.category);
}
function replace(pid) {
  if (!selected || isSample) return;
  const id = selected, before = snapshot(), o = items.find((o2) => o2.id === id);
  if (catalog[pid].categoryKey !== catalog[o.pid].categoryKey) {
    toast("\u8BF7\u9009\u62E9\u540C\u7C7B\u5BB6\u5177\u8FDB\u884C\u66FF\u6362");
    return;
  }
  o.pid = pid;
  buildItems();
  select(id);
  history(before);
  inventory();
  toast(problem(o) || "\u5DF2\u66FF\u6362\uFF0C\u4F4D\u7F6E\u4E0E\u89D2\u5EA6\u4FDD\u7559");
}
function remove() {
  if (!selected || isSample) return;
  const before = snapshot();
  items = items.filter((o) => o.id !== selected);
  buildItems();
  history(before);
  inventory();
}
function blocked(x, z) {
  if (!allowed(x, z, data.layout)) return true;
  for (const [x1, z1, x2, z2, t] of data.layout.walls) {
    const dx = x2 - x1, dz = z2 - z1, f = T.MathUtils.clamp(((x - x1) * dx + (z - z1) * dz) / (dx * dx + dz * dz), 0, 1);
    if (Math.hypot(x - x1 - dx * f, z - z1 - dz * f) < t / 2 + 0.13) return true;
  }
  for (const w of data.layout.windows) {
    const dx = w.x2 - w.x1, dz = w.z2 - w.z1, f = T.MathUtils.clamp(((x - w.x1) * dx + (z - w.z1) * dz) / (dx * dx + dz * dz), 0, 1);
    if (Math.hypot(x - w.x1 - dx * f, z - w.z1 - dz * f) < 0.2) return true;
  }
  for (const b of obstacles("B", data.layout).slice(data.layout.walls.length)) {
    if (Math.abs(x - b.o.x) < b.p.dims[0] / 2 + 0.1 && Math.abs(z - b.o.z) < b.p.dims[1] / 2 + 0.1) return true;
  }
  for (const door of architecture?.userData.doors || []) {
    const b = new T.Box3().setFromObject(door);
    if (x > b.min.x - 0.12 && x < b.max.x + 0.12 && z > b.min.z - 0.12 && z < b.max.z + 0.12) return true;
  }
  for (const o of items) {
    const p = catalog[o.pid];
    if (["wall", "ceiling", "surface"].includes(p.mount) || ["rug", "roundrug"].includes(p.kind) || o.y > 1.7) continue;
    const a = o.rotation * Math.PI / 180, dx = x - o.x, dz = z - o.z, lx = Math.cos(a) * dx - Math.sin(a) * dz, lz = Math.sin(a) * dx + Math.cos(a) * dz;
    if (Math.abs(lx) < p.dims[0] / 2 + 0.06 && Math.abs(lz) < p.dims[1] / 2 + 0.06) return true;
  }
  return false;
}
function move(v) {
  const n = Math.ceil(v.length() / 0.04) || 1;
  v.multiplyScalar(1 / n);
  v.y = 0;
  for (let i = 0; i < n; i++) {
    if (!blocked(camera.position.x + v.x, camera.position.z)) camera.position.x += v.x;
    if (!blocked(camera.position.x, camera.position.z + v.z)) camera.position.z += v.z;
  }
}
function direction(k) {
  const dir = new T.Vector3();
  camera.getWorldDirection(dir);
  dir.y = 0;
  dir.normalize();
  if (k === "back" || k === "s") dir.negate();
  if (["left", "right", "a", "d"].includes(k)) {
    dir.cross(new T.Vector3(0, 1, 0));
    if (k === "left" || k === "a") dir.negate();
  }
  return dir;
}
const ray = new T.Raycaster(), pointer = new T.Vector2();
function pointerRay(e) {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  ray.setFromCamera(pointer, camera);
}
function pick(e) {
  pointerRay(e);
  for (const h of ray.intersectObjects(objects.children, true)) {
    let o = h.object;
    while (o && !o.userData.instanceId) o = o.parent;
    if (o) return o.userData.instanceId;
  }
  return null;
}
function ground(e, y = 0) {
  pointerRay(e);
  const point = new T.Vector3();
  return ray.ray.intersectPlane(new T.Plane(new T.Vector3(0, 1, 0), -y), point);
}
function bindCanvas() {
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 && e.button !== 2) return;
    canvas.focus();
    if (e.button === 0) {
      pointerRay(e);
      const hit = ray.intersectObjects(architecture.userData.doors, true)[0];
      if (hit && hit.distance < 2.6) {
        let d = hit.object;
        while (d && !d.userData.door) d = d.parent;
        if (d) {
          d.userData.open = !d.userData.open;
          d.rotation.y = d.userData.open ? d.userData.openAngle : 0;
          renderer.shadowMap.needsUpdate = true;
          toast(d.userData.label + (d.userData.open ? "\u5DF2\u6253\u5F00" : "\u5DF2\u5173\u95ED"));
          return;
        }
      }
    }
    if (gizmo.dragging || gizmo.axis) return;
    if (walk && (isSample || e.button === 2)) {
      looking = { x: e.clientX, y: e.clientY, q: camera.quaternion.clone() };
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    const id = pick(e);
    select(id);
    inventory();
    if (id && tool === "drag") {
      const item = items.find((o) => o.id === id), pt = ground(e, item.y);
      if (pt) {
        drag = { id, before: snapshot(), dx: item.x - pt.x, dz: item.z - pt.z, y: item.y };
        orbit.enabled = false;
        canvas.setPointerCapture(e.pointerId);
      }
    } else if (!id && walk) {
      looking = { x: e.clientX, y: e.clientY, q: camera.quaternion.clone() };
      canvas.setPointerCapture(e.pointerId);
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (looking) {
      const q = new T.Euler().setFromQuaternion(looking.q, "YXZ");
      q.y -= (e.clientX - looking.x) * 3e-3;
      q.x = T.MathUtils.clamp(q.x - (e.clientY - looking.y) * 3e-3, -1.35, 1.35);
      camera.quaternion.setFromEuler(q);
      return;
    }
    if (drag) {
      const pt = ground(e, drag.y);
      if (!pt) return;
      const o = items.find((x) => x.id === drag.id);
      o.x = Math.round((pt.x + drag.dx) * 100) / 100;
      o.z = Math.round((pt.z + drag.dz) * 100) / 100;
      sync(o);
      panel();
    }
  });
  const end = () => {
    looking = null;
    if (drag) {
      history(drag.before);
      drag = null;
      orbit.enabled = !walk;
      updateCount();
    }
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("wheel", (e) => {
    if (walk) move(direction("forward").multiplyScalar(-e.deltaY * 2e-3));
  }, { passive: true });
  canvas.ondragover = (e) => {
    e.preventDefault();
  };
  canvas.ondrop = (e) => {
    e.preventDefault();
    const pid = e.dataTransfer.getData("text/plain");
    pointerRay(e);
    if (pid.startsWith("finish:")) {
      const hit2 = ray.intersectObjects(Object.values(architecture.userData.finishGroups), true)[0];
      if (hit2) {
        applyRoomFinish(hit2.object.userData.roomFinish, pid.split(":")[1]);
      } else toast("\u8BF7\u628A\u8272\u5361\u62D6\u5230\u623F\u95F4\u5899\u9762\u4E0A");
      return;
    }
    const hit = ray.intersectObjects([architecture, objects], true).find((h) => h.face && h.face.normal.clone().transformDirection(h.object.matrixWorld).y > 0.6);
    const pt = catalog[pid]?.mount === "surface" && hit ? hit.point : ground(e);
    if (catalog[pid] && pt) add(pid, pt);
  };
}
function resize() {
  if (!renderer) return;
  const el = $("#canvas"), w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
}
function animate(t) {
  requestAnimationFrame(animate);
  if (!ready) return;
  const dt = Math.min((t - last) / 1e3, 0.05);
  last = t;
  if (walk && keys.size) {
    let v = new T.Vector3();
    for (const k of keys) v.add(direction(k));
    if (v.lengthSq()) move(v.normalize().multiplyScalar(dt * 1.1));
  }
  if (!walk) orbit.update();
  selectionBox?.update();
  composer.render();
  const canvas = renderer.domElement;
  canvas.dataset.camera = camera.position.toArray().map((n) => n.toFixed(3)).join(",");
  canvas.dataset.mode = isSample ? "sample" : "placement";
  canvas.dataset.selected = selected || "";
  frames++;
  if (t - fpsTime > 1e3) {
    canvas.dataset.fps = Math.round(frames * 1e3 / (t - fpsTime));
    frames = 0;
    fpsTime = t;
  }
}
async function init() {
  try {
    data = await loadData();
    for (const p of data.catalog) catalog[p.id] = p;
    M = await materials();
    scene = new T.Scene();
    scene.background = new T.Color("#070d1b");
    renderer = new T.WebGLRenderer({ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    $("#canvas").appendChild(renderer.domElement);
    camera = new T.PerspectiveCamera(66, 1, 0.055, 90);
    orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.enabled = false;
    orbit.minDistance = 1;
    orbit.maxDistance = 28;
    orbit.maxPolarAngle = Math.PI * 0.49;
    const studio = new RoomEnvironment(), pmrem = new T.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(studio, 0.04).texture;
    scene.environmentIntensity = 0.12;
    studio.dispose();
    scene.add(new T.HemisphereLight("#7c95c3", "#8c6949", 0.48));
    const moon = new T.DirectionalLight("#9fb8ee", 0.55);
    moon.position.set(7, 12, -6);
    moon.castShadow = true;
    moon.shadow.mapSize.set(2048, 2048);
    Object.assign(moon.shadow.camera, { left: -8, right: 8, top: 10, bottom: -10, near: 0.5, far: 35 });
    moon.shadow.normalBias = 8e-3;
    scene.add(moon);
    const starPositions = [];
    let seed = 42;
    const rnd = () => {
      seed = seed * 1664525 + 1013904223 >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < 850; i++) {
      const a = rnd() * Math.PI * 2, e = 0.12 + rnd() * 1.35, r = 44;
      starPositions.push(Math.cos(a) * Math.cos(e) * r + 3, Math.sin(e) * r, Math.sin(a) * Math.cos(e) * r + 5);
    }
    const sg = new T.BufferGeometry();
    sg.setAttribute("position", new T.Float32BufferAttribute(starPositions, 3));
    scene.add(new T.Points(sg, new T.PointsMaterial({ color: "#e4eaff", size: 0.065, sizeAttenuation: true })));
    const moonDisc = new T.Mesh(new T.SphereGeometry(0.42, 24, 16), new T.MeshBasicMaterial({ color: "#d8e2f1" }));
    moonDisc.position.set(-15, 22, 25);
    scene.add(moonDisc);
    for (const [x, z, power] of [[1.5, 3.8, 32], [4.4, 2.6, 24], [4.5, 6.7, 24], [1.8, 6.7, 22], [6.6, 5.3, 18], [6.6, 7, 20]]) {
      const l = new T.SpotLight("#ffcf91", power, 6, 1.18, 0.72, 2);
      l.position.set(x, 2.56, z);
      l.target.position.set(x, 0.1, z + 0.3);
      l.castShadow = true;
      l.shadow.mapSize.set(1024, 1024);
      l.shadow.normalBias = z === 7 ? 0.022 : 8e-3;
      l.shadow.bias = z === 7 ? -7e-4 : -12e-5;
      scene.add(l, l.target);
    }
    for (const [x, y, z, power] of [[0.58, 1.7, 2.4, 3], [3.35, 1.4, 2.9, 2], [0.98, 1.12, 6.65, 3.4], [4.02, 1.36, 2.02, 2.5], [3.85, 1.42, 5.48, 1.8], [5.15, 1.45, 7.9, 3], [1.3, 1.8, 4.05, 3], [6.9, 1.85, 5.2, 2], [2.3, 2.1, 10.2, 10], [2.6, 1.5, -0.7, 6], [0.87, 0.5, 9.1, 1], [3.61, 0.5, 11, 1]]) {
      const l = new T.PointLight("#ffce88", power, 4.3, 2);
      l.position.set(x, y, z);
      scene.add(l);
    }
    architecture = house(data.layout, M);
    decorateWalls(architecture, data.layout, M);
    scene.add(architecture);
    const ground2 = new T.Mesh(new T.PlaneGeometry(180, 180), new T.MeshStandardMaterial({ color: "#20291e", roughness: 1 }));
    ground2.rotation.x = -Math.PI / 2;
    ground2.position.y = -0.12;
    ground2.receiveShadow = true;
    scene.add(ground2);
    const target = new T.WebGLCubeRenderTarget(256, { type: T.HalfFloatType }), cube = new T.CubeCamera(0.05, 40, target);
    cube.position.set(1.6, 1.45, 3.7);
    scene.add(cube);
    renderer.shadowMap.needsUpdate = true;
    cube.update(renderer, scene);
    const interior = pmrem.fromCubemap(target.texture).texture;
    for (const m of [M.chrome, M.brushed, M.glass]) {
      m.envMap = interior;
      m.envMapIntensity = 0.85;
    }
    scene.remove(cube);
    target.dispose();
    pmrem.dispose();
    objects = new T.Group();
    scene.add(objects);
    grid = new T.GridHelper(16, 80, "#636f63", "#a4a999");
    grid.position.set(3.8, 0.025, 5);
    grid.visible = false;
    scene.add(grid);
    gizmo = new TransformControls(camera, renderer.domElement);
    gizmo.setSize(0.8);
    scene.add(gizmo.getHelper());
    gizmo.addEventListener("dragging-changed", (e) => orbit.enabled = !e.value && !walk);
    gizmo.addEventListener("mouseDown", () => beforeTransform = snapshot());
    gizmo.addEventListener("objectChange", () => {
      if (!selected || !gizmo.dragging) return;
      const g = instances.get(selected), o = items.find((o2) => o2.id === selected);
      o.x = g.position.x;
      o.y = T.MathUtils.clamp(g.position.y, 0, 2.6);
      o.z = g.position.z;
      o.rotation = T.MathUtils.radToDeg(g.rotation.y);
      g.position.y = o.y;
      renderer.shadowMap.needsUpdate = true;
      panel();
    });
    gizmo.addEventListener("mouseUp", () => {
      if (beforeTransform) {
        history(beforeTransform);
        beforeTransform = null;
        updateCount();
      }
    });
    const rt = new T.WebGLRenderTarget(1, 1, { type: T.HalfFloatType, samples: 4 });
    composer = new EffectComposer(renderer, rt);
    composer.addPass(new RenderPass(scene, camera));
    ao = new GTAOPass(scene, camera, 1, 1);
    ao.updateGtaoMaterial({ radius: 0.22, thickness: 0.25, distanceFallOff: 1, scale: 1.1, samples: 12 });
    ao.blendIntensity = 0.72;
    composer.addPass(ao);
    composer.addPass(new OutputPass());
    let saved;
    try {
      saved = JSON.parse(localStorage.getItem(KEY) || "null");
    } catch {
    }
    editItems = valid(saved?.items) ? saved.items : structuredClone(data.layout.placed);
    editFinishes = { ...defaults, ...saved?.finishes };
    items = structuredClone(data.layout.placed);
    buildItems();
    bindCanvas();
    $("#room-filter").innerHTML = '<option value="">\u5168\u90E8\u7A7A\u95F4</option>' + [...new Set(data.catalog.map((p) => p.room))].map((r) => `<option>${esc(r)}</option>`).join("");
    inventory();
    finishLibrary();
    new ResizeObserver(resize).observe($("#canvas"));
    resize();
    visit("bedroom");
    ready = true;
    $("#loading").hidden = true;
    requestAnimationFrame(animate);
  } catch (e) {
    $("#loading").innerHTML = "<h2>\u623F\u95F4\u6682\u672A\u52A0\u8F7D\u5B8C\u6210</h2><p>" + esc(e.message) + '</p><button onclick="location.reload()">\u91CD\u65B0\u52A0\u8F7D</button>';
    console.error(e);
  }
}
$("#sample").onclick = () => setMode(true);
$("#edit").onclick = () => setMode(false);
$$("[data-room]").forEach((b) => b.onclick = () => visit(b.dataset.room));
$$("[data-tool]").forEach((b) => b.onclick = () => setTool(b.dataset.tool));
$("#inside").onclick = () => visit("dining");
$("#overview").onclick = () => planView(false);
$("#top").onclick = () => planView(true);
$("#grid").onclick = () => {
  grid.visible = !grid.visible;
  $("#grid").classList.toggle("active", grid.visible);
};
$("#search").oninput = inventory;
$("#room-filter").onchange = inventory;
for (const [id, key] of [["px", "x"], ["py", "y"], ["pz", "z"], ["angle", "rotation"]]) $("#" + id).onblur = $("#" + id).onchange = (e) => {
  const n = Number(e.target.value);
  if (Number.isFinite(n)) change((o) => o[key] = n);
};
$("#delete").onclick = remove;
$("#duplicate").onclick = () => {
  const o = items.find((o2) => o2.id === selected);
  if (o) add(o.pid, new T.Vector3(o.x + 0.3, 0, o.z + 0.3));
};
$("#undo").onclick = () => {
  if (!undo.length || isSample) return;
  redo.push(snapshot());
  restore(undo.pop());
  store();
  $("#undo").disabled = !undo.length;
  $("#redo").disabled = false;
};
$("#redo").onclick = () => {
  if (!redo.length || isSample) return;
  undo.push(snapshot());
  restore(redo.pop());
  store();
  $("#redo").disabled = !redo.length;
  $("#undo").disabled = false;
};
$("#reset").onclick = () => {
  if (isSample) setMode(false);
  const before = snapshot();
  items = structuredClone(data.layout.placed);
  finishes = { ...defaults };
  applyFinishes();
  buildItems();
  history(before);
  toast("\u5DF2\u8FD8\u539F\u6837\u677F\u95F4\uFF0C\u53EF\u64A4\u9500");
};
$("#mobile-plan").onclick = () => $("#workspace").classList.toggle("plan-open");
$("#mobile-library").onclick = () => $("#workspace").classList.toggle("library-open");
$("#export").onclick = () => {
  const blob = new Blob([JSON.stringify({ format: "bauhaus-layout-v2", items, finishes, created: (/* @__PURE__ */ new Date()).toISOString() }, null, 2)], { type: "application/json" }), a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "\u5305\u8C6A\u65AF\u6446\u653E\u65B9\u6848.json";
  a.click();
  URL.revokeObjectURL(a.href);
};
$("#import").onclick = () => $("#file").click();
$("#file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    if (f.size > 2e6) throw Error("\u6587\u4EF6\u8FC7\u5927");
    const d = JSON.parse(await f.text());
    if (!["bauhaus-layout-v1", "bauhaus-layout-v2"].includes(d.format) || !valid(d.items)) throw Error("\u65B9\u6848\u683C\u5F0F\u6216\u5750\u6807\u65E0\u6548");
    if (isSample) setMode(false);
    const before = snapshot();
    items = d.items;
    finishes = { ...defaults };
    for (const [k, v] of Object.entries(d.finishes || {})) {
      if (k in defaults && /^#[0-9a-f]{6}$/i.test(v)) finishes[k] = v;
    }
    applyFinishes();
    buildItems();
    history(before);
    toast("\u65B9\u6848\u5DF2\u5BFC\u5165");
  } catch (e2) {
    toast(e2.message);
  }
  e.target.value = "";
};
$("#help").onclick = () => $("#guide").showModal();
$("#close-guide").onclick = () => $("#guide").close();
document.addEventListener("keydown", (e) => {
  if (!ready || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) || $("#guide").open) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z") {
    e.preventDefault();
    (e.shiftKey ? $("#redo") : $("#undo")).click();
    return;
  }
  if (walk && ["w", "a", "s", "d"].includes(k)) {
    keys.add(k);
    e.preventDefault();
  }
  if (isSample) return;
  if (["v", "g", "r", "h"].includes(k)) setTool({ v: "drag", g: "translate", r: "rotate", h: "height" }[k]);
  if (k === "delete") remove();
  if (k === "escape") select(null);
  if (selected && ["arrowleft", "arrowright", "arrowup", "arrowdown", "pageup", "pagedown"].includes(k)) {
    e.preventDefault();
    const s = e.shiftKey ? 0.01 : 0.05;
    change((o) => {
      if (k === "arrowleft") o.x -= s;
      if (k === "arrowright") o.x += s;
      if (k === "arrowup") o.z -= s;
      if (k === "arrowdown") o.z += s;
      if (k === "pageup") o.y += s;
      if (k === "pagedown") o.y -= s;
    });
  }
});
document.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener("blur", () => {
  keys.clear();
  looking = null;
});
$$("[data-walk]").forEach((b) => {
  let timer;
  const step = () => {
    if (!walk) visit("dining");
    move(direction(b.dataset.walk).multiplyScalar(0.1));
  };
  b.onpointerdown = (e) => {
    e.preventDefault();
    step();
    timer = setInterval(step, 90);
    b.setPointerCapture(e.pointerId);
  };
  b.onpointerup = b.onpointercancel = () => clearInterval(timer);
});
function applyRoomFinish(room, color) {
  if (!room || !/^#[0-9a-f]{6}$/i.test(color)) return;
  if (isSample) setMode(false);
  const before = snapshot();
  finishes[room] = color;
  applyFinishes();
  history(before);
  toast(roomNames[room] + " \xB7 \u56DB\u5468\u5899\u9762\u5DF2\u7EDF\u4E00");
  $("#finish-room").value = room;
  $("#finish-color").value = color;
}
function finishLibrary() {
  $("#finish-room").innerHTML = Object.entries(roomNames).map(([key, name]) => `<option value="${key}">${name}</option>`).join("");
  $("#finish-cards").innerHTML = data.finishes.map((p) => `<article class="candidate"><figure draggable="true" data-finish="${p.id}"><img src="${p.photo}" draggable="false" alt="${esc(p.name)}"></figure><h3>${esc(p.name)}</h3><p>${esc(p.sizeText)}</p>${productInfo({ ...p, note: (p.note || "").replace(/未读评论正文。?/g, ""), review: p.review || "\u6682\u65E0\u5DF2\u6838\u9A8C\u8BC4\u8BBA\uFF1B\u672A\u8BFB\u53D6\u8BC4\u8BBA\u6B63\u6587\u3002" })}<button data-apply-finish="${p.id}">\u5E94\u7528\u5F53\u524D\u989C\u8272\u5230\u6574\u95F4</button></article>`).join("");
  $$("[data-finish]").forEach((el) => el.ondragstart = (e) => e.dataTransfer.setData("text/plain", "finish:" + $("#finish-color").value));
  $$("[data-apply-finish]").forEach((el) => el.onclick = () => applyRoomFinish($("#finish-room").value, $("#finish-color").value));
  $("#finish-room").onchange = () => $("#finish-color").value = finishes[$("#finish-room").value];
  $("#apply-color").onclick = () => applyRoomFinish($("#finish-room").value, $("#finish-color").value);
  $$("[data-palette]").forEach((el) => el.onclick = () => {
    $("#finish-color").value = el.dataset.palette;
    applyRoomFinish($("#finish-room").value, el.dataset.palette);
  });
  $("#show-finishes").onclick = () => {
    $("#furniture-library").hidden = true;
    $("#finish-library").hidden = false;
  };
  $("#show-furniture").onclick = () => {
    $("#furniture-library").hidden = false;
    $("#finish-library").hidden = true;
  };
}
init();
