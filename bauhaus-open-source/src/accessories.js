import * as T from "three";
const textureCache = /* @__PURE__ */ new Map();
function painting(p) {
  if (textureCache.has(p.id)) return textureCache.get(p.id);
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const x = c.getContext("2d");
  x.fillStyle = "#e8dfc8";
  x.fillRect(0, 0, 512, 512);
  const colors = ["#a33f27", "#163d5a", "#d3a347", "#2a4636"];
  for (let i = 0; i < 12; i++) {
    x.fillStyle = colors[(i + p.slot) % 4];
    const px = 32 + i % 3 * 155, py = 32 + Math.floor(i / 3) * 112;
    if (i % 2) {
      x.fillRect(px, py, 125, 90);
    } else {
      x.beginPath();
      x.arc(px + 58, py + 45, 44, 0, Math.PI * 2);
      x.fill();
    }
  }
  const t = new T.CanvasTexture(c);
  t.colorSpace = T.SRGBColorSpace;
  textureCache.set(p.id, t);
  return t;
}
function accessory(p, g, M, H) {
  const { box: B, cylinder: C, ball: S, rod: R } = H;
  let [w, d, h] = p.dims, k = p.kind, n = p.name, v = p.slot;
  const metal = M.brushed, wood = /黑胡桃|棕|木墨/.test(n) ? M.darkWood : M.wood, col = /红|苹果/.test(n) ? M.red : /绿/.test(n) ? M.green : /蓝/.test(n) ? M.blue : /黄|KOPPAR/.test(n) ? M.yellow : M.white;
  const bx = (a, b, c, x, y, z, m = col, r = 6e-3) => B(g, a, b, c, x, y, z, m, r), cy = (a, b, c, x, y, z, m = col, seg = 24) => C(g, a, b, c, x, y, z, m, seg), rod = (a, b, r = 5e-3, m = metal) => R(g, a, b, r, m);
  const tor = (radius, tube, x, y, z, m = metal, rx = 0) => {
    const o = new T.Mesh(new T.TorusGeometry(radius, tube, 8, 36), m);
    o.position.set(x, y, z);
    o.rotation.x = rx;
    g.add(o);
    return o;
  };
  const towel = (x, y, z, a = w, b = d, c = h) => {
    bx(a, c, b, x, y + c / 2, z, M.cream, 0.025);
    for (let j = 0; j < 3; j++) bx(a * 0.95, 2e-3, b * 0.035, x, y + c + 2e-3, z - b * 0.4 + j * b * 0.07, M.linen, 1e-3);
  };
  const bowl = (x, y, z, r, ht, m = M.white) => {
    const pts = [[r * 0.6, 0], [r * 0.78, 0.015], [r, ht], [r * 0.94, ht], [r * 0.72, 0.025], [0, 0.025]].map((a) => new T.Vector2(...a));
    const o = new T.Mesh(new T.LatheGeometry(pts, 32), m);
    o.position.set(x, y, z);
    g.add(o);
  };
  if (["wallshelf", "wiretray"].includes(k)) {
    if (/木墨/.test(n)) {
      for (let j = 0; j < 3; j++) bx(w, 0.025, d, 0, 0.07 + j * (h - 0.14) / 2, 0, wood);
      for (let a of [-1, 0, 1]) bx(0.035, h, 0.04, a * w * 0.45, h / 2, d * 0.35, wood);
      for (let j = 0; j < 2; j++) for (let i = 0; i < 5; i++) {
        const ht = 0.12 + i % 3 * 0.02;
        bx(0.025, ht, d * 0.68, -w * 0.36 + i * 0.055, 0.084 + j * (h - 0.14) / 2 + ht / 2, 0, [M.paper, M.darkWood, M.blue][i % 3], 1e-3);
      }
    } else {
      bx(w, 0.018, d, 0, 0.012, 0, /铁|CD架/.test(n) ? M.black : wood);
      bx(w, h, 0.012, 0, h / 2, d / 2 - 6e-3, /铁|CD架/.test(n) ? M.black : wood);
      bx(w, Math.min(0.035, h), 0.012, 0, Math.min(0.035, h) / 2, -d / 2, M.black);
      if (/杂志|书报/.test(n)) {
        rod([-w / 2, h * 0.55, -d * 0.43], [w / 2, h * 0.55, -d * 0.43]);
        for (const a of [-1, 1]) rod([a * w * 0.47, 0, -d * 0.43], [a * w * 0.47, h * 0.55, -d * 0.43]);
      }
    }
  } else if (k === "clock") {
    bx(w, h, d, 0, h / 2, 0, metal, 0.018);
    cy(w * 0.43, w * 0.43, 5e-3, 0, h / 2, -d * 0.51, col, 64).rotation.x = Math.PI / 2;
    for (let i = 0; i < 12; i++) {
      const a = i * Math.PI / 6;
      const b = bx(6e-3, 0.018, 3e-3, Math.sin(a) * w * 0.36, h / 2 + Math.cos(a) * h * 0.36, -d * 0.56, M.paper, 1e-3);
      b.rotation.z = -a;
    }
    rod([0, h / 2, -d * 0.58], [w * 0.22, h * 0.62, -d * 0.58], 4e-3, M.white);
    rod([0, h / 2, -d * 0.59], [-w * 0.12, h * 0.74, -d * 0.59], 4e-3, M.white);
    if (v === 2) bx(w, 0.012, d * 2.5, 0, 7e-3, -d * 0.7, metal);
  } else if (k === "record") {
    const disc = cy(w / 2, w / 2, d * 0.35, 0, h / 2, 0, M.black, 64);
    disc.rotation.x = Math.PI / 2;
    cy(w * 0.16, w * 0.16, 2e-3, 0, h / 2, -d / 2, M.paper, 32).rotation.x = Math.PI / 2;
    for (let i = 0; i < 8; i++) tor(w * (0.22 + i * 0.032), 7e-4, 0, h / 2, -d * 0.22, metal);
    cy(5e-3, 5e-3, 3e-3, 0, h / 2, -d * 0.6, M.black, 12).rotation.x = Math.PI / 2;
  } else if (["media", "cabinet", "storage", "filebox", "basket", "bin"].includes(k)) {
    const open = ["basket", "bin", "filebox"].includes(k);
    const mat = k === "basket" ? M.linen : k === "media" ? wood : col;
    bx(w, 0.018, d, 0, 0.015, 0, mat);
    for (const a of [-1, 1]) bx(0.018, h, d, a * (w / 2 - 9e-3), h / 2, 0, mat);
    bx(w, h, 0.018, 0, h / 2, d / 2 - 9e-3, mat);
    bx(w, h * (open ? 0.35 : 1), 0.018, 0, h * (open ? 0.175 : 0.5), -d / 2 + 9e-3, mat);
    if (!open) {
      bx(w, 0.018, d, 0, h - 9e-3, 0, mat);
      for (const a of [-1, 1]) bx(0.012, h - 0.05, 8e-3, a * w * 0.16, h / 2, -d / 2 - 6e-3, metal);
    } else {
      for (let i = 0; i < Math.min(18, w / 0.025); i++) rod([-w / 2 + i * w / 18, 0.01, -d / 2 - 2e-3], [-w / 2 + i * w / 18, h, -d / 2 - 2e-3], 15e-4, M.darkWood);
    }
  } else if (["cushion", "pillow", "bedding", "cover", "mattress", "throw", "napkins", "towels", "placemat", "runner", "furniturecover"].includes(k)) {
    if (k === "towels" || k === "napkins") {
      for (let i = 0; i < 3; i++) towel(0, i * h / 3, 0, w, d, h / 3);
    } else {
      bx(w, h, d, 0, h / 2, 0, k === "mattress" ? M.cream : /蓝|KLÖSAN|深蓝/.test(n) ? M.blue : k === "furniturecover" ? M.black : col, 0.035);
      if (k === "cushion" || k === "pillow" || k === "mattress") {
        for (let x of [-1, 1]) for (let z of [-1, 1]) S(g, x * w * 0.23, h * 0.96, z * d * 0.23, 0.01, 4e-3, 0.01, M.gray);
      }
      if (k === "bedding" || k === "throw") {
        for (let i = 0; i < 12; i++) rod([-w / 2, h + 2e-3, -d / 2 + i * d / 12], [w / 2, h + 2e-3, -d / 2 + i * d / 12], 15e-4, M.linen);
      }
    }
  } else if (k === "hooks" || k === "towelrail") {
    bx(w, 0.04, 0.025, 0, h * 0.7, d * 0.35, wood);
    for (let i = 0; i < (k === "hooks" ? 5 : 2); i++) {
      const x = -w * 0.4 + i * w * 0.8 / (k === "hooks" ? 4 : 1);
      rod([x, h * 0.7, d * 0.4], [x, h * 0.7, -d * 0.4]);
      if (k === "hooks") rod([x, h * 0.7, -d * 0.4], [x, h * 0.9, -d * 0.4]);
    }
    if (k === "towelrail") rod([-w * 0.4, h * 0.7, -d * 0.4], [w * 0.4, h * 0.7, -d * 0.4]);
  } else if (k === "curtain") {
    const geo = new T.PlaneGeometry(w, h, 80, 12), pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) pos.setZ(i, Math.sin((pos.getX(i) / w + 0.5) * Math.PI * 24) * d * 0.46);
    geo.computeVertexNormals();
    const mat = M.linen.clone();
    mat.side = T.DoubleSide;
    const o = new T.Mesh(geo, mat);
    o.position.y = h / 2;
    g.add(o);
    rod([-w / 2, h, 0], [w / 2, h, 0], 0.012, metal);
  } else if (k === "blind") {
    for (let y = 0.025; y < h; y += 0.035) bx(w, 0.012, d, 0, y, 0, wood, 2e-3);
    for (let x of [-w * 0.3, w * 0.3]) rod([x, 0, 0], [x, h, 0], 15e-4, M.black);
  } else if (k === "curtainrod") {
    rod([-w / 2, h / 2, 0], [w / 2, h / 2, 0], h * 0.16, metal);
    for (let x of [-w / 2, w / 2]) S(g, x, h / 2, 0, h * 0.3, h * 0.3, h * 0.3, metal);
  } else if (["tasklamp", "walllamp", "nightlight", "bulb", "decorlamp", "donutlamp", "lantern"].includes(k)) {
    if (k === "donutlamp") {
      tor(w * 0.32, w * 0.15, 0, h / 2, 0, M.amber);
    } else if (k === "tasklamp") {
      cy(w * 0.23, w * 0.25, 0.02, 0, 0.01, 0, metal);
      rod([0, 0.02, 0], [w * 0.15, h * 0.48, 0], 0.012);
      rod([w * 0.15, h * 0.48, 0], [-w * 0.18, h * 0.78, 0], 0.012);
      for (let z of [-0.014, 0.014]) rod([0.02, 0.06, z], [w * 0.17, h * 0.46, z], 4e-3);
      const sh = cy(w * 0.09, w * 0.22, h * 0.2, -w * 0.18, h * 0.85, 0, /TERTIAL/.test(n) ? M.black : metal);
      sh.rotation.z = 0.25;
      S(g, -w * 0.18, h * 0.76, 0, 0.04, 0.02, 0.04, M.glow);
    } else if (k === "lantern") {
      cy(w * 0.4, w * 0.4, 0.025, 0, 0.012, 0, wood);
      cy(w * 0.43, w * 0.4, 0.025, 0, h * 0.84, 0, wood);
      for (let i = 0; i < 20; i++) {
        let a = i * Math.PI / 10;
        rod([Math.cos(a) * w * 0.38, 0.02, Math.sin(a) * w * 0.38], [Math.cos(a) * w * 0.4, h * 0.84, Math.sin(a) * w * 0.4], 5e-3, wood);
      }
      S(g, 0, h * 0.42, 0, w * 0.19, h * 0.28, w * 0.19, M.glow);
      tor(w * 0.23, 8e-3, 0, h * 0.85, 0, metal);
    } else if (k === "walllamp") {
      bx(w * 0.34, h * 0.4, 0.02, 0, h * 0.5, d * 0.4, metal);
      rod([0, h * 0.5, d * 0.4], [0, h * 0.6, -d * 0.2], 0.01);
      cy(w * 0.45, w * 0.5, h * 0.42, 0, h * 0.75, -d * 0.2, M.paper);
      S(g, 0, h * 0.54, -d * 0.2, 0.04, 0.025, 0.04, M.glow);
    } else if (k === "bulb" || k === "nightlight") {
      cy(w * 0.3, w * 0.3, h * 0.3, 0, h * 0.15, 0, metal);
      S(g, 0, h * 0.64, 0, w * 0.5, h * 0.36, d * 0.5, M.glow);
    } else {
      cy(w * 0.3, w * 0.35, h * 0.08, 0, h * 0.04, 0, col);
      rod([0, h * 0.05, 0], [0, h * 0.65, 0], 0.013, metal);
      S(g, 0, h * 0.72, 0, w * 0.48, h * 0.28, d * 0.48, /蛋挞/.test(n) ? M.yellow : M.paper);
      cy(w * 0.44, w * 0.44, 8e-3, 0, h * 0.69, 0, M.glow);
    }
  } else if (["cookware", "dishes", "cups", "containers", "tray", "soapset", "dishrack", "cuttingboard", "utensils"].includes(k)) {
    if (k === "cuttingboard") {
      bx(w, h, d, 0, h / 2, 0, wood, 0.02);
      bx(w * 0.62, 3e-3, 0.028, 0, h + 3e-3, 0, metal);
      bx(w * 0.2, 0.015, 0.026, -w * 0.37, h + 8e-3, 0, M.black);
    } else if (k === "dishrack") {
      for (const x of [-w * 0.47, w * 0.47]) for (const z of [-d * 0.47, d * 0.47]) rod([x, 0, z], [x, h, z]);
      for (let y of [0.02, h]) for (const z of [-d * 0.47, d * 0.47]) rod([-w * 0.47, y, z], [w * 0.47, y, z]);
      for (let i = 0; i < 10; i++) rod([-w * 0.45 + i * w * 0.1, 0.02, -d * 0.45], [-w * 0.45 + i * w * 0.1, h * 0.7, d * 0.45], 3e-3);
    } else if (k === "utensils") {
      cy(w * 0.28, w * 0.26, h * 0.5, 0, h * 0.25, 0, metal);
      for (let i = 0; i < 4; i++) {
        let x = (i - 1.5) * w * 0.16;
        rod([x, h * 0.3, 0], [x, h * 0.9, 0], 5e-3, metal);
        S(g, x, h * 0.9, 0, w * 0.08, h * 0.09, 6e-3, metal);
      }
    } else if (k === "tray") {
      bx(w, 8e-3, d, 0, 4e-3, 0, metal, 0.02);
      for (const x of [-1, 1]) bx(8e-3, h, d, x * (w / 2 - 4e-3), h / 2, 0, metal);
      for (const z of [-1, 1]) bx(w, h, 8e-3, 0, h / 2, z * (d / 2 - 4e-3), metal);
    } else if (k === "soapset") {
      cy(w * 0.16, w * 0.16, h * 0.78, -w * 0.25, h * 0.39, 0, M.glass);
      rod([-w * 0.25, h * 0.78, 0], [-w * 0.25, h, 0], 0.012);
      rod([-w * 0.25, h, 0], [-w * 0.05, h, 0], 9e-3);
      bowl(w * 0.25, 0, 0, w * 0.2, h * 0.25, M.white);
    } else if (k === "cookware") {
      bowl(0, 0.015, 0, d * 0.43, h, metal);
      for (let a of [-1, 1]) rod([a * d * 0.4, h * 0.75, 0], [a * w * 0.48, h * 0.75, 0], 0.015, M.black);
      cy(d * 0.44, d * 0.44, 8e-3, 0, h + 0.014, 0, metal);
      S(g, 0, h + 0.035, 0, 0.035, 0.02, 0.02, M.black);
    } else if (k === "cups") {
      for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) bowl((i - 1) * w * 0.29, 0, (j - 0.5) * d * 0.52, Math.min(w / 8, d / 5), h, M.glass);
    } else if (k === "containers") {
      for (let i = 0; i < 3; i++) {
        bx(w * 0.28, h * (0.6 + i * 0.2), d * 0.9, (i - 1) * w * 0.32, h * (0.3 + i * 0.1), 0, M.glass, 0.015);
        bx(w * 0.3, 0.014, d * 0.94, (i - 1) * w * 0.32, h * (0.6 + i * 0.2), 0, /PRUTA/.test(n) ? M.green : col);
      }
    } else for (let i = 0; i < 4; i++) bowl(0, i * 0.014, 0, w * 0.48, h * 0.24, M.white);
  } else if (["bathmirror", "bathset"].includes(k)) {
    if (k === "bathmirror") {
      bx(w, h, d, 0, h / 2, 0, M.white);
      bx(w * 0.95, h * 0.95, 5e-3, 0, h / 2, -d / 2, M.chrome);
    } else {
      cy(w * 0.2, w * 0.2, h * 0.5, -w * 0.24, h * 0.25, 0, M.white);
      rod([-w * 0.24, h * 0.3, 0], [-w * 0.24, h * 0.92, 0], 9e-3, metal);
      cy(w * 0.23, w * 0.23, 0.02, w * 0.25, h * 0.8, 0, M.white).rotation.x = Math.PI / 2;
    }
  } else if (k === "dryer") {
    for (let z of [-d * 0.4, d * 0.4]) {
      rod([-w * 0.4, 0.02, z], [w * 0.35, h * 0.85, z], 8e-3);
      rod([w * 0.4, 0.02, z], [-w * 0.35, h * 0.85, z], 8e-3);
    }
    for (let i = 0; i < 12; i++) rod([-w * 0.5 + i * w / 11, h, -d * 0.5], [-w * 0.5 + i * w / 11, h, d * 0.5], 3e-3);
    for (let z of [-d * 0.5, d * 0.5]) rod([-w * 0.5, h, z], [w * 0.5, h, z]);
  } else if (k === "mop") {
    bx(w, 0.03, d, 0, 0.015, 0, M.white, 0.02);
    rod([0, 0.03, 0], [w * 0.12, h, 0], 0.012, metal);
  } else if (k === "lintroller") {
    cy(w * 0.4, w * 0.4, d, 0, h * 0.65, 0, M.paper).rotation.z = Math.PI / 2;
    rod([w * 0.45, h * 0.65, 0], [w * 0.45, 0, 0], 0.012, M.white);
  } else if (k === "clips") {
    for (let i = 0; i < 8; i++) {
      let a = i * Math.PI / 4;
      bx(w * 0.12, h * 0.5, d * 0.12, Math.sin(a) * w * 0.35, h * 0.25, Math.cos(a) * d * 0.35, M.green);
    }
    tor(w * 0.38, 5e-3, 0, h * 0.7, 0, metal, Math.PI / 2);
    rod([0, h * 0.7, 0], [0, h, 0]);
  } else if (k === "stepstool") {
    bx(w, 0.025, d, 0, h - 0.012, 0, M.orange);
    for (let x of [-1, 1]) for (let z of [-1, 1]) rod([x * w * 0.48, 0, z * d * 0.48], [x * w * 0.37, h - 0.02, z * d * 0.37], 0.014, M.orange);
    bx(w * 0.75, 0.02, d * 0.8, 0, h * 0.35, 0, M.orange);
  } else if (k === "watering") {
    cy(w * 0.23, w * 0.22, h * 0.7, 0, h * 0.35, 0, M.white);
    rod([w * 0.12, h * 0.2, 0], [w * 0.5, h * 0.85, 0], 0.035, M.white);
    tor(h * 0.27, 0.018, -w * 0.18, h * 0.5, 0, M.white);
  } else if (k === "umbrella") {
    cy(0.022, 0.022, h * 0.94, 0, h * 0.47, 0, metal);
    cy(w * 0.01, w * 0.5, h * 0.16, 0, h * 0.9, 0, M.cream, 12);
    cy(0.25, 0.28, 0.05, 0, 0.025, 0, M.black);
  } else if (k === "woodhand") {
    bx(w * 0.7, h * 0.4, d * 0.8, 0, h * 0.27, 0, wood, 0.025);
    for (let i = 0; i < 4; i++) {
      const x = (i - 1.5) * w * 0.18;
      rod([x, h * 0.44, 0], [x, h * (0.9 + Math.sin(i) * 0.05), 0], w * 0.065, wood);
      S(g, x, h * 0.65, 0, w * 0.071, 0.02, w * 0.071, wood);
    }
    rod([-w * 0.3, h * 0.28, 0], [-w * 0.48, h * 0.57, 0], w * 0.09, wood);
    cy(w * 0.22, w * 0.25, h * 0.08, 0, h * 0.04, 0, wood);
  } else if (k === "foldchair") {
    bx(w * 0.8, 0.04, d * 0.7, 0, h * 0.52, 0, col);
    for (let x of [-w * 0.36, w * 0.36]) {
      rod([x, 0, -d * 0.35], [x, h * 0.56, d * 0.3], 0.011);
      rod([x, 0, d * 0.35], [x, h * 0.95, -d * 0.3], 0.011);
    }
    bx(w * 0.85, h * 0.22, 0.025, 0, h * 0.86, -d * 0.3, col);
  } else return false;
  return true;
}
export {
  accessory
};
