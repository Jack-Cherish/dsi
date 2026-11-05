import * as T from "three";
function photoModel(p, g, M, H) {
  const { box: B, cylinder: C, ball: S, rod: R } = H, [w, d, h] = p.dims, k = p.modelV4;
  if (!k) return false;
  const tube = (pts, r, m) => {
    const o = new T.Mesh(new T.TubeGeometry(new T.CatmullRomCurve3(pts.map((a) => new T.Vector3(...a))), 36, r, 8, false), m);
    g.add(o);
    return o;
  };
  const lathe = (pts, m) => {
    const o = new T.Mesh(new T.LatheGeometry(pts.map((a) => new T.Vector2(...a)), 56), m);
    g.add(o);
    return o;
  };
  const ring = (rad, r, x, y, z, m, axis = "y") => {
    const o = new T.Mesh(new T.TorusGeometry(rad, r, 10, 64), m);
    o.position.set(x, y, z);
    if (axis === "y") o.rotation.x = Math.PI / 2;
    g.add(o);
    return o;
  };
  const book = (x, y, z, ww, dd, hh, col) => {
    B(g, ww, hh, dd, x, y, z, col, 1e-3);
    B(g, ww - 5e-3, hh - 4e-3, dd - 8e-3, x, y, z - 3e-3, M.paper, 1e-3);
  };
  if (k === "sayl") {
    const sy = 0.47, white = M.white;
    C(g, 0.024, 0.032, 0.34, 0, 0.25, 0, M.brushed);
    C(g, 0.045, 0.045, 0.11, 0, 0.13, 0, white);
    for (let i = 0; i < 5; i++) {
      let a = i * Math.PI * 2 / 5;
      const x = Math.cos(a) * w * 0.44, z = Math.sin(a) * d * 0.44;
      R(g, [0, 0.15, 0], [x, 0.075, z], 0.021, white);
      for (let j of [-1, 1]) {
        const wheel = C(g, 0.031, 0.031, 0.017, x + j * 0.014, 0.033, z, M.black, 20);
        wheel.rotation.z = Math.PI / 2;
      }
    }
    B(g, w * 0.69, 0.065, d * 0.65, 0, sy, -0.035, M.gray, 0.032);
    B(g, w * 0.57, 0.035, d * 0.44, 0, sy - 0.05, 0, M.black, 0.012);
    const bw = (y) => w * (0.29 + 0.07 * Math.sin((y - 0.53) / (h - 0.53) * Math.PI));
    const bz = (y) => d * 0.27 + 0.045 * Math.sin((y - 0.53) / (h - 0.53) * Math.PI) - 0.05 * (y - 0.53);
    const ymin = 0.53, ymax = h - 0.025;
    for (let j = 0; j <= 14; j++) {
      const y = ymin + (ymax - ymin) * j / 14, ww = bw(y);
      tube([[-ww, y, bz(y)], [0, y - 5e-3, bz(y) - 0.034], [ww, y, bz(y)]], 35e-4, white);
    }
    for (let i = -8; i <= 8; i++) {
      const pts = [];
      for (let j = 0; j <= 10; j++) {
        const y = ymin + (ymax - ymin) * j / 10;
        pts.push([bw(y) * i / 8, y, bz(y) - 0.034 * (1 - (i / 8) ** 2)]);
      }
      tube(pts, 35e-4, white);
    }
    tube([[0, 0.36, d * 0.18], [0, 0.65, d * 0.36], [0, h - 0.1, d * 0.31]], 0.017, white);
    for (let a of [-1, 1]) tube([[0, 0.65, d * 0.36], [a * w * 0.18, 0.77, d * 0.35], [a * w * 0.28, h - 0.035, d * 0.27]], 0.012, white);
    for (let s of [-1, 1]) {
      R(g, [s * w * 0.32, 0.43, 0], [s * w * 0.36, 0.64, 0], 0.013, white);
      B(g, 0.075, 0.028, 0.23, s * w * 0.36, 0.66, -0.035, white, 0.012);
    }
  } else if (k === "cart") {
    for (let x of [-1, 1]) for (let z of [-1, 1]) {
      R(g, [x * w * 0.38, 0.035, z * d * 0.36], [x * w * 0.38, 0.08, z * d * 0.36], 9e-3, M.brushed);
      C(g, 0.025, 0.025, 0.025, x * w * 0.38, 0.027, z * d * 0.36, M.black, 16).rotation.z = Math.PI / 2;
    }
    B(g, w - 0.035, h - 0.08, d, 0, (h + 0.08) / 2, 0, M.cream, 0.014);
    for (let j = 0; j < 3; j++) {
      let y = 0.1 + (h - 0.11) * (j + 0.5) / 3;
      B(g, w - 0.055, (h - 0.11) / 3 - 0.016, 0.018, 0, y, -d / 2 - 2e-3, M.cream, 4e-3);
      B(g, w - 0.09, 0.015, 9e-3, 0, y + (h - 0.11) / 6 - 0.014, -d / 2 - 0.014, M.black, 2e-3);
    }
    for (let s of [-1, 1]) B(g, 0.035, h - 0.065, d, s * (w / 2 - 0.018), (h + 0.065) / 2, 0, M.cream, 0.014);
  } else if (k === "glassTable") {
    C(g, w / 2, w / 2, 0.01, 0, h - 5e-3, 0, M.glass, 72);
    B(g, 0.5, 8e-3, 0.31, 0, 0.035, 0, M.glass, 0.02);
    for (let y of [0.018, h - 0.022]) tube([[-0.19, y, -0.18], [-0.25, y, -0.12], [-0.25, y, 0.12], [-0.19, y, 0.18], [0.19, y, 0.18], [0.25, y, 0.12], [0.25, y, -0.12], [0.19, y, -0.18], [-0.19, y, -0.18]], 9e-3, M.chrome);
    for (let x of [-0.19, 0.19]) for (let z of [-0.13, 0.13]) R(g, [x, 0.02, z], [x, h - 0.02, z], 9e-3, M.chrome);
  } else if (k === "tower") {
    B(g, w, 0.012, d, 0, 6e-3, 0, M.gray, 5e-3);
    B(g, 0.023, h, 0.028, 0, h / 2, d * 0.24, M.gray, 4e-3);
    for (let j = 0; j < 9; j++) {
      let y = 0.12 + j * (h - 0.2) / 8;
      B(g, w * 0.65, 8e-3, d * 0.7, 0, y, -d * 0.035, M.gray, 3e-3);
      for (let i = 0; i < 3; i++) book(i % 2 * 9e-3, y + 0.019 + i * 0.022, -0.015, w * 0.63, d * 0.69, 0.018, [M.paper, M.darkWood, M.gray][(j + i) % 3]);
    }
  } else if (k === "gridCabinet") {
    B(g, w, h - 0.08, d, 0, (h + 0.08) / 2, 0, M.wood, 5e-3);
    for (let x of [-1, 1]) for (let z of [-1, 1]) B(g, 0.055, 0.08, 0.055, x * w * 0.43, 0.04, z * d * 0.37, M.darkWood, 5e-3);
    for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) {
      const xx = -w / 2 + w * (x + 0.5) / 4, yy = 0.09 + (h - 0.11) * (y + 0.5) / 4;
      B(g, w / 4 - 5e-3, (h - 0.11) / 4 - 6e-3, 0.018, xx, yy, -d / 2 - 3e-3, M.wood, 2e-3);
      B(g, 0.033, 0.033, 8e-3, xx, yy, -d / 2 - 0.016, M.black, 2e-3);
    }
  } else if (k === "threeCabinet") {
    for (let x of [-1, 1]) for (let z of [-1, 1]) {
      R(g, [x * w * 0.48, 0.02, z * d * 0.46], [x * w * 0.48, h, z * d * 0.46], 0.011, M.chrome);
      S(g, x * w * 0.48, h, z * d * 0.46, 0.016, 0.016, 0.016, M.chrome);
    }
    for (let y of [0.07, h * 0.38, h * 0.72, h - 0.02]) B(g, w - 0.022, 0.018, d, 0, y, 0, M.wood, 3e-3);
    for (let s of [-1, 1]) B(g, 0.015, h - 0.1, d, s * w * 0.46, (h + 0.1) / 2, 0, M.wood, 2e-3);
    const face = new T.Shape();
    face.moveTo(-w * 0.44, h * 0.74);
    face.lineTo(w * 0.44, h * 0.74);
    face.lineTo(w * 0.44, h * 0.96);
    face.lineTo(-w * 0.44, h * 0.96);
    face.closePath();
    for (let x of [-w * 0.29, 0, w * 0.29]) {
      const hole = new T.Path();
      hole.absarc(x, h * 0.85, w * 0.105, 0, Math.PI * 2, true);
      face.holes.push(hole);
    }
    const fm = new T.Mesh(new T.ExtrudeGeometry(face, { depth: 0.012, bevelEnabled: false, curveSegments: 32 }), M.cream);
    fm.position.z = -d / 2;
    g.add(fm);
    B(g, w * 0.88, h * 0.3, 0.022, 0, h * 0.55, -d / 2, M.wood, 2e-3);
    for (let j = 0; j < 5; j++) B(g, w * 0.55, 8e-3, 7e-3, w * 0.1, h * 0.43 + j * 0.047, -d / 2 - 0.014, M.darkWood, 1e-3);
    C(g, w * 0.085, w * 0.085, 0.012, -w * 0.28, h * 0.55, -d / 2 - 0.016, M.orange).rotation.x = Math.PI / 2;
    for (let i = 0; i < 5; i++) book(-w * 0.3 + i * 0.1, h * 0.21, -d * 0.2, 0.07, 0.06, h * 0.23, [M.paper, M.darkWood, M.gray][i % 3]);
  } else if (k === "patternLamp") {
    const mat = M.paper.clone();
    if (typeof document !== "undefined") {
      const c = document.createElement("canvas");
      c.width = c.height = 512;
      let ctx = c.getContext("2d");
      ctx.fillStyle = "#d6bd8b";
      ctx.fillRect(0, 0, 512, 512);
      ctx.strokeStyle = "#302b22";
      ctx.lineWidth = 3;
      for (let i = 0; i < 85; i++) {
        let x = i * 127 % 512, y = i * 97 % 512;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.bezierCurveTo(x + 35, y - 40, x - 38, y + 30, x + 14, y + 52);
        ctx.stroke();
      }
      mat.map = new T.CanvasTexture(c);
      mat.map.colorSpace = T.SRGBColorSpace;
      mat.emissive = new T.Color("#bf9853");
      mat.emissiveIntensity = 0.18;
    }
    C(g, w * 0.21, w * 0.21, h * 0.53, 0, h * 0.275, 0, mat);
    C(g, w * 0.5, w * 0.5, h * 0.38, 0, h * 0.8, 0, mat, 64);
    C(g, w * 0.46, w * 0.46, 6e-3, 0, h * 0.61, 0, M.glow, 64);
    C(g, w * 0.22, w * 0.22, 0.01, 0, 5e-3, 0, M.black);
  } else if (k === "domeLamp") {
    C(g, w * 0.28, w * 0.28, 0.018, 0, 9e-3, 0, M.black);
    C(g, w * 0.13, w * 0.13, h * 0.48, 0, h * 0.25, 0, M.black);
    const m = M.cream.clone();
    m.emissive = new T.Color("#ffc968");
    m.emissiveIntensity = 0.7;
    lathe([[0, h], [w * 0.18, h * 0.97], [w * 0.35, h * 0.84], [w * 0.46, h * 0.66], [w * 0.5, h * 0.53], [w * 0.47, h * 0.53], [w * 0.43, h * 0.66], [w * 0.32, h * 0.82], [w * 0.16, h * 0.94], [0, h * 0.96]], m);
  } else if (k === "candle") {
    lathe([[0, 0], [w * 0.44, 0], [w * 0.5, 0.015], [w * 0.48, 0.035], [w * 0.35, 0.017], [w * 0.12, 0.02], [w * 0.09, h * 0.42], [w * 0.2, h * 0.46], [w * 0.2, h * 0.5], [w * 0.08, h * 0.5], [w * 0.08, h * 0.16], [0, h * 0.16]], M.chrome);
    C(g, 0.01, 0.01, h * 0.68, 0, h * 0.67, 0, M.paper, 24);
    R(g, [0, h, 0], [0, h + 6e-3, 0], 1e-3, M.black);
  } else if (k === "tray") {
    lathe([[0, 0], [w * 0.42, 0], [w * 0.49, h * 0.5], [w * 0.5, h], [w * 0.48, h], [w * 0.41, 5e-3], [0, 5e-3]], M.chrome);
  } else if (k === "mottledRug") {
    const mat = M.blue.clone();
    mat.color.set("#c1c8ce");
    mat.normalScale.set(0.06, 0.06);
    if (typeof document !== "undefined") {
      const c = document.createElement("canvas");
      c.width = c.height = 512;
      let ctx = c.getContext("2d");
      ctx.fillStyle = "#263449";
      ctx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 12e3; i++) {
        ctx.fillStyle = ["#8792a2", "#4a596c", "#303e52", "#b0a994"][i % 4];
        ctx.globalAlpha = 0.1;
        ctx.fillRect(i * 131 % 512, (i * 73 + Math.sin(i) * 30) % 512, 3 + i % 18, 1 + i % 9);
      }
      mat.map = new T.CanvasTexture(c);
      mat.map.colorSpace = T.SRGBColorSpace;
    }
    C(g, w * 0.5, w * 0.5, h, 0, h * 0.5, 0, mat, 96);
    ring(w * 0.495, 3e-3, 0, h, 0, M.gray);
  } else if (k === "darkBed") {
    B(g, w, 0.2, d, 0, h - 0.1, 0, M.darkWood, 9e-3);
    for (let x of [-1, 1]) for (let z of [-1, 1]) B(g, 0.13, h - 0.15, 0.13, x * (w * 0.5 - 0.18), (h - 0.15) / 2, z * (d * 0.5 - 0.18), M.darkWood, 8e-3);
    for (let z = -d * 0.43; z < d * 0.45; z += 0.14) B(g, w - 0.08, 0.018, 0.075, 0, h - 5e-3, z, M.wood, 2e-3);
  } else if (k === "greyBedding") {
    const cloth = M.gray.clone();
    cloth.color.set("#a4a5a2");
    cloth.roughness = 1;
    cloth.normalScale.set(0.025, 0.025);
    cloth.sheen = 0.25;
    cloth.side = T.DoubleSide;
    const geo = new T.PlaneGeometry(w, d * 0.82, 70, 76), a = geo.attributes.position;
    for (let i = 0; i < a.count; i++) {
      const x = a.getX(i), z = a.getY(i), edge = Math.max(0, Math.abs(x) - (w / 2 - 0.12)) / 0.12, foot = Math.max(0, -z / (d * 0.82) - 0.36) / 0.14;
      const wr = 0.012 * Math.sin(x * 18 + z * 6) + 8e-3 * Math.sin(z * 31 - x * 12) + 0.025 * Math.exp(-Math.pow((x - 0.21 + z * 0.12) * 8, 2)) * Math.sin(z * 8) + 0.018 * Math.sin(x * 9) * Math.sin(z * 5);
      a.setXYZ(i, x, h * 0.28 + wr - 0.16 * edge * edge - 0.14 * foot * foot, z - d * 0.07);
    }
    geo.computeVertexNormals();
    g.add(new T.Mesh(geo, cloth));
    for (let x of [-w * 0.25, w * 0.25]) {
      const b = B(g, w * 0.43, 0.17, 0.42, x, 0.15, d * 0.33, cloth, 0.07);
      b.rotation.y = x > 0 ? 0.05 : -0.07;
    }
  } else if (k === "bedCabinet") {
    B(g, w, 0.018, d, 0, 0.03, 0, M.darkWood, 3e-3);
    B(g, w, 0.025, d, 0, h - 0.012, 0, M.darkWood, 4e-3);
    for (let x of [-1, 1]) B(g, 0.025, h, d, x * (w / 2 - 0.013), h / 2, 0, M.darkWood, 3e-3);
    B(g, w, h, 0.015, 0, h / 2, d / 2 - 8e-3, M.darkWood, 2e-3);
    for (let x of [-1, 1]) {
      B(g, w * 0.47, h - 0.1, 0.018, x * w * 0.245, h * 0.48, -d / 2 + 0.02, x < 0 ? M.darkWood : M.cream, 3e-3);
      B(g, 0.06, 8e-3, 8e-3, x * w * 0.245, h * 0.54, -d / 2 + 7e-3, M.brushed, 2e-3);
    }
  } else if (k === "turntable") {
    B(g, w, h * 0.6, d, 0, h * 0.3, 0, M.wood, 7e-3);
    C(g, d * 0.4, d * 0.4, 0.012, -w * 0.09, h * 0.63, 0, M.black, 64);
    for (let i = 0; i < 9; i++) ring(d * (0.15 + i * 0.025), 7e-4, -w * 0.09, h * 0.64, 0, M.gray);
    C(g, d * 0.08, d * 0.08, 2e-3, -w * 0.09, h * 0.645, 0, M.red, 32);
    R(g, [w * 0.36, h * 0.65, d * 0.32], [w * 0.34, h * 0.7, -d * 0.18], 5e-3, M.brushed);
    B(g, 0.025, 0.018, 0.045, w * 0.3, h * 0.68, -d * 0.22, M.black, 2e-3);
  } else if (k === "squareClock") {
    B(g, w, h, d, 0, h / 2, 0, M.chrome, 0.018);
    const mat = M.blue.clone();
    if (typeof document !== "undefined") {
      let c = document.createElement("canvas");
      c.width = c.height = 512;
      let ctx = c.getContext("2d");
      ctx.fillStyle = "#4d7d9e";
      ctx.fillRect(0, 0, 512, 512);
      ctx.strokeStyle = "#a8b3a2";
      ctx.lineWidth = 2;
      for (let i = 0; i <= 12; i++) {
        ctx.beginPath();
        ctx.moveTo(i * 512 / 12, 0);
        ctx.lineTo(i * 512 / 12, 512);
        ctx.moveTo(0, i * 512 / 12);
        ctx.lineTo(512, i * 512 / 12);
        ctx.stroke();
      }
      ctx.fillStyle = "#e0e1de";
      ctx.font = "44px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let n = 1; n <= 12; n++) {
        const a = n * Math.PI / 6;
        ctx.fillText(n, 256 + Math.sin(a) * 205, 256 - Math.cos(a) * 205);
      }
      mat.map = new T.CanvasTexture(c);
      mat.map.colorSpace = T.SRGBColorSpace;
    }
    B(g, w - 0.026, h - 0.026, 2e-3, 0, h / 2, -d / 2 - 2e-3, M.brushed, 4e-3);
    const face = C(g, w * 0.4, w * 0.4, 3e-3, 0, h / 2, -d / 2 - 4e-3, mat, 64);
    face.rotation.x = Math.PI / 2;
    R(g, [0, h / 2, -d / 2 - 9e-3], [-w * 0.18, h * 0.62, -d / 2 - 9e-3], 4e-3, M.paper);
    R(g, [0, h / 2, -d / 2 - 0.011], [w * 0.25, h * 0.48, -d / 2 - 0.011], 2e-3, M.paper);
  } else if (k === "donut") {
    const mat = M.amber.clone();
    mat.roughness = 0.18;
    mat.emissive = new T.Color("#b4450c");
    mat.emissiveIntensity = 0.4;
    ring(w * 0.29, w * 0.2, 0, h / 2, 0, mat, "z");
    B(g, w * 0.37, h * 0.36, d * 0.4, 0, h * 0.32, d * 0.2, M.orange, 0.02);
  } else if (k === "lantern") {
    const cage = M.cream.clone();
    cage.color.set("#a99f68");
    for (let y of [0.06, h * 0.29, h * 0.53, h * 0.76]) {
      const r = y === 0.06 ? w * 0.25 : y === h * 0.76 ? w * 0.32 : w * 0.44;
      ring(r, 6e-3, 0, y, 0, cage);
    }
    for (let i = 0; i < 24; i++) {
      let a = i * Math.PI / 12;
      const pts = [[w * 0.25, 0.06], [w * 0.43, h * 0.26], [w * 0.45, h * 0.48], [w * 0.32, h * 0.77]].map(([r, y]) => [Math.cos(a) * r, y, Math.sin(a) * r]);
      tube(pts, 45e-4, cage);
    }
    for (let a of [0, Math.PI * 0.66, Math.PI * 1.33]) R(g, [Math.cos(a) * w * 0.28, 0, Math.sin(a) * w * 0.28], [Math.cos(a) * w * 0.28, h * 0.18, Math.sin(a) * w * 0.28], 6e-3, cage);
    ring(w * 0.32, 0.012, 0, h * 0.78, 0, cage);
    ring(w * 0.32, 8e-3, 0, h * 0.84, 0, cage);
    C(g, w * 0.1, w * 0.1, h * 0.42, 0, h * 0.38, 0, M.glow);
    tube([[-w * 0.34, h * 0.8, 0], [-w * 0.34, h * 0.95, 0], [0, h, 0], [w * 0.34, h * 0.95, 0], [w * 0.34, h * 0.8, 0]], 7e-3, cage);
  } else return false;
  return true;
}
export {
  photoModel
};
