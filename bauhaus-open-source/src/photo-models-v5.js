import * as T from "three";
function photoV5(p, g, M, H) {
  const { box: B, cylinder: C, ball: S, rod: R } = H, [w, d, h] = p.dims;
  const tube = (pts, r, m) => {
    const o = new T.Mesh(new T.TubeGeometry(new T.CatmullRomCurve3(pts.map((a) => new T.Vector3(...a))), 48, r, 10, false), m);
    g.add(o);
    return o;
  };
  const leather = (color) => new T.MeshPhysicalMaterial({ color, roughness: 0.32, clearcoat: 0.28, clearcoatRoughness: 0.38, sheen: 0.12, sheenRoughness: 0.8 });
  const black = leather("#121312");
  const pipe = (ww, dd, y, x, z, mat) => tube([[-ww / 2, y, -dd / 2], [-ww / 2, y, dd / 2], [ww / 2, y, dd / 2], [ww / 2, y, -dd / 2]].map((a) => [a[0] + x, a[1], a[2] + z]), 2e-3, mat);
  if (p.kind === "sofa") {
    const id = p.sourceKey || "", is = (key) => id === key;
    let mat = M.gray.clone(), arm = 0.1, leg = 0.12, seat = 0.39, backN = 2, armH = 0.31, armTop = 0.58, backH = 0.34;
    mat.color.set("#303634");
    if (is("N007_8")) {
      mat = black;
      backN = 1;
      arm = 0.13;
    }
    if (is("N007_4") || is("N007_5")) mat = leather("#74412b");
    if (is("N007_9")) mat = leather("#a74c20");
    if (is("N007_11") || is("N007_15")) mat = leather("#c9c4b5");
    if (is("N007_7")) {
      mat = M.linen.clone();
      mat.color.set("#828884");
      arm = 0.13;
      leg = 0.1;
    }
    if (is("N007_1")) {
      arm = 0.055;
      leg = 0.22;
      seat = 0.35;
      armH = 0.25;
      armTop = 0.55;
    }
    if (is("N007_2")) {
      arm = 0.105;
      leg = 0.14;
      backN = 1;
      armH = 0.46;
      armTop = 0.66;
    }
    if (is("N007_3")) {
      arm = 0.075;
      leg = 0.11;
      backN = 1;
      armH = 0.29;
      armTop = 0.54;
    }
    if (is("N007_4")) {
      arm = 0;
      leg = 0.21;
      backN = 1;
    }
    if (is("N007_5")) {
      arm = 0.105;
      leg = 0.17;
      backN = 3;
      armH = 0.43;
      armTop = 0.7;
    }
    if (is("N007_11")) {
      arm = 0.15;
      armH = 0.4;
      armTop = 0.65;
      backN = 1;
    }
    for (let x of [-1, 1]) for (let z of [-1, 1]) {
      const xx = x * (w / 2 - 0.085), zz = z * (d / 2 - 0.075);
      if (is("N007_2")) R(g, [xx + x * 0.025, 0.012, zz + z * 0.025], [xx, leg, zz], 0.021, M.wood);
      else if (is("N007_4") || is("N007_5")) B(g, 0.045, leg, 0.045, xx, leg / 2, zz, M.wood, 4e-3);
      else {
        R(g, [xx, 0.01, zz], [xx - x * 0.012, leg, zz - z * 8e-3], is("N007_1") ? 0.011 : 0.017, M.black);
        if (is("N007_9") || is("N007_15")) C(g, 0.018, 0.018, 0.035, xx, 0.024, zz, M.brushed, 16);
      }
    }
    const inner = w - 2 * arm - 0.025, sh = 0.14, sy = Math.max(leg + 0.15, seat);
    const baseH = sy - 0.045 - leg;
    B(g, w - 0.03, baseH, d - 0.03, 0, leg + baseH / 2, 0, is("N007_4") ? M.wood : mat, 0.025);
    B(g, inner, sh, d - 0.16, 0, sy, -0.045, mat, 0.045);
    if (backN === 2) R(g, [0, sy + sh / 2, -d * 0.47], [0, sy + sh / 2, d * 0.23], 16e-4, M.black);
    const by = h - backH / 2 - 0.015, bz = d * 0.35;
    for (let i = 0; i < backN; i++) {
      const bw = (inner - 0.012 * (backN - 1)) / backN, x = (i - (backN - 1) / 2) * (bw + 0.012);
      const tuft = is("N007_8") || is("N007_4") || is("N007_11");
      if (!tuft) {
        const b = B(g, bw, backH, 0.17, x, by, bz, mat, 0.06);
        b.rotation.x = -0.13;
      }
    }
    if (arm) for (let x of [-1, 1]) {
      const ar = B(g, arm, armH, d - 0.025, x * (w / 2 - arm / 2), armTop - armH / 2, 0, mat, is("N007_1") ? 0.016 : 0.04);
      if (is("N007_1")) {
        const a = ar.geometry.attributes.position;
        for (let i = 0; i < a.count; i++) a.setY(i, a.getY(i) + 0.1 * (a.getZ(i) / d) * Math.max(0, a.getY(i) / armH + 0.5));
        ar.geometry.computeVertexNormals();
      }
    }
    if (is("N007_8") || is("N007_4") || is("N007_11")) {
      const cols = is("N007_11") ? 3 : 4, rows = is("N007_8") ? 1 : 2;
      const tuftPoints = [];
      for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) tuftPoints.push([(i - (cols - 1) / 2) * inner / (cols + 0.5), (j - (rows - 1) / 2) * 0.15]);
      const quilt = new T.BoxGeometry(inner, backH, 0.17, 64, 24, 8), qa = quilt.attributes.position, qn = quilt.attributes.normal;
      for (let i = 0; i < qa.count; i++) {
        const v = new T.Vector3().fromBufferAttribute(qa, i), q = new T.Vector3(T.MathUtils.clamp(v.x, -inner / 2 + 0.045, inner / 2 - 0.045), T.MathUtils.clamp(v.y, -backH / 2 + 0.045, backH / 2 - 0.045), T.MathUtils.clamp(v.z, -0.04, 0.04));
        v.sub(q).normalize().multiplyScalar(0.045).add(q);
        qa.setXYZ(i, v.x, v.y, v.z);
      }
      for (let i = 0; i < qa.count; i++) {
        const x = qa.getX(i), y = qa.getY(i), f = Math.max(0, -qn.getZ(i));
        let z = qa.getZ(i) - f * 0.012 * Math.sin((x / inner + 0.5) * Math.PI) * Math.cos(y / backH * Math.PI);
        for (const [tx, ty] of tuftPoints) {
          const dx = x - tx, dy = y - ty;
          z += f * (0.04 * Math.exp(-(dx * dx + dy * dy) / 13e-4) + 8e-3 * Math.exp(-Math.pow(Math.abs(dx) - Math.abs(dy), 2) / 12e-5) * Math.exp(-(dx * dx + dy * dy) / 0.018));
        }
        qa.setXYZ(i, x, y + by, z + bz);
      }
      quilt.computeVertexNormals();
      g.add(new T.Mesh(quilt, mat));
      for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
        const x = (i - (cols - 1) / 2) * inner / (cols + 0.5), y = by + (j - (rows - 1) / 2) * 0.15;
        S(g, x, y, bz - 0.056, 8e-3, 8e-3, 3e-3, mat);
      }
      for (let i = 1; i < 6; i++) {
        const x = -inner / 2 + i * inner / 6;
        R(g, [x, sy + 0.072, -d * 0.42], [x, sy + 0.072, d * 0.15], 15e-4, mat);
      }
    }
    if (is("N007_9") || is("N007_15")) for (let x of [-1, 1]) {
      const q = B(g, 0.25, 0.25, 0.11, x * (inner / 2 - 0.11), sy + 0.18, d * 0.13, mat, 0.04);
      q.rotation.z = x * 0.18;
      q.rotation.x = -0.18;
    }
    return true;
  }
  if (p.id === "B_shoe_0") {
    B(g, w, h, 0.016, 0, h / 2, d / 2 - 8e-3, M.white, 2e-3);
    for (let x of [-1, 1]) B(g, 0.018, h, d, x * (w / 2 - 9e-3), h / 2, 0, M.white, 2e-3);
    for (let y of [0.012, h - 0.012]) B(g, w, 0.024, d, 0, y, 0, M.white, 2e-3);
    for (let j = 0; j < 2; j++) {
      let yy = 0.04 + j * 0.445;
      B(g, w - 0.042, 0.405, 0.025, 0, yy + 0.2025, -d / 2 + 0.012, M.white, 2e-3);
      B(g, w - 0.06, 0.013, 0.028, 0, yy + 0.421, -d / 2 + 0.035, M.gray, 1e-3);
    }
    return true;
  }
  if (!p.modelV5) return false;
  if (p.modelV5 === "cubeChair") {
    B(g, w - 0.08, 0.12, d - 0.06, 0, 0.24, 0, black, 0.024);
    B(g, w - 0.29, 0.19, d - 0.14, 0, 0.38, -0.025, black, 0.045);
    for (let x of [-1, 1]) {
      B(g, 0.14, 0.39, d - 0.07, x * (w / 2 - 0.105), 0.49, 0, black, 0.035);
      for (let z of [-1, 1]) R(g, [x * (w / 2 - 0.024), 0.012, z * (d / 2 - 0.045)], [x * (w / 2 - 0.024), 0.56, z * (d / 2 - 0.045)], 0.012, M.chrome);
    }
    B(g, w - 0.26, 0.31, 0.16, 0, h - 0.165, d / 2 - 0.105, black, 0.045);
    for (let y of [0.3, 0.61]) tube([[-w / 2 + 0.025, y, -d / 2 + 0.045], [-w / 2 + 0.025, y, d / 2 - 0.025], [w / 2 - 0.025, y, d / 2 - 0.025], [w / 2 - 0.025, y, -d / 2 + 0.045]], 0.012, M.chrome);
    pipe(w - 0.29, d - 0.14, 0.48, 0, -0.025, black);
  } else if (p.modelV5 === "tub") {
    const white = new T.MeshPhysicalMaterial({ color: "#f0f0e8", roughness: 0.19, clearcoat: 0.5 });
    const pts = [], idx = [], n = 96, rings = [[w * 0.94, d * 0.94, 0.012], [w, d, h - 0.025], [w - 0.02, d - 0.02, h], [w - 0.066, d - 0.066, h - 0.015], [w - 0.13, d - 0.22, 0.1], [0, 0, 0.1]];
    for (const [a, b, y] of rings) for (let i = 0; i < n; i++) {
      const t = i / n * Math.PI * 2, c = Math.cos(t), s = Math.sin(t);
      pts.push(Math.sign(c) * Math.pow(Math.abs(c), 0.32) * a / 2, y, Math.sign(s) * Math.pow(Math.abs(s), 0.32) * b / 2);
    }
    for (let j = 0; j < rings.length - 1; j++) for (let i = 0; i < n; i++) {
      const a = j * n + i, b = j * n + (i + 1) % n, c = b + n, e = a + n;
      idx.push(a, e, b, b, e, c);
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.Float32BufferAttribute(pts, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    const o = new T.Mesh(geo, white);
    white.side = T.DoubleSide;
    o.castShadow = o.receiveShadow = true;
    g.add(o);
    C(g, 0.022, 0.022, 3e-3, 0, 0.104, 0, M.chrome, 32);
    B(g, 0.09, 0.018, 4e-3, 0, h - 0.13, d / 2 - 0.047, M.chrome, 2e-3);
  } else if (p.modelV5 === "railShelf") {
    const step = w / 3;
    for (let k = 0; k < 4; k++) {
      const x = -w / 2 + 0.018 + k * (w - 0.036) / 3;
      B(g, 0.023, h, 0.018, x, h / 2, d / 2 - 9e-3, M.brushed, 2e-3);
      for (let y = 0.04; y < h - 0.02; y += 0.035) B(g, 7e-3, 0.018, 3e-3, x, y, d / 2 - 0.02, M.black, 1e-3);
    }
    for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) {
      const x = (col - 1) * step, y = 0.025 + row * 0.36, sw = step - 0.035;
      B(g, sw, 9e-3, d - 0.025, x, y, -0.01, M.brushed, 2e-3);
      B(g, sw, row === 2 ? 0.2 : 0.055, 8e-3, x, y + (row === 2 ? 0.1 : 0.027), d / 2 - 0.03, M.brushed, 2e-3);
      for (let s of [-1, 1]) B(g, 8e-3, 0.04, d - 0.025, x + s * (sw / 2 - 4e-3), y + 0.02, -0.01, M.brushed, 1e-3);
      if ((row + col) % 2 === 0) for (let k = 0; k < 4; k++) B(g, 0.022, 0.15 + k % 2 * 0.035, 0.12, x - sw * 0.29 + k * 0.027, y + 0.085, 0, [M.paper, M.blue, M.darkWood][k % 3], 1e-3);
      else if (col === 2) {
        C(g, 0.032, 0.022, 0.13, x, y + 0.07, 0, M.black, 24);
      } else {
        B(g, 0.19, 0.025, 0.12, x, y + 0.02, 0, M.paper, 2e-3);
        C(g, 0.035, 0.025, 0.08, x, y + 0.074, 0, M.earth, 32);
      }
    }
  } else if (p.modelV5 === "screen") {
    const col = 3, row = 12, cw = w / col, ch = h / row;
    const cream = new T.MeshStandardMaterial({ color: "#d8d3c5", roughness: 0.85 });
    for (let i = 0; i < col; i++) for (let j = 0; j < row; j++) {
      const x = -w / 2 + cw * (i + 0.5), y = ch * (j + 0.5), outerW = cw - 3e-3, outerH = ch - 3e-3, iw = cw * 0.27, ih = ch * 0.65;
      const shape = new T.Shape();
      shape.moveTo(-outerW / 2, -outerH / 2);
      shape.lineTo(outerW / 2, -outerH / 2);
      shape.lineTo(outerW / 2, outerH / 2);
      shape.lineTo(-outerW / 2, outerH / 2);
      shape.closePath();
      const hole = new T.Path();
      hole.moveTo(-iw / 2, -ih / 2);
      hole.lineTo(-iw / 2, ih / 2);
      hole.lineTo(iw / 2, ih / 2);
      hole.lineTo(iw / 2, -ih / 2);
      hole.closePath();
      shape.holes.push(hole);
      const o = new T.Mesh(new T.ExtrudeGeometry(shape, { depth: d - 0.012, steps: 1, bevelEnabled: true, bevelSize: 6e-3, bevelThickness: 6e-3, bevelSegments: 1 }), cream);
      o.position.set(x, y, -d / 2 + 6e-3);
      g.add(o);
    }
  } else if (p.modelV5 === "starTable") {
    const glass = new T.MeshPhysicalMaterial({ color: "#d7e6e3", roughness: 0.08, metalness: 0.05, transparent: true, opacity: 0.22, depthWrite: false });
    C(g, w / 2, w / 2, 9e-3, 0, h - 5e-3, 0, glass, 80);
    for (let i = 0; i < 3; i++) {
      const sh = new T.Shape();
      sh.moveTo(-0.075, h - 0.015);
      sh.lineTo(0.27, 7e-3);
      sh.lineTo(0.075, h - 0.015);
      sh.closePath();
      const o = new T.Mesh(new T.ExtrudeGeometry(sh, { depth: 6e-3, bevelEnabled: true, bevelSize: 1e-3, bevelThickness: 1e-3, bevelSegments: 1, steps: 1 }), M.brushed);
      o.rotation.y = i * Math.PI * 2 / 3;
      o.position.z = -3e-3;
      g.add(o);
    }
  } else if (p.modelV5 === "plaidBedding") {
    const mat = M.cream.clone();
    mat.color.set("#ffffff");
    mat.side = T.DoubleSide;
    mat.normalScale?.set(0.02, 0.02);
    if (typeof document !== "undefined") {
      const c = document.createElement("canvas");
      c.width = c.height = 512;
      const q = c.getContext("2d");
      q.fillStyle = "#c8c6ac";
      q.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 2; i++) {
        q.fillStyle = "#6f8180";
        q.fillRect(i * 256, 0, 100, 512);
        q.fillRect(0, i * 256, 512, 100);
        q.fillStyle = "#435d64";
        q.fillRect(i * 256, 0, 100, 100);
        q.fillRect(i * 256, 256, 100, 100);
        q.fillStyle = "#e4dfcd";
        q.fillRect(i * 256 + 116, 0, 24, 512);
        q.fillRect(0, i * 256 + 116, 512, 24);
        q.fillStyle = "#a78059";
        q.fillRect(i * 256 + 165, 0, 5, 512);
        q.fillRect(0, i * 256 + 165, 512, 5);
      }
      const t = new T.CanvasTexture(c);
      t.colorSpace = T.SRGBColorSpace;
      t.wrapS = t.wrapT = T.RepeatWrapping;
      t.repeat.set(2, 3.5);
      mat.map = t;
    }
    const geo = new T.PlaneGeometry(w, d * 0.81, 40, 64), a = geo.attributes.position;
    for (let i = 0; i < a.count; i++) {
      const x = a.getX(i), z = a.getY(i) - d * 0.095, edge = Math.max(0, (Math.abs(x) - w * 0.43) / (w * 0.07));
      a.setXYZ(i, x, 0.13 - 0.12 * edge * edge + 0.012 * Math.sin(x * 28 + z * 14) + 6e-3 * Math.cos(z * 31), z);
    }
    geo.computeVertexNormals();
    g.add(new T.Mesh(geo, mat));
    const pillow = B(g, w * 0.75, 0.14, 0.36, 0, 0.2, d * 0.36, mat, 0.06);
    pillow.rotation.y = -0.025;
    const fold = M.gray.clone();
    fold.color.set("#a7ada6");
    fold.side = T.DoubleSide;
    const fg = new T.PlaneGeometry(w * 0.85, 0.22, 30, 8), fa = fg.attributes.position;
    for (let i = 0; i < fa.count; i++) {
      const x = fa.getX(i), z = fa.getY(i);
      fa.setXYZ(i, x, 0.151 + 8e-3 * Math.sin(x * 28 + z * 14) + 0.014 * Math.sin((z + 0.11) / 0.22 * Math.PI), z + d * 0.18);
    }
    fg.computeVertexNormals();
    g.add(new T.Mesh(fg, fold));
  } else return false;
  return true;
}
export {
  photoV5
};
