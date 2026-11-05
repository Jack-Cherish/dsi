import * as T from "three";
function hero(p, g, M, H) {
  const { box: B, cylinder: C, ball: S, rod: R } = H;
  const [w, d, h] = p.dims, n = p.name;
  const lathe = (pts, m) => {
    const o = new T.Mesh(new T.LatheGeometry(pts.map((a) => new T.Vector2(...a)), 64), m);
    g.add(o);
    return o;
  };
  const tube = (pts, r, m) => {
    const curve = new T.CatmullRomCurve3(pts.map((a) => new T.Vector3(...a)));
    const o = new T.Mesh(new T.TubeGeometry(curve, 40, r, 10, false), m);
    g.add(o);
    return o;
  };
  if (p.kind === "steelTable") {
    if (p.sourceKey === "J05_2") {
      for (let y of [0.04, h - 0.014]) B(g, w, 0.016, d, 0, y, 0, M.black, 3e-3);
      for (let x of [-1, 1]) for (let z of [-1, 1]) {
        R(g, [x * w * 0.48, 0.02, z * d * 0.47], [x * w * 0.48, h, z * d * 0.47], 9e-3, M.chrome);
        S(g, x * w * 0.48, h, z * d * 0.47, 0.014, 0.014, 0.014, M.chrome);
      }
      for (let y of [0.055, h - 0.02]) {
        for (let x of [-1, 1]) R(g, [x * w * 0.48, y, -d * 0.47], [x * w * 0.48, y, d * 0.47], 9e-3, M.chrome);
        for (let z of [-1, 1]) R(g, [-w * 0.48, y, z * d * 0.47], [w * 0.48, y, z * d * 0.47], 9e-3, M.chrome);
      }
    } else if (p.sourceKey === "J05_1") {
      B(g, w - 0.035, 0.014, d - 0.035, 0, h - 0.022, 0, M.chrome, 7e-3);
      for (let z of [-d * 0.47, d * 0.47]) tube([[-w * 0.43, h - 0.02, z], [-w * 0.47, h - 0.06, z], [-w * 0.47, 0.06, z], [-w * 0.4, 0.018, z], [w * 0.4, 0.018, z], [w * 0.47, 0.06, z], [w * 0.47, h - 0.06, z], [w * 0.43, h - 0.02, z]], 0.012, M.red);
      for (let x of [-w * 0.45, w * 0.45]) R(g, [x, h - 0.02, -d * 0.47], [x, h - 0.02, d * 0.47], 0.012, M.red);
    } else {
      C(g, w * 0.49, w * 0.49, 0.012, 0, h - 6e-3, 0, M.black, 64);
      C(g, w * 0.37, w * 0.37, 0.016, 0, 8e-3, 0, M.chrome, 64);
      const sh = new T.Shape();
      sh.moveTo(-w * 0.26, h - 0.02);
      sh.bezierCurveTo(-w * 0.46, h * 0.6, w * 0.28, h * 0.65, w * 0.26, h * 0.4);
      sh.bezierCurveTo(w * 0.25, h * 0.2, -w * 0.2, h * 0.24, -w * 0.2, 0.02);
      sh.lineTo(w * 0.09, 0.02);
      sh.bezierCurveTo(w * 0.5, h * 0.31, -w * 0.27, h * 0.4, -w * 0.24, h * 0.6);
      sh.bezierCurveTo(-w * 0.22, h * 0.79, w * 0.24, h * 0.74, w * 0.24, h - 0.02);
      sh.closePath();
      const mesh = new T.Mesh(new T.ExtrudeGeometry(sh, { depth: 6e-3, bevelEnabled: true, bevelSize: 1e-3, bevelThickness: 1e-3, bevelSegments: 2, steps: 1, curveSegments: 32 }), M.chrome);
      mesh.position.z = -3e-3;
      g.add(mesh);
    }
  } else if (p.kind === "roundrug") {
    C(g, w / 2, w / 2, h, 0, h / 2, 0, p.sourceKey === "J06_1" ? M.blue : p.sourceKey === "J06_2" ? M.cream : M.black, 96);
  } else if (/GLOSTAD/.test(n)) {
    for (let x of [-1, 1]) for (let z of [-1, 1]) R(g, [x * (w / 2 - 0.08), 0.01, z * (d / 2 - 0.06)], [x * (w / 2 - 0.12), 0.23, z * (d / 2 - 0.1)], 0.013, M.black);
    B(g, w - 0.04, 0.15, d - 0.05, 0, 0.27, 0, M.gray, 0.025);
    B(g, w - 0.16, 0.13, d - 0.13, 0, 0.39, -0.03, M.gray, 0.045);
    for (let x of [-1, 1]) {
      B(g, 0.075, 0.32, d - 0.035, x * (w / 2 - 0.037), 0.4, 0, M.gray, 0.024);
      const back = B(g, (w - 0.18) / 2, 0.33, 0.14, x * (w - 0.18) / 4, h - 0.18, d * 0.34, M.gray, 0.045);
      back.rotation.x = -0.12;
    }
  } else if (p.sourceKey === "N048_4") {
    for (let x of [-1, 1]) for (let z of [-1, 1]) R(g, [x * w * 0.44, 0.01, z * d * 0.4], [x * w * 0.4, 0.15, z * d * 0.35], 0.019, M.wood);
    B(g, w, 0.025, d, 0, 0.15, 0, M.wood, 3e-3);
    for (let x of [-1, 1]) B(g, 0.021, h - 0.15, d, x * (w / 2 - 0.01), (h + 0.15) / 2, 0, M.wood, 3e-3);
    B(g, w, 0.022, d, 0, h - 0.011, 0, M.wood, 3e-3);
    B(g, w, h - 0.15, 0.012, 0, (h + 0.15) / 2, d / 2 - 6e-3, M.wood, 2e-3);
    for (let j = 0; j < 3; j++) {
      const y = h - 0.085 - j * 0.14;
      B(g, w - 0.045, 0.132, d - 0.03, 0, y, 0, M.wood, 3e-3);
      C(g, 0.011, 0.011, 0.012, 0, y, -d / 2 - 5e-3, M.darkWood, 16).rotation.x = Math.PI / 2;
    }
    B(g, w - 0.04, 0.11, d - 0.02, 0, 0.22, 0, M.wood, 3e-3);
    B(g, w - 0.04, 0.018, d - 0.01, 0, 0.31, 0, M.wood, 3e-3);
  } else if (p.sourceKey === "N045_6") {
    for (let x of [-1, 1]) B(g, 0.018, h, d, x * (w / 2 - 9e-3), h / 2, 0, M.wood, 2e-3);
    B(g, w, h, 0.012, 0, h / 2, d / 2 - 6e-3, M.white, 1e-3);
    for (let j = 0; j < 6; j++) {
      const y = 0.025 + j * (h - 0.04) / 5;
      B(g, w, 0.018, d, 0, y, 0, M.wood, 2e-3);
      if (j < 5) for (let i = 0; i < 6; i++) B(g, 0.035, 0.16 + i % 3 * 0.025, d * 0.62, -w * 0.4 + i * 0.075, y + 0.1, 0, [M.paper, M.blue, M.darkWood][i % 3], 1e-3);
    }
  } else if (p.sourceKey === "N039_7") {
    for (let x of [-1, 1]) for (let z of [-1, 1]) C(g, 0.032, 0.032, 0.3, x * (w / 2 - 0.05), 0.15, z * (d / 2 - 0.06), M.wood, 24);
    B(g, w, 0.13, d, 0, 0.245, 0, M.wood, 8e-3);
    B(g, w - 0.03, 0.22, 0.035, 0, h - 0.11, d / 2 - 0.02, M.wood, 8e-3);
    for (let x of [-1, 1]) B(g, 0.045, h, 0.045, x * (w / 2 - 0.025), h / 2, d / 2 - 0.025, M.wood, 6e-3);
    B(g, w - 0.05, 0.12, d - 0.1, 0, 0.37, -0.025, M.cream, 0.045);
    if (!p.bareV5) {
      B(g, w - 0.08, 0.07, d * 0.65, 0, 0.46, -d * 0.16, M.linen, 0.025);
      B(g, w * 0.72, 0.1, 0.35, 0, 0.48, d * 0.32, M.cream, 0.045);
    }
  } else if (p.sourceKey === "N030_2") {
    const mat = M.cream.clone();
    if (typeof document !== "undefined") {
      const c = document.createElement("canvas");
      c.width = c.height = 256;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#e8e6dd";
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = "#30302e";
      for (let i = 0; i < 110; i++) {
        const x = i * 79 % 256, y = i * 113 % 256;
        ctx.beginPath();
        ctx.ellipse(x, y, 2 + i % 3, 2 + i % 2, i, 0, 7);
        ctx.fill();
      }
      const t = new T.CanvasTexture(c);
      t.colorSpace = T.SRGBColorSpace;
      t.wrapS = t.wrapT = T.RepeatWrapping;
      t.repeat.set(3, 4);
      mat.map = t;
    }
    const geo = new T.PlaneGeometry(w, d * 0.74, 45, 60), a = geo.attributes.position;
    for (let i = 0; i < a.count; i++) {
      let x = a.getX(i), z = a.getY(i);
      a.setXYZ(i, x, h * 0.28 + 9e-3 * Math.sin(x * 25 + z * 16) + 6e-3 * Math.cos(z * 48), z - d * 0.12);
    }
    geo.computeVertexNormals();
    mat.side = T.DoubleSide;
    g.add(new T.Mesh(geo, mat));
    for (let x of [-w * 0.24, w * 0.24]) B(g, w * 0.42, h * 0.6, 0.38, x, h * 0.65, d * 0.36, mat, 0.035);
  } else if (p.kind === "towelrail" && /TISKEN/.test(n)) {
    for (let x of [-w * 0.43, w * 0.43]) {
      C(g, 0.035, 0.035, 0.018, x, h * 0.5, d * 0.35, M.white, 32).rotation.x = Math.PI / 2;
      R(g, [x, h * 0.5, d * 0.35], [x, h * 0.5, -d * 0.3], 0.012, M.white);
    }
    R(g, [-w * 0.43, h * 0.5, -d * 0.3], [w * 0.43, h * 0.5, -d * 0.3], 0.011, M.white);
  } else if (/何佳功.*沙发/.test(n)) {
    for (let x of [-1, 1]) for (let z of [-1, 1]) C(g, 0.018, 0.024, 0.13, x * (w / 2 - 0.1), 0.065, z * (d / 2 - 0.09), M.brushed, 16);
    B(g, w - 0.04, 0.2, d - 0.05, 0, 0.23, 0, M.orange, 0.045);
    B(g, w - 0.21, 0.14, d - 0.08, 0, 0.38, -0.03, M.orange, 0.06);
    for (let x of [-1, 1]) {
      B(g, 0.105, 0.34, d, x * (w / 2 - 0.055), 0.4, 0, M.orange, 0.045);
      const b = B(g, (w - 0.24) / 2, 0.39, 0.15, x * (w - 0.24) / 4, h - 0.22, d * 0.32, M.orange, 0.052);
      b.rotation.x = -0.15;
    }
  } else if (/SANDSBERG.*餐椅/.test(n)) {
    const m = M.red, sy = h * 0.58;
    B(g, w * 0.94, 0.035, d * 0.78, 0, sy, -d * 0.06, m, 0.025);
    for (let x of [-1, 1]) for (let z of [-1, 1]) R(g, [x * w * 0.43, 0.015, z * d * 0.43], [x * w * 0.4, sy, z * d * 0.32], 0.011, m);
    for (let x of [-1, 1]) R(g, [x * w * 0.4, sy, d * 0.32], [x * w * 0.4, h - 0.02, d * 0.4], 0.013, m);
    B(g, w * 0.9, 0.037, 0.035, 0, h - 0.02, d * 0.4, m, 0.012);
    B(g, w * 0.9, 0.035, 0.035, 0, sy + 0.1, d * 0.37, m, 9e-3);
    for (let j = 0; j < 4; j++) B(g, w * 0.84, 0.013, 0.027, 0, sy + 0.13 + j * 0.042, d * 0.385, m, 4e-3);
    for (let i = 0; i < 8; i++) B(g, 0.013, h - sy - 0.12, 0.027, -w * 0.35 + i * w * 0.1, (h + sy + 0.08) / 2, d * 0.385, m, 4e-3);
  } else if (/莫比乌斯椅/.test(n)) {
    B(g, w, 0.035, d, 0, h * 0.58, 0, M.red, 3e-3);
    for (let x of [-1, 1]) {
      B(g, 0.035, h * 0.58, 0.035, x * w * 0.46, h * 0.29, -d * 0.42, M.red, 2e-3);
      B(g, 0.035, h, 0.035, x * w * 0.46, h / 2, d * 0.42, M.red, 2e-3);
    }
    B(g, w, 0.035, 0.035, 0, h - 0.017, d * 0.42, M.red, 2e-3);
  } else if (/SUNDS.*桌/.test(n)) {
    B(g, w, 0.018, d, 0, h - 9e-3, 0, M.blue, 0.025);
    for (let z of [-d * 0.38, d * 0.38]) {
      R(g, [-w * 0.4, 0.02, z], [w * 0.36, h - 0.02, z], 9e-3, M.blue);
      R(g, [w * 0.4, 0.02, z], [-w * 0.36, h - 0.02, z], 9e-3, M.blue);
      R(g, [-w * 0.4, 0.04, z], [w * 0.4, 0.04, z], 7e-3, M.blue);
    }
    for (let x of [-w * 0.36, w * 0.36]) R(g, [x, h * 0.7, -d * 0.38], [x, h * 0.7, d * 0.38], 6e-3, M.blue);
  } else if (/NYM.*吊灯/.test(n)) {
    const shadeH = h * 0.43, base = h * 0.38;
    lathe([[w * 0.02, base + shadeH], [w * 0.2, base + shadeH * 0.96], [w * 0.4, base + shadeH * 0.65], [w * 0.49, base + shadeH * 0.15], [w * 0.49, base], [w * 0.47, base], [w * 0.45, base + shadeH * 0.14], [w * 0.37, base + shadeH * 0.6], [w * 0.18, base + shadeH * 0.89], [w * 0.02, base + shadeH * 0.9]], M.white);
    R(g, [0, base + shadeH, 0], [0, h, 0], 3e-3, M.white);
    C(g, 0.045, 0.045, 0.025, 0, h - 0.012, 0, M.white);
    C(g, w * 0.43, w * 0.43, 4e-3, 0, base + 0.01, 0, M.glow, 64);
  } else if (/ISJAKT/.test(n)) {
    C(g, w * 0.46, w * 0.46, 0.018, 0, 9e-3, 0, M.brushed);
    R(g, [0, 0.02, 0], [0, h - 0.015, 0], 9e-3, M.brushed);
    C(g, w * 0.35, w * 0.35, 0.015, 0, h - 0.012, 0, M.brushed, 64);
    C(g, w * 0.3, w * 0.3, 4e-3, 0, h - 2e-3, 0, M.glow, 48);
    tube([[0, h * 0.62, 0], [0, h * 0.7, 0], [w * 0.3, h * 0.74, 0]], 9e-3, M.brushed);
    C(g, 0.06, 0.06, 0.012, w * 0.3, h * 0.74, 0, M.brushed);
    C(g, 0.052, 0.052, 3e-3, w * 0.3, h * 0.73, 0, M.glow);
  } else if (/FORSÅ|TERTIAL/.test(n)) {
    C(g, w * 0.21, w * 0.24, 0.025, -w * 0.25, 0.014, 0, M.chrome);
    const A = [-w * 0.25, 0.06, 0], B1 = [w * 0.2, h * 0.5, 0], D = [-w * 0.12, h * 0.9, 0];
    for (let z of [-0.018, 0.018]) {
      R(g, [A[0], A[1], z], [B1[0], B1[1], z], 7e-3, M.chrome);
      R(g, [B1[0], B1[1], z], [D[0], D[1], z], 7e-3, M.chrome);
    }
    for (const pt of [A, B1, D]) S(g, ...pt, 0.021, 0.021, 0.021, M.chrome);
    for (let j = 0; j < 9; j++) C(g, 9e-3, 9e-3, 4e-3, -w * 0.16 + j * 0.012, 0.13 + j * 0.01, 0.024, M.black, 12);
    const shade = lathe([[0.015, 0.12], [w * 0.12, 0.1], [w * 0.23, 0.015], [w * 0.23, 0], [w * 0.21, 0], [w * 0.1, 0.08], [0.015, 0.1]], M.chrome);
    shade.position.set(-w * 0.12, h * 0.77, 0);
    C(g, w * 0.18, w * 0.18, 3e-3, -w * 0.12, h * 0.77, 0, M.glow, 40);
  } else if (p.kind === "plant") {
    const ph = Math.min(0.25, h * 0.32);
    if (/VITLÖK/.test(n)) {
      B(g, w, ph, d, 0, ph / 2, 0, M.black, 0.04);
      B(g, w - 0.03, 8e-3, d - 0.025, 0, ph, 0, M.earth, 0.02);
      for (let x of [-1, 1]) tube([[x * w * 0.36, ph * 0.4, -d * 0.52], [x * w * 0.36, ph + 0.07, -d * 0.52], [x * w * 0.36, ph + 0.07, d * 0.52], [x * w * 0.36, ph * 0.4, d * 0.52]], 4e-3, M.black);
    } else {
      lathe([[w * 0.32, 0], [w * 0.44, ph * 0.9], [w * 0.47, ph], [w * 0.42, ph], [w * 0.4, ph * 0.87], [w * 0.3, 0.02]], M.earth);
      C(g, w * 0.39, w * 0.39, 6e-3, 0, ph * 0.9, 0, M.earth);
    }
    for (let i = 0; i < 9; i++) {
      const a = i * 2.399, rr = w * 0.22, tx = Math.cos(a) * rr, tz = Math.sin(a) * d * 0.26, top = ph + (h - ph) * (0.55 + 0.05 * (i % 5));
      tube([[tx, ph, tz], [tx * 0.8, (ph + top) / 2, tz * 0.7], [tx * 0.9, top, tz * 0.8]], 3e-3, M.darkWood);
      for (let j = 0; j < 4; j++) {
        const y = ph + (top - ph) * (j + 1) / 4, aa = a + j * 1.3;
        const leaf = new T.Shape();
        leaf.moveTo(0, 0);
        leaf.quadraticCurveTo(0.07, 0.055, 0.16, 0);
        leaf.quadraticCurveTo(0.075, -0.05, 0, 0);
        const geo = new T.ShapeGeometry(leaf, 8);
        const o = new T.Mesh(geo, M.leaf);
        o.position.set(tx, y, tz);
        o.rotation.set(-0.7, aa, 0.3);
        g.add(o);
        const back = o.clone();
        back.rotation.x += Math.PI;
        g.add(back);
      }
    }
  } else if (/KOPPARBJÖRK/.test(n)) {
    const mat = M.yellow.clone();
    mat.roughness = 0.22;
    lathe([[w * 0.45, 0], [w * 0.49, 0.015], [w * 0.48, h * 0.35], [w * 0.36, h * 0.78], [w * 0.32, h * 0.9], [w * 0.37, h * 0.98], [w * 0.36, h], [w * 0.29, h], [w * 0.29, h * 0.9], [w * 0.31, h * 0.78]], mat);
  } else if (p.kind === "bed" && /齐边|无床头|圆腿/.test(n)) {
    B(g, w, h * 0.55, d, 0, h * 0.65, 0, M.wood, 9e-3);
    for (let x of [-1, 1]) for (let z of [-1, 1]) C(g, 0.035, 0.035, h * 0.4, x * (w / 2 - 0.07), h * 0.2, z * (d / 2 - 0.07), M.wood, 20);
    for (let z = -d / 2 + 0.07; z < d / 2; z += 0.13) B(g, w - 0.08, 0.018, 0.07, 0, h - 9e-3, z, M.wood, 3e-3);
  } else if (p.kind === "towels") {
    for (let i = 0; i < 3; i++) {
      B(g, w, h * 0.27, d, 0, h * 0.15 + i * h * 0.29, 0, M.cream, 0.016);
      for (let j = 0; j < 4; j++) B(g, w * 0.95, 15e-4, 4e-3, 0, h * 0.29 + i * h * 0.29, -d * 0.36 + j * 0.015, M.linen, 1e-3);
    }
  } else if (p.kind === "rug" && /YU-LIVE/.test(n)) {
    B(g, w, h, d, 0, h / 2, 0, M.blue, 3e-3);
  } else return false;
  return true;
}
export {
  hero
};
