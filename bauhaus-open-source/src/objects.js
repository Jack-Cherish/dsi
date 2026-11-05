import * as T from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { accessory } from "./accessories.js";
import { hero } from "./hero-models.js";
import { photoV5 } from "./photo-models-v5.js";
import { photoModel } from "./photo-models-v4.js";
function batch(g) {
  g.updateMatrixWorld(true);
  const groups = /* @__PURE__ */ new Map(), inverse = g.matrixWorld.clone().invert();
  g.traverse((o) => {
    if (!o.isMesh) return;
    let geometry = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone();
    geometry.applyMatrix4(inverse.clone().multiply(o.matrixWorld));
    let a = groups.get(o.material);
    if (!a) groups.set(o.material, a = []);
    a.push(geometry);
  });
  g.clear();
  for (const [material, geos] of groups) {
    const geometry = mergeGeometries(geos, false);
    if (!geometry) throw Error("Geometry batch failed");
    const m = new T.Mesh(geometry, material);
    m.castShadow = !material.transparent;
    m.receiveShadow = true;
    g.add(m);
    geos.forEach((x) => x.dispose());
  }
  return g;
}
function box(g, w, h, d, x, y, z, m, r = 8e-3) {
  const soft = !!m.sheen && r > 0.035;
  const geo = r > 0 ? new RoundedBoxGeometry(w, h, d, soft ? 5 : 2, Math.min(r, w / 3, h / 3, d / 3)) : new T.BoxGeometry(w, h, d);
  if (soft) {
    const p = geo.attributes.position, n = geo.attributes.normal;
    for (let i = 0; i < p.count; i++) {
      const a = p.getX(i), b = p.getY(i), c = p.getZ(i), f = 2e-3 * (Math.sin(a * 48 + c * 9) * Math.cos(b * 31) + Math.sin(c * 59 - a * 12) * 0.4);
      p.setXYZ(i, a + n.getX(i) * f, b + n.getY(i) * f, c + n.getZ(i) * f);
    }
    geo.computeVertexNormals();
  }
  const o = new T.Mesh(geo, m);
  o.position.set(x, y, z);
  o.castShadow = o.receiveShadow = true;
  g.add(o);
  return o;
}
function cylinder(g, rt, rb, h, x, y, z, m, n = 32) {
  const o = new T.Mesh(new T.CylinderGeometry(rt, rb, h, n), m);
  o.position.set(x, y, z);
  o.castShadow = o.receiveShadow = true;
  g.add(o);
  return o;
}
function ball(g, x, y, z, sx, sy, sz, m) {
  const o = new T.Mesh(new T.SphereGeometry(1, 24, 16), m);
  o.position.set(x, y, z);
  o.scale.set(sx, sy, sz);
  o.castShadow = o.receiveShadow = true;
  g.add(o);
  return o;
}
function rod(g, a, b, r, m) {
  const p = new T.Vector3(...a), q = new T.Vector3(...b), o = cylinder(g, r, r, p.distanceTo(q), ...p.clone().add(q).multiplyScalar(0.5).toArray(), m, 12);
  o.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), q.sub(p).normalize());
  return o;
}
function seams(g, w, d, y, m) {
  for (const z of [-d / 2, d / 2]) rod(g, [-w / 2, y, z], [w / 2, y, z], 18e-4, m);
}
function cloth(g, w, d, y, color) {
  const geo = new T.PlaneGeometry(w, d, 36, 52), p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), z = p.getY(i);
    const hem = Math.max(0, -z / d - 0.26) * 2;
    const ripple = 8e-3 * Math.sin(x * 36 + z * 11) + 9e-3 * Math.sin(x * 17 - z * 28);
    p.setXYZ(i, x, y + ripple - 0.17 * hem * hem, -z);
  }
  geo.computeVertexNormals();
  const o = new T.Mesh(geo, color);
  o.castShadow = o.receiveShadow = true;
  g.add(o);
}
function plant(g, w, d, h, M) {
  const rad = Math.min(w, d) * 0.36, ph = Math.min(h * 0.3, 0.35);
  cylinder(g, rad, rad * 0.7, ph, 0, ph / 2, 0, M.white);
  cylinder(g, rad * 0.94, rad * 0.94, 0.014, 0, ph, 0, M.earth);
  for (let k = 0; k < 5; k++) {
    const angle = k * 2.399;
    const tip = [Math.sin(angle) * w * 0.22, h * (0.72 + 0.05 * k), Math.cos(angle) * d * 0.22];
    rod(g, [0, ph, 0], tip, 8e-3, M.darkWood);
    for (let j = 1; j < 5; j++) {
      const f = j / 5, pos = tip.map((v, i) => i === 1 ? ph + (v - ph) * f : v * f), a = angle + j * 1.7;
      const end = [pos[0] + Math.cos(a) * w * 0.3, pos[1] + h * 0.08, pos[2] + Math.sin(a) * d * 0.3];
      rod(g, pos, end, 25e-4, M.darkWood);
      const leaf = ball(g, ...end, w * 0.16, 0.013, d * 0.065, M.leaf);
      leaf.rotation.set(0.2, -a, 0.3 * Math.cos(a));
    }
  }
}
function books(g, w, d, y, M) {
  const colors = [M.red, M.blue, M.paper, M.darkWood, M.yellow];
  for (let i = 0; i < Math.floor(w / 0.09); i++) {
    const h = 0.17 + i % 3 * 0.025;
    box(g, 0.035 + i % 2 * 0.013, h, d * 0.65, -w / 2 + 0.065 + i * 0.07, y + h / 2, 0, colors[i % 5], 1e-3);
  }
}
function product(p, M) {
  const g = new T.Group();
  g.name = p.id;
  let [w, d, h] = p.dims;
  const k = p.kind, v = p.slot, wood = v === 2 ? M.darkWood : M.wood;
  if (photoV5(p, g, M, { box, cylinder, ball, rod })) {
  } else if (photoModel(p, g, M, { box, cylinder, ball, rod })) {
  } else if (hero(p, g, M, { box, cylinder, ball, rod })) {
  } else if (accessory(p, g, M, { box, cylinder, ball, rod })) {
  } else if (k === "sofa") {
    const mat = /橙/.test(p.name) ? M.orange : /黑/.test(p.name) ? M.leather : /米白|小沙发/.test(p.name) ? M.linen : M.gray;
    box(g, w - 0.08, 0.12, d - 0.07, 0, 0.19, 0, wood, 0.025);
    for (let i of [-1, 1]) for (let j of [-1, 1]) rod(g, [i * (w / 2 - 0.12), 0.04, j * (d / 2 - 0.12)], [i * (w / 2 - 0.14), 0.21, j * (d / 2 - 0.13)], 0.025, M.black);
    for (let i of [-1, 1]) {
      box(g, (w - 0.24) / 2, 0.16, d - 0.13, i * (w - 0.24) / 4, 0.39, -0.035, mat, 0.065);
      const b = box(g, (w - 0.2) / 2, 0.31, 0.14, i * (w - 0.2) / 4, h - 0.18, d / 2 - 0.1, mat, 0.055);
      b.rotation.x = -0.08;
      if (/木墨/.test(p.name)) {
        box(g, 0.055, 0.05, d - 0.03, i * (w / 2 - 0.04), 0.56, 0, wood, 0.012);
        for (let z of [-1, 1]) rod(g, [i * (w / 2 - 0.04), 0.15, z * d * 0.35], [i * (w / 2 - 0.04), 0.56, z * d * 0.35], 0.018, wood);
      } else box(g, 0.12, 0.3, d - 0.03, i * (w / 2 - 0.06), 0.43, 0, mat, 0.045);
    }
    seams(g, w - 0.22, d - 0.16, 0.463, mat);
  } else if (k === "armchair") {
    if (v === 0) {
      cylinder(g, 0.042, 0.055, 0.27, 0, 0.18, 0, M.chrome);
      for (let j = 0; j < 4; j++) {
        const a = j * Math.PI / 2 + 0.4;
        rod(g, [0, 0.1, 0], [Math.cos(a) * w * 0.44, 0.035, Math.sin(a) * d * 0.4], 0.019, M.chrome);
      }
      box(g, w * 0.9, 0.16, d * 0.65, 0, 0.33, -d * 0.08, M.green, 0.075);
      const b = box(g, w * 0.9, 0.42, 0.18, 0, 0.5, d * 0.24, M.green, 0.08);
      b.rotation.x = -0.2;
      for (let i of [-1, 1]) for (let j = 0; j < 2; j++) ball(g, i * w * 0.17, 0.45 + j * 0.13, d * 0.12, 0.018, 0.018, 9e-3, M.green);
    } else {
      for (let x of [-1, 1]) for (let z of [-1, 1]) rod(g, [x * w * 0.4, 0.02, z * d * 0.36], [x * w * 0.35, 0.5, z * d * 0.3], 0.028, wood);
      box(g, w * 0.84, 0.14, d * 0.7, 0, 0.4, -0.04, M.linen, 0.05);
      const b = box(g, w * 0.84, 0.38, 0.14, 0, 0.61, d * 0.26, M.linen, 0.05);
      b.rotation.x = -0.15;
      for (let x of [-1, 1]) box(g, 0.045, 0.05, d * 0.85, x * w * 0.47, 0.56, 0, wood, 0.015);
    }
  } else if (k === "bed" || k === "guest") {
    const frame = Math.min(h, 0.32);
    box(g, w, frame - 0.08, d, 0, (frame + 0.08) / 2, 0, v === 1 ? M.gray : wood, 0.018);
    for (let x of [-1, 1]) for (let z of [-1, 1]) box(g, 0.055, 0.09, 0.055, x * (w / 2 - 0.09), 0.045, z * (d / 2 - 0.09), M.black);
    box(g, w - 0.04, 0.16, d - 0.06, 0, frame + 0.08, 0, M.cream, 0.055);
    if (h > 0.55) box(g, w, 0.7, 0.085, 0, 0.48, d / 2 - 0.045, M.gray, 0.04);
    for (let x of k === "bed" ? [-w * 0.24, w * 0.24] : [0]) {
      const o = box(g, k === "bed" ? w * 0.4 : w * 0.76, 0.13, 0.43, x, frame + 0.225, d / 2 - 0.32, M.cream, 0.06);
      o.rotation.y = 0.04;
    }
    cloth(g, w * 0.99, d * 0.72, frame + 0.18, k === "guest" ? M.blue : M.gray);
    box(g, w * 0.94, 0.045, 0.4, 0, frame + 0.2, -d * 0.25, k === "guest" ? M.linen : M.orange, 0.018);
  } else if (k === "table" || k === "desk" || k === "outtable") {
    if (k === "outtable") {
      cylinder(g, w / 2, w / 2, 0.025, 0, h - 0.012, 0, v === 0 ? M.blue : M.black);
      for (let i of [-1, 1]) {
        rod(g, [-w * 0.28, 0.02, i * d * 0.22], [w * 0.28, h - 0.04, i * d * 0.22], 0.012, M.black);
        rod(g, [w * 0.28, 0.02, i * d * 0.22], [-w * 0.28, h - 0.04, i * d * 0.22], 0.012, M.black);
      }
    } else {
      box(g, w, 0.034, d, 0, h - 0.017, 0, k === "table" ? M.darkWood : v === 0 ? M.white : wood, 6e-3);
      for (let x of [-1, 1]) for (let z of [-1, 1]) box(g, 0.032, h - 0.04, 0.032, x * (w / 2 - 0.065), (h - 0.04) / 2, z * (d / 2 - 0.065), k === "table" ? M.black : wood, 5e-3);
      if (k === "desk") {
        box(g, w - 0.12, 0.1, d - 0.09, 0, h - 0.1, 0, wood);
        box(g, 0.17, 0.012, 8e-3, 0, h - 0.1, -d / 2 - 2e-3, M.brushed, 2e-3);
      }
    }
  } else if (k === "chair" || k === "outchair" || k === "office") {
    const seat = k === "office" ? M.leather : /白/.test(p.name) ? M.white : /绿/.test(p.name) ? M.green : /蓝/.test(p.name) ? M.blue : /红/.test(p.name) ? M.red : /木|吱音/.test(p.name) ? wood : M.linen, sy = k === "office" ? 0.47 : 0.45;
    box(g, w * 0.84, 0.055, d * 0.8, 0, sy, 0, seat, 0.023);
    const back = box(g, w * 0.82, h - sy - 0.12, 0.045, 0, (h + sy) / 2, d * 0.36, seat, 0.02);
    back.rotation.x = -0.06;
    if (k === "office") {
      cylinder(g, 0.03, 0.03, 0.38, 0, 0.25, 0, M.chrome);
      for (let i = 0; i < 5; i++) {
        const a = i * Math.PI * 0.4;
        rod(g, [0, 0.12, 0], [Math.cos(a) * w * 0.42, 0.06, Math.sin(a) * d * 0.42], 0.017, M.chrome);
        ball(g, Math.cos(a) * w * 0.42, 0.035, Math.sin(a) * d * 0.42, 0.026, 0.025, 0.025, M.black);
      }
    } else for (let x of [-1, 1]) for (let z of [-1, 1]) rod(g, [x * w * 0.43, 0.015, z * d * 0.41], [x * w * 0.35, sy - 0.03, z * d * 0.31], 0.011, k === "chair" && v === 1 ? wood : M.black);
  } else if (k === "wardrobe" || k === "shoe" || k === "nightstand" || k === "shelf") {
    const mat = /棕色|胡桃色/.test(p.name) ? M.wood : /TRONES/.test(p.name) ? M.black : /BILLY|BAGGEBO|MACKAP|BISSA/.test(p.name) ? M.white : k === "wardrobe" && v === 2 ? M.wood : k === "shelf" || k === "nightstand" && v === 2 ? wood : M.white;
    if (k === "nightstand" && v === 1) box(g, w, h, d, 0, h / 2, 0, mat);
    else if (k === "nightstand" && v === 2) {
      box(g, w, 0.04, d, 0, h - 0.02, 0, mat);
      for (let x of [-1, 1]) for (let z of [-1, 1]) box(g, 0.035, h - 0.04, 0.035, x * (w / 2 - 0.02), (h - 0.04) / 2, z * (d / 2 - 0.02), mat);
    } else {
      box(g, w, 0.025, d, 0, 0.03, 0, mat);
      box(g, w, 0.025, d, 0, h - 0.0125, 0, mat);
      box(g, w, 0.018, d, 0, h * 0.5, 0, mat);
      for (let x of [-1, 1]) box(g, 0.023, h, d, x * (w / 2 - 0.012), h / 2, 0, mat);
      box(g, w, h, 0.012, 0, h / 2, d / 2 - 6e-3, mat, 0);
      if (k === "shelf") {
        for (let j = 1; j < 5; j++) {
          const y = j * h / 5;
          box(g, w, 0.018, d, 0, y, 0, mat);
          books(g, w - 0.06, d, y + 0.01, M);
        }
      } else if (k === "nightstand") {
        books(g, w - 0.06, d, 0.05, M);
      } else {
        for (let x of [-1, 1]) {
          box(g, w / 2 - 6e-3, h - 0.07, 0.025, x * w / 4, h / 2, -d / 2, mat, 4e-3);
          rod(g, [x * 0.035, h * 0.49, -d / 2 - 0.025], [x * 0.035, h * 0.57, -d / 2 - 0.025], 5e-3, M.chrome);
        }
      }
    }
  } else if (k === "bench") {
    box(g, w, 0.04, d, 0, h - 0.02, 0, wood);
    for (let x of [-1, 1]) box(g, 0.04, h - 0.04, d * 0.85, x * (w / 2 - 0.065), (h - 0.04) / 2, 0, wood);
    box(g, w - 0.13, 0.03, d * 0.88, 0, 0.15, 0, wood);
  } else if (k === "side" || k === "ottoman") {
    if (/BURVIK/.test(p.name)) {
      cylinder(g, w / 2, w / 2, 0.025, 0, h * 0.61, 0, M.red);
      for (let a of [0, 2.094, 4.189]) rod(g, [Math.cos(a) * w * 0.33, 0.01, Math.sin(a) * d * 0.33], [Math.cos(a) * w * 0.31, h * 0.6, Math.sin(a) * d * 0.31], 0.012, M.red);
      const curve = new T.CatmullRomCurve3([new T.Vector3(-w * 0.22, h * 0.62, 0), new T.Vector3(-w * 0.22, h * 0.92, 0), new T.Vector3(0, h, 0), new T.Vector3(w * 0.22, h * 0.92, 0), new T.Vector3(w * 0.22, h * 0.62, 0)]);
      g.add(new T.Mesh(new T.TubeGeometry(curve, 32, 8e-3, 8, false), M.red));
    } else if (/LACK/.test(p.name)) {
      box(g, w, 0.035, d, 0, h - 0.0175, 0, M.white);
      for (let x of [-1, 1]) for (let z of [-1, 1]) box(g, 0.045, h - 0.035, 0.045, x * (w / 2 - 0.03), (h - 0.035) / 2, z * (d / 2 - 0.03), M.white);
    } else if (/GLADOM/.test(p.name)) {
      cylinder(g, w / 2, w / 2, 0.025, 0, h - 0.014, 0, M.green);
      for (let a of [0, Math.PI / 2, Math.PI, Math.PI * 1.5]) rod(g, [Math.cos(a) * w * 0.34, 0.01, Math.sin(a) * d * 0.34], [Math.cos(a) * w * 0.34, h - 0.025, Math.sin(a) * d * 0.34], 9e-3, M.green);
    } else if (/信凳/.test(p.name)) {
      const blue = new T.MeshPhysicalMaterial({ color: 1521838, roughness: 0.32, clearcoat: 0.3 });
      box(g, w, 0.025, d, 0, h - 0.013, 0, blue, 0.012);
      for (let z of [-1, 1]) {
        const pts = [[-0.4, 0.96], [-0.47, 0.25], [-0.42, 0.12], [-0.25, 0.025], [0, 0.025], [0.25, 0.025], [0.42, 0.12], [0.47, 0.25], [0.4, 0.96]].map(([x, y]) => new T.Vector3(x * w, y * h, z * d * 0.3));
        const curve = new T.CatmullRomCurve3(pts);
        const profile = new T.Shape();
        profile.moveTo(-0.017, -9e-3);
        profile.lineTo(0.017, -9e-3);
        profile.lineTo(0.017, 9e-3);
        profile.lineTo(-0.017, 9e-3);
        profile.closePath();
        g.add(new T.Mesh(new T.ExtrudeGeometry(profile, { steps: 52, bevelEnabled: false, extrudePath: curve }), blue));
      }
    } else {
      cylinder(g, w / 2, w / 2, 0.035, 0, h - 0.018, 0, k === "ottoman" ? M.linen : M.black);
      cylinder(g, w * 0.2, w * 0.28, h - 0.035, 0, (h - 0.035) / 2, 0, wood);
    }
  } else if (k === "rug") {
    box(g, w, h, d, 0, h / 2 + 6e-3, 0, v === 0 ? M.rug : v === 1 ? M.linen : M.cream, 3e-3);
  } else if (k === "pendant") {
    cylinder(g, 0.025, 0.025, 0.55, 0, h + 0.26, 0, M.black, 12);
    cylinder(g, 0.08, 0.08, 0.015, 0, h + 0.54, 0, M.black);
    const points = [new T.Vector2(0.035, h), new T.Vector2(0.055, h - 0.09), new T.Vector2(w * 0.4, h - 0.13), new T.Vector2(w * 0.5, h - 0.17), new T.Vector2(w * 0.49, h - 0.2), new T.Vector2(w * 0.28, h - 0.24), new T.Vector2(0.03, h - 0.28)];
    const o = new T.Mesh(new T.LatheGeometry(points, 64), v === 0 ? M.white : M.chrome);
    o.castShadow = true;
    g.add(o);
    ball(g, 0, h - 0.23, 0, 0.045, 0.035, 0.045, M.glow);
  } else if (k === "floorlamp") {
    cylinder(g, w * 0.44, w * 0.44, 0.025, 0, 0.013, 0, M.brushed);
    rod(g, [0, 0.02, 0], [0, h - 0.12, 0], 9e-3, M.brushed);
    const points = [new T.Vector2(0.025, h), new T.Vector2(0.08, h - 0.02), new T.Vector2(w * 0.45, h - 0.12), new T.Vector2(w * 0.45, h - 0.15)];
    const o = new T.Mesh(new T.LatheGeometry(points, 48), v === 2 ? M.cream : M.brushed);
    g.add(o);
    ball(g, 0, h - 0.12, 0, 0.065, 0.015, 0.065, M.glow);
  } else if (k === "plant") plant(g, w, d, h, M);
  else if (k === "mirror" || k === "art") {
    box(g, w + 0.02, h + 0.02, d, 0, h / 2, 0, k === "art" ? M.darkWood : M.black, 4e-3);
    const mat = k === "mirror" ? new T.MeshPhysicalMaterial({ color: "#c3c7c4", metalness: 1, roughness: 0.04 }) : new T.MeshStandardMaterial({ color: "#e4e0d6", roughness: 0.95 });
    if (k === "art" && p.photo) {
      const t = new T.TextureLoader().load(p.photo);
      t.colorSpace = T.SRGBColorSpace;
      let crop = p.sourceKey === "J03_1" ? [0.24, 0.12, 0.74, 0.61] : p.sourceKey === "J03_2" ? [0.18, 0.07, 0.86, 0.79] : p.sourceKey === "J03_3" ? [0.34, 0.1, 0.82, 0.7] : p.imageKey === "N097_3" ? [60 / 1400, 459 / 1400, 1339 / 1400, 940 / 1400] : p.imageKey === "N097_1" ? [236 / 1400, 48 / 1400, 1164 / 1400, 1352 / 1400] : [220 / 1400, 29 / 1400, 1178 / 1400, 1370 / 1400];
      t.repeat.set(crop[2] - crop[0], crop[3] - crop[1]);
      t.offset.set(crop[0], 1 - crop[3]);
      mat.color.set("#ffffff");
      mat.map = t;
    }
    box(g, w - 0.025, h - 0.025, 2e-3, 0, h / 2, -d / 2 - 2e-3, mat, 0);
  } else if (k === "vase") {
    if (/CYLINDER/.test(p.name)) {
      for (let i = 0; i < 3; i++) {
        const r = w * 0.155, ht = h * (1 - i * 0.25), pts = [[r, 0], [r, ht], [r - 3e-3, ht], [r - 3e-3, 4e-3], [0, 4e-3]].map((a) => new T.Vector2(...a)), o = new T.Mesh(new T.LatheGeometry(pts, 40), M.glass);
        o.position.set((i - 1) * w * 0.33, 0, i % 2 * d * 0.25);
        g.add(o);
      }
    } else {
      const points = [new T.Vector2(w * 0.25, 0), new T.Vector2(w * 0.47, h * 0.12), new T.Vector2(w * 0.5, h * 0.5), new T.Vector2(w * 0.26, h * 0.82), new T.Vector2(w * 0.27, h), new T.Vector2(w * 0.23, h), new T.Vector2(w * 0.22, h * 0.78)];
      g.add(new T.Mesh(new T.LatheGeometry(points, 48), /KOPPAR/.test(p.name) ? M.yellow : /STILREN/.test(p.name) ? M.white : v === 0 ? M.green : v === 1 ? M.glass : M.white));
    }
  } else {
    g.userData.fallback = true;
    box(g, w, h, d, 0, h / 2, 0, wood);
  }
  batch(g);
  const bounds = new T.Box3().setFromObject(g), size = bounds.getSize(new T.Vector3()), center = bounds.getCenter(new T.Vector3());
  for (const mesh of g.children) {
    mesh.geometry.translate(-center.x, -bounds.min.y, -center.z);
    mesh.geometry.scale(w / Math.max(size.x, 1e-3), h / Math.max(size.y, 1e-3), d / Math.max(size.z, 1e-3));
  }
  g.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = k !== "rug" && k !== "mirror";
      o.receiveShadow = true;
      o.userData.kind = k;
    }
  });
  const placement = new T.Group();
  placement.name = p.id;
  placement.userData.fallback = !!g.userData.fallback;
  g.rotation.y = Math.PI;
  placement.add(g);
  return placement;
}
export {
  ball,
  batch,
  box,
  cylinder,
  product,
  rod
};
