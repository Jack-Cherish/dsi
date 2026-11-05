import * as T from "three";
import { box, cylinder, rod, ball, batch } from "./objects.js";
function polygon(points, y, m) {
  const s = new T.Shape();
  points.forEach(([x, z], i) => i ? s.lineTo(x, -z) : s.moveTo(x, -z));
  s.closePath();
  const g = new T.ShapeGeometry(s);
  const uv = g.attributes.uv, pos = g.attributes.position;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, pos.getX(i) / 2, -pos.getY(i) / 2);
  const o = new T.Mesh(g, m);
  o.rotation.x = -Math.PI / 2;
  o.position.y = y;
  o.receiveShadow = true;
  return o;
}
function segment(g, a, b, thick, height, bottom, m) {
  const dx = b[0] - a[0], dz = b[1] - a[1], l = Math.hypot(dx, dz);
  const o = box(g, l, height, thick, (a[0] + b[0]) / 2, bottom + height / 2, (a[1] + b[1]) / 2, m, 2e-3);
  o.rotation.y = -Math.atan2(dz, dx);
  return o;
}
function cup(g, x, y, z, M) {
  cylinder(g, 0.034, 0.026, 0.067, x, y + 0.033, z, M.white);
  cylinder(g, 0.028, 0.028, 2e-3, x, y + 0.067, z, M.darkWood);
  const tor = new T.Mesh(new T.TorusGeometry(0.022, 5e-3, 8, 20), M.white);
  tor.position.set(x + 0.039, y + 0.035, z);
  g.add(tor);
}
function windowUnit(root, w, M) {
  const { x1, z1, x2, z2 } = w, l = Math.hypot(x2 - x1, z2 - z1), g = new T.Group();
  g.position.set((x1 + x2) / 2, 0, (z1 + z2) / 2);
  g.rotation.y = -Math.atan2(z2 - z1, x2 - x1);
  root.add(g);
  box(g, l, 0.78, 0.2, 0, 0.39, 0, M.wall, 0);
  box(g, l, 0.35, 0.2, 0, 2.525, 0, M.wall, 0);
  for (let x of [-l / 2 + 0.022, l / 2 - 0.022]) box(g, 0.04, 1.57, 0.065, x, 1.565, 0, M.darkWood, 3e-3);
  for (let y of [0.79, 2.35]) box(g, l, 0.04, 0.065, 0, y, 0, M.darkWood, 3e-3);
  box(g, l + 0.055, 0.028, 0.27, 0, 0.8, 0, M.wood, 5e-3);
  const glass = box(g, l - 0.07, 1.5, 4e-3, 0, 1.57, 0, M.glass, 0);
  glass.castShadow = false;
  for (let j = 0; j < 14; j++) {
    const slat = box(g, l - 0.05, 0.012, 0.036, 0, 2.3 - j * 0.033, 0.055, M.darkWood, 2e-3);
    slat.rotation.x = -0.28;
  }
  for (let x of [-l * 0.28, l * 0.28]) rod(g, [x, 1.86, 0.055], [x, 2.34, 0.055], 16e-4, M.black);
}
function house(layout, M) {
  const root = new T.Group();
  root.name = "\u5305\u8C6A\u65AF\u539F\u6237\u578B";
  const walls = new T.Group(), ceiling = new T.Group(), fixed = new T.Group(), yard = new T.Group();
  root.add(walls, ceiling, fixed, yard);
  root.add(polygon(layout.floor, 0, M.floor));
  for (const [x1, z1, x2, z2, t] of layout.walls) {
    segment(walls, [x1, z1], [x2, z2], t, 2.7, 0, M.wall);
    segment(walls, [x1, z1], [x2, z2], t + 0.014, 0.065, 2e-3, M.black);
  }
  layout.windows.forEach((w) => windowUnit(walls, w, M));
  for (const door of layout.doors) {
    const { x1, z1, x2, z2 } = door;
    segment(walls, [x1, z1], [x2, z2], 0.2, 0.48, 2.22, M.wall);
    for (let x of [x1, x2]) box(walls, 0.035, 2.22, 0.12, x, 1.11, z1, M.wood, 4e-3);
  }
  segment(walls, [2.41, 5], [3.21, 5], 0.12, 0.48, 2.22, M.wall);
  segment(walls, [3.01, 3.954], [5.745, 3.954], 0.065, 0.28, 2.42, M.wall);
  const ceilMat = M.ceiling.clone();
  ceilMat.side = T.DoubleSide;
  const roof = polygon(layout.floor, 2.7, ceilMat);
  roof.castShadow = true;
  ceiling.add(roof);
  box(yard, 5.4, 0.1, 1.75, 2.72, -0.06, -0.75, M.stone, 3e-3);
  box(yard, 2.35, 0.1, 1.5, 4.2, -0.06, 0.8, M.stone, 3e-3);
  box(yard, 3.78, 0.1, 3.72, 2.33, -0.06, 10.04, M.stone, 3e-3);
  for (let x = 0.5; x < 4.2; x += 0.47) box(yard, 3e-3, 2e-3, 3.67, x, 1e-3, 10.05, M.grout, 0);
  for (let z = 8.3; z < 11.9; z += 0.47) box(yard, 3.75, 2e-3, 3e-3, 2.33, 1e-3, z, M.grout, 0);
  segment(yard, [0.445, 8.28], [0.445, 11.9], 0.12, 1.3, 0, M.wall);
  segment(yard, [4.22, 8.28], [4.22, 11.9], 0.12, 1.3, 0, M.wall);
  segment(yard, [0.445, 11.9], [4.22, 11.9], 0.12, 1.3, 0, M.wall);
  segment(yard, [0.05, -1.55], [4, -1.55], 0.12, 1.4, 0, M.wall);
  segment(yard, [0.05, -1.55], [0.05, 0.1], 0.12, 1.4, 0, M.wall);
  box(fixed, 0.6, 0.86, 1.85, 0.515, 0.43, 2.525, M.olive, 6e-3);
  box(fixed, 0.625, 0.045, 1.88, 0.525, 0.8925, 2.525, M.black, 6e-3);
  for (let z of [1.9, 2.525, 3.15]) {
    box(fixed, 0.022, 0.72, 0.6, 0.827, 0.43, z, M.olive, 3e-3);
    rod(fixed, [0.845, 0.76, z - 0.13], [0.845, 0.76, z + 0.13], 5e-3, M.brushed);
  }
  box(fixed, 0.61, 1.95, 0.6, 0.515, 0.975, 1.3, M.brushed, 0.012);
  box(fixed, 0.012, 1.16, 0.575, 0.829, 1.33, 1.3, M.brushed, 4e-3);
  box(fixed, 0.012, 0.68, 0.575, 0.829, 0.385, 1.3, M.brushed, 4e-3);
  rod(fixed, [0.853, 0.8, 1.15], [0.853, 1.72, 1.15], 8e-3, M.black);
  box(fixed, 0.012, 0.72, 1.85, 0.224, 1.28, 2.525, M.grout, 0);
  const tiles = Array.from({ length: 9 }, (_, i) => {
    const m = M.tile.clone();
    m.color.multiplyScalar(0.88 + i * 0.025);
    return m;
  });
  for (let j = 0; j < 3; j++) for (let i = 0; i < 24; i++) {
    box(fixed, 0.018, 0.224, 0.07, 0.238, 1.04 + j * 0.234, 1.63 + i * 0.076, tiles[(i * 7 + j * 3) % 9], 3e-3);
  }
  box(fixed, 0.32, 0.52, 1.12, 0.38, 2.06, 2.4, M.olive, 6e-3);
  for (let z of [2.12, 2.68]) box(fixed, 0.015, 0.49, 0.535, 0.549, 2.06, z, M.olive, 2e-3);
  box(fixed, 0.36, 0.027, 0.52, 0.4, 1.57, 3.2, M.wood, 4e-3);
  box(fixed, 0.012, 0.012, 1.08, 0.545, 1.79, 2.4, M.glow, 2e-3);
  box(fixed, 0.35, 0.015, 0.38, 0.54, 0.922, 1.93, M.brushed, 0.012);
  box(fixed, 0.28, 9e-3, 0.29, 0.54, 0.929, 1.93, M.black, 0.025);
  rod(fixed, [0.37, 0.925, 1.93], [0.37, 1.16, 1.93], 0.012, M.chrome);
  rod(fixed, [0.37, 1.16, 1.93], [0.53, 1.16, 1.93], 0.012, M.chrome);
  box(fixed, 0.43, 0.012, 0.52, 0.54, 0.925, 2.69, M.black, 7e-3);
  for (let z of [2.54, 2.84]) {
    cylinder(fixed, 0.09, 0.09, 0.011, 0.54, 0.937, z, M.brushed);
    cylinder(fixed, 0.07, 0.07, 0.013, 0.54, 0.944, z, M.black);
  }
  box(fixed, 0.27, 0.29, 0.24, 0.53, 1.08, 3.2, M.brushed, 0.013);
  box(fixed, 0.017, 0.17, 0.21, 0.676, 1.1, 3.2, M.black, 3e-3);
  box(fixed, 0.13, 0.012, 0.25, 0.69, 0.955, 3.2, M.chrome, 3e-3);
  rod(fixed, [0.7, 1.1, 3.16], [0.81, 1.1, 3.16], 0.014, M.darkWood);
  cup(fixed, 0.68, 0.962, 3.25, M);
  for (let x = 0.22; x < 1.34; x += 0.16) for (let z = 1.61; z < 3.44; z += 0.16) box(fixed, 0.157, 6e-3, 0.157, x + 0.08, 5e-3, z + 0.08, (Math.round(x / 0.16) + Math.round(z / 0.16)) % 2 ? M.white : M.black, 0);
  box(fixed, 0.58, 0.35, 0.56, 7.03, 0.62, 5.21, M.wood, 8e-3);
  box(fixed, 0.58, 0.025, 0.56, 7.03, 0.1675, 5.21, M.wood, 5e-3);
  for (const z of [4.945, 5.475]) box(fixed, 0.58, 0.28, 0.025, 7.03, 0.3, z, M.wood, 3e-3);
  box(fixed, 0.6, 0.05, 0.58, 7.03, 0.825, 5.21, M.stone, 8e-3);
  box(fixed, 0.42, 0.025, 0.36, 7.03, 0.859, 5.21, M.white, 0.06);
  for (let x of [-1, 1]) box(fixed, 0.033, 0.08, 0.36, 7.03 + x * 0.2, 0.888, 5.21, M.white, 0.015);
  for (let z of [-1, 1]) box(fixed, 0.4, 0.08, 0.025, 7.03, 0.888, 5.21 + z * 0.168, M.white, 0.01);
  rod(fixed, [7.25, 0.88, 5.21], [7.25, 1.08, 5.21], 0.012, M.chrome);
  rod(fixed, [7.25, 1.08, 5.21], [7.1, 1.08, 5.21], 0.012, M.chrome);
  const mirror = new T.MeshPhysicalMaterial({ color: "#bfc5bd", metalness: 1, roughness: 0.045 });
  box(fixed, 0.024, 0.78, 0.61, 7.327, 1.49, 5.21, M.black, 3e-3);
  box(fixed, 5e-3, 0.74, 0.57, 7.31, 1.49, 5.21, mirror, 0);
  for (let x = 5.56; x < 7.35; x += 0.22) for (let z = 6.06; z < 8.16; z += 0.22) box(fixed, 0.216, 0.012, 0.216, x + 0.108, 7e-3, z + 0.108, M.tile, 2e-3);
  rod(fixed, [7.31, 0.78, 7.5], [7.14, 0.78, 7.5], 0.013, M.chrome);
  rod(fixed, [7.14, 0.78, 7.5], [7.14, 0.7, 7.5], 0.013, M.chrome);
  cylinder(fixed, 0.025, 0.025, 0.035, 7.3, 0.78, 7.38, M.chrome).rotation.z = Math.PI / 2;
  box(fixed, 0.36, 0.55, 0.18, 5.91, 0.31, 7.9, M.white, 0.04);
  ball(fixed, 5.91, 0.3, 7.57, 0.22, 0.22, 0.31, M.white);
  box(fixed, 0.39, 0.035, 0.52, 5.91, 0.48, 7.57, M.white, 0.08);
  for (const [x, z] of [[1.8, 1], [1.8, 3], [4.4, 2.6], [4.4, 4.5], [4.3, 6.8], [1.8, 6.8], [6.4, 5.4], [6.4, 6.9]]) {
    cylinder(ceiling, 0.035, 0.035, 0.012, x, 2.688, z, M.black);
    cylinder(ceiling, 0.021, 0.021, 0.013, x, 2.68, z, M.glow);
  }
  for (const x of [1, 2.45]) box(ceiling, 0.012, 8e-3, 1.5, x, 2.687, 3.95, M.glow, 0);
  for (const z of [3.2, 4.7]) box(ceiling, 1.45, 8e-3, 0.012, 1.725, 2.687, z, M.glow, 0);
  for (let x = 0.58; x < 4.14; x += 0.12) box(yard, 0.048, 0.85, 0.035, x, 1.72, 11.85, M.darkWood, 3e-3);
  for (const x of [0.64, 4.02]) {
    rod(yard, [x, 0, 9], [x, 2.35, 9], 0.018, M.black);
    rod(yard, [x, 0, 11.55], [x, 2.35, 11.55], 0.018, M.black);
  }
  for (let j = 0; j < 2; j++) {
    const z = 9 + j * 2.55;
    for (let i = 0; i < 14; i++) {
      const x = 0.64 + i * 3.38 / 13, y = 2.35 - 0.22 * Math.sin(i / 13 * Math.PI);
      if (i < 13) {
        const nx = 0.64 + (i + 1) * 3.38 / 13, ny = 2.35 - 0.22 * Math.sin((i + 1) / 13 * Math.PI);
        rod(yard, [x, y, z], [nx, ny, z], 2e-3, M.black);
      }
      ball(yard, x, y - 0.055, z, 0.018, 0.025, 0.018, M.glow);
    }
  }
  for (const [x, z] of [[0.7, -0.7], [3.8, -1.2], [0.7, 8.9], [4, 9.6], [0.7, 11.4], [4, 11.4]]) {
    box(yard, 0.07, 0.34, 0.07, x, 0.17, z, M.black, 5e-3);
    box(yard, 0.074, 0.06, 0.074, x, 0.3, z, M.glow, 3e-3);
  }
  for (const group of [walls, ceiling, fixed, yard]) batch(group);
  root.userData = { walls, ceiling, fixed, yard };
  return root;
}
export {
  house
};
