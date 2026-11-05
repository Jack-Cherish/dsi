function corners(o, p) {
  const a = o.rotation * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), w = p.dims[0] / 2, d = p.dims[1] / 2;
  return [[-w, -d], [w, -d], [w, d], [-w, d]].map(([x, z]) => [o.x + c * x + s * z, o.z - s * x + c * z]);
}
function overlap(a, b, tolerance = 0.012) {
  for (const poly of [a, b]) for (let i = 0; i < 4; i++) {
    let x = poly[(i + 1) % 4][0] - poly[i][0], z = poly[(i + 1) % 4][1] - poly[i][1], l = Math.hypot(x, z), axis = [-z / l, x / l];
    let pa = a.map((p) => p[0] * axis[0] + p[1] * axis[1]), pb = b.map((p) => p[0] * axis[0] + p[1] * axis[1]);
    if (Math.min(Math.max(...pa), Math.max(...pb)) - Math.max(Math.min(...pa), Math.min(...pb)) <= tolerance) return false;
  }
  return true;
}
function inPoly(x, z, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    let [xi, zi] = poly[i], [xj, zj] = poly[j];
    if (zi > z != zj > z && x < (xj - xi) * (z - zi) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}
function allowed(x, z, layout) {
  return inPoly(x, z, layout.floor) || x >= 0.445 && x <= 4.22 && z >= 8.175 && z <= 11.9 || x >= 0.05 && x <= 5.4 && z >= -1.55 && z <= 0.2 || x >= 3.15 && x <= 5.4 && z >= 0 && z <= 1.55;
}
function obstacle(x, z, w, d, h = 2.65) {
  return { o: { x, z, y: 0, rotation: 0 }, p: { dims: [w, d, h] } };
}
function obstacles(style, layout) {
  let obs = layout.walls.map(([x1, z1, x2, z2, t]) => ({ o: { x: (x1 + x2) / 2, z: (z1 + z2) / 2, y: 0, rotation: -Math.atan2(z2 - z1, x2 - x1) * 180 / Math.PI }, p: { dims: [Math.hypot(x2 - x1, z2 - z1), t, 2.65] } }));
  obs.push(...style === "C" ? [obstacle(0.515, 3.75, 0.6, 1.9, 0.91), obstacle(2.4, 3.1, 0.6, 0.6, 2)] : [obstacle(0.515, 2.525, 0.6, 1.85, 0.91), obstacle(0.515, 1.3, 0.6, 0.6, 2)]);
  obs.push(obstacle(7.03, 5.21, 0.58, 0.56, 0.9), obstacle(5.91, 7.55, 0.44, 0.65, 0.8), ...layout.bathTub ? [] : [obstacle(6.9, 7.675, 0.9, 1, 0.12)]);
  return obs;
}
function height(p) {
  return p.kind === "bed" || p.kind === "guest" ? Math.max(0.65, p.dims[2]) : p.dims[2];
}
function issue(o, all, products, layout, obs) {
  const p = products[o.pid];
  if (!p) return "\u672A\u77E5\u5546\u54C1";
  if (!Number.isFinite(o.x + o.y + o.z + o.rotation) || o.y < 0 || o.y > 2.6) return "\u9AD8\u5EA6\u5E94\u5728 0\u20132.6 m \u4E4B\u95F4";
  const cs = corners(o, p);
  if (cs.some(([x, z]) => !allowed(x, z, layout))) return "\u5BB6\u5177\u8D85\u51FA\u53EF\u7528\u5730\u9762";
  if (["mirror", "art"].includes(p.kind)) return null;
  for (const b of obs) {
    if (o.y >= b.p.dims[2] - 0.01) continue;
    if (overlap(cs, corners(b.o, b.p), 0.015)) return "\u4E0E\u5899\u4F53\u6216\u56FA\u5B9A\u8BBE\u65BD\u91CD\u53E0";
  }
  if (p.kind === "rug") return null;
  for (const other of all) {
    if (other.id === o.id) continue;
    let q = products[other.pid];
    if (!q || ["rug", "mirror", "art"].includes(q.kind)) continue;
    if (o.y + height(p) <= other.y + 0.015 || other.y + height(q) <= o.y + 0.015) continue;
    if (overlap(cs, corners(other, q), 0.015)) return `\u4E0E${q.category}\u91CD\u53E0`;
  }
  return null;
}
export {
  allowed,
  corners,
  height,
  inPoly,
  issue,
  obstacle,
  obstacles,
  overlap
};
