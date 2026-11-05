import * as T from "three";
import { box, rod, batch } from "./objects.js";
const roomNames = { public: "\u7384\u5173\u4E0E\u53A8\u9910", study: "\u4E66\u623F", bedroom: "\u4E3B\u5367", living: "\u8D77\u5C45\u5BA4", dry: "\u5E72\u533A", bath: "\u6D74\u5BA4" };
const defaults = { public: "#b9a88a", study: "#664737", bedroom: "#aaa08a", living: "#76523c", dry: "#a5987d", bath: "#1c2d4d" };
function roomAt(x, z) {
  if (x >= 5.49 && z >= 4.814) return z >= 5.995 ? "bath" : "dry";
  if (x >= 3.01 && x <= 5.85 && z >= 1.654 && z <= 3.954) return "study";
  if (z >= 5 && z <= 8.28) return x < 3.47 ? "bedroom" : "living";
  return "public";
}
function decorateWalls(root, layout, M) {
  const groups = {}, mats = {};
  for (const key of Object.keys(defaults)) {
    const mat = (key === "bath" ? M.navy : M.wall).clone();
    mat.color.set(defaults[key]);
    mats[key] = mat;
    groups[key] = new T.Group();
    root.add(groups[key]);
  }
  const inside = (x, z) => {
    let v = false, p = layout.floor;
    for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
      const [a, b] = p[i], [c, d] = p[j];
      if (b > z != d > z && x < (c - a) * (z - b) / (d - b) + a) v = !v;
    }
    return v;
  };
  const skin = (a, b, t, y, h) => {
    const dx = b[0] - a[0], dz = b[1] - a[1], l = Math.hypot(dx, dz), nx = -dz / l, nz = dx / l;
    const cuts = Math.ceil(l / 0.08);
    for (let side of [-1, 1]) for (let i = 0; i < cuts; i++) {
      const f = (i + 0.5) / cuts, x = a[0] + dx * f + nx * (t / 2 + 4e-3) * side, z = a[1] + dz * f + nz * (t / 2 + 4e-3) * side;
      if (!inside(x + nx * 0.015 * side, z + nz * 0.015 * side)) continue;
      const key = roomAt(x + nx * 0.03 * side, z + nz * 0.03 * side);
      const mesh = new T.Mesh(new T.PlaneGeometry(l / cuts + 1e-3, h), mats[key]);
      mesh.position.set(x, y + h / 2, z);
      mesh.rotation.y = Math.atan2(nx * side, nz * side);
      groups[key].add(mesh);
    }
  };
  for (const [x, z, a, b, t] of layout.walls) skin([x, z], [a, b], t, 0, 2.7);
  for (const w of layout.windows) {
    skin([w.x1, w.z1], [w.x2, w.z2], 0.2, 0, 0.78);
    skin([w.x1, w.z1], [w.x2, w.z2], 0.2, 2.35, 0.35);
  }
  skin([2.41, 5], [3.21, 5], 0.12, 2.22, 0.48);
  skin([3.01, 3.954], [5.745, 3.954], 0.065, 2.42, 0.28);
  for (const door of layout.doors) skin([door.x1, door.z1], [door.x2, door.z2], 0.2, 2.22, 0.48);
  for (const [key, g] of Object.entries(groups)) {
    batch(g);
    g.traverse((m) => {
      if (m.isMesh) {
        m.userData.roomFinish = key;
        m.castShadow = false;
        m.receiveShadow = true;
      }
    });
  }
  const tileGroup = groups.bath;
  tileGroup.traverse((m) => {
    if (!m.isMesh) return;
    m.material.onBeforeCompile = (s) => {
      s.vertexShader = s.vertexShader.replace("#include <common>", "#include <common>\nvarying vec3 wallWorld;").replace("#include <worldpos_vertex>", "#include <worldpos_vertex>\nwallWorld=(modelMatrix*vec4(transformed,1.)).xyz;");
      s.fragmentShader = s.fragmentShader.replace("#include <common>", "#include <common>\nvarying vec3 wallWorld;").replace("#include <color_fragment>", "#include <color_fragment>\nfloat vx=min(fract(wallWorld.x/0.15),1.-fract(wallWorld.x/0.15));float vz=min(fract(wallWorld.z/0.15),1.-fract(wallWorld.z/0.15));float vy=min(fract(wallWorld.y/0.30),1.-fract(wallWorld.y/0.30));float vertical=abs(cross(dFdx(wallWorld),dFdy(wallWorld)).x)>abs(cross(dFdx(wallWorld),dFdy(wallWorld)).z)?vz:vx;float line=1.-smoothstep(.008,.018,min(vertical,vy));diffuseColor.rgb=mix(diffuseColor.rgb,vec3(.11,.12,.15),line*.65);");
    };
  });
  const doors = [];
  const makeDoor = (label, x, z, w, sign, angle) => {
    const pivot = new T.Group();
    pivot.position.set(x, 0, z);
    root.add(pivot);
    box(pivot, w, 2.17, 0.037, sign * w / 2, 1.085, 0, M.darkWood, 8e-3);
    for (let y of [0.18, 1.12, 2.01]) box(pivot, w - 0.09, 0.015, 0.014, sign * w / 2, y, -0.025, M.wood, 3e-3);
    rod(pivot, [sign * (w - 0.1), 1.02, -0.045], [sign * (w - 0.23), 1.02, -0.045], 8e-3, M.brushed);
    pivot.rotation.y = angle;
    pivot.userData = { door: true, label, openAngle: angle, closedAngle: 0, open: true };
    doors.push(pivot);
  };
  makeDoor("\u4E3B\u5367\u95E8", 2.41, 5, 0.78, 1, -Math.PI / 2);
  makeDoor("\u6D74\u5BA4\u95E8", 6.7, 5.995, 0.77, -1, Math.PI / 2);
  makeDoor("\u5165\u6237\u95E8", 0.4, 0.1, 0.78, 1, Math.PI / 2);
  makeDoor("\u540E\u9662\u95E8", 4.27, 8.2775, 0.7, -1, -Math.PI / 2);
  root.userData.roomMaterials = mats;
  root.userData.finishGroups = groups;
  root.userData.doors = doors;
  return mats;
}
export {
  decorateWalls,
  defaults,
  roomAt,
  roomNames
};
