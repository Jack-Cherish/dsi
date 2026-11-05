import * as T from "three";
const loaded = {};
function canvasTexture(w, h, draw) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new T.CanvasTexture(c);
  t.colorSpace = T.SRGBColorSpace;
  t.wrapS = t.wrapT = T.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}
async function materials() {
  const names = ["oak_veneer_01_diff", "oak_veneer_01_nor_gl", "oak_veneer_01_rough", "wood_floor_diff", "wood_floor_nor_gl", "wood_floor_rough", "fabric_pattern_07_nor_gl", "plastered_wall_03_nor_gl"];
  await Promise.all(names.map(async (n) => {
    const t = await new T.TextureLoader().loadAsync("assets/" + n + ".jpg");
    t.colorSpace = n.endsWith("diff") ? T.SRGBColorSpace : T.NoColorSpace;
    t.wrapS = t.wrapT = T.RepeatWrapping;
    t.anisotropy = 16;
    loaded[n] = t;
  }));
  const map = (n, x = 1, y = 1) => {
    let t = loaded[n].clone();
    t.repeat.set(x, y);
    return t;
  };
  const M = {};
  M.wall = new T.MeshStandardMaterial({ color: "#b9a88a", roughness: 0.91, normalMap: map("plastered_wall_03_nor_gl", 2, 2), normalScale: new T.Vector2(0.035, 0.035) });
  M.ceiling = new T.MeshStandardMaterial({ color: "#e8e2d5", roughness: 0.98 });
  M.wood = new T.MeshPhysicalMaterial({ color: "#b58d67", map: map("oak_veneer_01_diff", 0.85, 1), normalMap: map("oak_veneer_01_nor_gl", 0.85, 1), normalScale: new T.Vector2(0.14, 0.14), roughnessMap: map("oak_veneer_01_rough", 0.85, 1), roughness: 0.77, clearcoat: 0.13, clearcoatRoughness: 0.38 });
  M.floor = new T.MeshPhysicalMaterial({ color: "#ba8f66", map: map("wood_floor_diff", 1, 1), normalMap: map("wood_floor_nor_gl"), normalScale: new T.Vector2(0.18, 0.18), roughnessMap: map("wood_floor_rough"), roughness: 0.65, clearcoat: 0.17, clearcoatRoughness: 0.4 });
  M.darkWood = M.wood.clone();
  M.darkWood.color.set("#4e3d2e");
  M.darkWood.roughness = 0.7;
  M.black = new T.MeshPhysicalMaterial({ color: "#222522", roughness: 0.34, metalness: 0.18, clearcoat: 0.2 });
  M.chrome = new T.MeshPhysicalMaterial({ color: "#d8dfe1", roughness: 0.2, metalness: 1 });
  M.brushed = M.chrome.clone();
  M.brushed.roughness = 0.38;
  M.white = new T.MeshStandardMaterial({ color: "#deded6", roughness: 0.43 });
  M.grout = new T.MeshStandardMaterial({ color: "#b6aaa0", roughness: 0.94 });
  M.tile = new T.MeshPhysicalMaterial({ color: "#13323a", roughness: 0.24, metalness: 0.03, clearcoat: 0.5, clearcoatRoughness: 0.16 });
  M.stone = new T.MeshStandardMaterial({ color: "#dedbd0", roughness: 0.38 });
  M.stone.map = canvasTexture(1024, 1024, (c, w, h) => {
    c.fillStyle = "#e9e6de";
    c.fillRect(0, 0, w, h);
    for (let k = 0; k < 9; k++) {
      c.beginPath();
      c.moveTo(k * 175 - 300, 0);
      for (let y = 0; y <= h; y += 16) c.lineTo(k * 175 - 300 + y * 0.55 + Math.sin(y * 0.017 + k) * 22 + Math.sin(y * 0.047) * 7, y);
      c.strokeStyle = k % 2 ? "#b8b5b0" : "#d0ccc5";
      c.lineWidth = k % 3 + 1;
      c.stroke();
    }
  });
  const weave = canvasTexture(512, 512, (c, w, h) => {
    c.fillStyle = "#d8d8d8";
    c.fillRect(0, 0, w, h);
    let s = 83;
    for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
      s = s * 1664525 + 1013904223 >>> 0;
      const a = 190 + s % 45;
      c.fillStyle = `rgb(${a},${a},${a})`;
      c.fillRect(x, y, 1, 2);
    }
    c.globalAlpha = 0.15;
    c.fillStyle = "#ffffff";
    for (let y = 0; y < h; y += 4) c.fillRect(0, y, w, 1);
  });
  weave.repeat.set(3, 3);
  const fabric = (color, roughness = 0.94) => new T.MeshPhysicalMaterial({ color, map: weave, roughness, normalMap: map("fabric_pattern_07_nor_gl", 5, 5), normalScale: new T.Vector2(0.13, 0.13), sheen: 0.45, sheenRoughness: 0.9, sheenColor: new T.Color(color).lerp(new T.Color("white"), 0.15) });
  M.orange = fabric("#a95225", 0.84);
  M.green = fabric("#174a3c", 0.91);
  M.gray = fabric("#4b5151");
  M.linen = fabric("#b9b7a9");
  M.cream = fabric("#dfd8c5");
  M.blue = fabric("#253e50");
  M.leather = new T.MeshPhysicalMaterial({ color: "#9a4d25", roughness: 0.51, normalMap: map("fabric_pattern_07_nor_gl", 12, 12), normalScale: new T.Vector2(0.035, 0.035), clearcoat: 0.13 });
  M.rug = new T.MeshStandardMaterial({ color: "#304958", roughness: 1, normalMap: map("fabric_pattern_07_nor_gl", 8, 12), normalScale: new T.Vector2(0.32, 0.32) });
  M.rug.map = canvasTexture(512, 512, (c, w, h) => {
    c.fillStyle = "#adb9b9";
    c.fillRect(0, 0, w, h);
    let s = 99;
    const rnd = () => {
      s = s * 1664525 + 1013904223 >>> 0;
      return s / 4294967296;
    };
    for (let i = 0; i < 26e3; i++) {
      c.fillStyle = rnd() > 0.5 ? "#7c9098" : "#d0d1c5";
      c.globalAlpha = 0.25 + rnd() * 0.5;
      c.fillRect(rnd() * w, rnd() * h, 1, 2);
    }
    c.globalAlpha = 1;
  });
  M.rug.map.repeat.set(3, 4);
  M.glass = new T.MeshPhysicalMaterial({ color: "#c5d4cd", roughness: 0.08, transparent: true, opacity: 0.15, metalness: 0.12, depthWrite: false, side: T.DoubleSide });
  M.earth = new T.MeshStandardMaterial({ color: "#25251b", roughness: 1 });
  M.leaf = new T.MeshPhysicalMaterial({ color: "#2d5129", roughness: 0.6, side: T.DoubleSide, clearcoat: 0.15 });
  M.red = new T.MeshStandardMaterial({ color: "#994333", roughness: 0.5 });
  M.yellow = new T.MeshStandardMaterial({ color: "#ac883b", roughness: 0.65 });
  M.paper = new T.MeshStandardMaterial({ color: "#ddd8cb", roughness: 1 });
  M.glow = new T.MeshBasicMaterial({ color: "#ffdf9f" });
  M.amber = new T.MeshPhysicalMaterial({ color: "#ad4c0f", emissive: "#d87820", emissiveIntensity: 0.75, roughness: 0.18, metalness: 0.12, clearcoat: 1 });
  M.olive = M.wall.clone();
  M.olive.color.set("#50583b");
  M.olive.roughness = 0.63;
  M.brown = M.wall.clone();
  M.brown.color.set("#664737");
  M.bedwall = M.wall.clone();
  M.bedwall.color.set("#686348");
  M.navy = M.tile.clone();
  M.navy.color.set("#1c2d4d");
  M.wood.color.set("#a5764c");
  M.floor.color.set("#98704c");
  M.rug.map = canvasTexture(512, 512, (c, w, h) => {
    c.fillStyle = "#243d58";
    c.fillRect(0, 0, w, h);
    c.fillStyle = "#cfb995";
    for (let i = 0; i < 4; i++) {
      c.fillRect(i * 128 + 22, 0, 12, h);
      c.fillRect(0, i * 128 + 22, w, 12);
      c.fillRect(i * 128 + 48, 0, 5, h);
      c.fillRect(0, i * 128 + 48, w, 5);
    }
    c.fillStyle = "#a67946";
    for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) c.fillRect(x * 128 + 21, y * 128 + 21, 30, 30);
  });
  M.rug.map.repeat.set(1, 1);
  return M;
}
export {
  materials
};
