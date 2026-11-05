import { DefaultLoadingManager } from "three";
function assetURL(path) {
  return globalThis.__BAUHAUS_ASSETS__?.[path] || path;
}
DefaultLoadingManager.setURLModifier(assetURL);
async function loadData() {
  const embedded = globalThis.__BAUHAUS_DATA__;
  const data = embedded ? structuredClone(embedded) : await fetch("data.json").then((r) => {
    if (!r.ok) throw new Error(`\u6570\u636E\u52A0\u8F7D\u5931\u8D25\uFF1AHTTP ${r.status}`);
    return r.json();
  });
  for (const product of [...data.catalog, ...data.finishes || []]) {
    if (product.photo) product.photo = assetURL(product.photo);
    if (product.url && !/^https?:\/\//.test(product.url)) product.url = assetURL(product.url);
  }
  return data;
}
export {
  assetURL,
  loadData
};
