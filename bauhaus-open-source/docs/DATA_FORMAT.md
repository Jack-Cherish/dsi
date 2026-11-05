# 数据、模型和坐标

`public/data.json` 是成品数据源。顶层包含 `catalog`、`layout`、`finishes`、`roomFinishes` 及历史预算/覆盖记录。当前版本号字段是制作阶段信息，不是 npm 包版本。

## 商品 catalog

| 字段 | 作用 |
| --- | --- |
| id / sourceKey | 内部主键／原清单或 SKU 标识 |
| name / brand / seller / brandIntro | 名称、品牌、店铺、介绍 |
| category / categoryKey / room / group | UI 分组与筛选，替换限于同类别键 |
| kind / slot / modelV5 | 通用模型种类、旧候选序号、v5 专用分支 |
| dims | [宽, 深, 高]，米；不是 Three.js 的 [x,y,z] 顺序 |
| sizeText / estimated | 原规格说明、是否包含暂定尺寸 |
| mount | floor / wall / ceiling / surface |
| photo / photoSource / photoStatus | 本地图片、原图 URL（如有）、证据说明 |
| price / url / brandIntro+modelNote / review | 界面的价格／跳转链接／产品介绍／评论情况四栏 |
| modelNote / status / note | 模型近似、候选状态及选品限制 |
| supportHeight / drapeOffset | 某些床架/床品支撑与下垂偏移，不能套用于所有家具 |

`price` 为保留原语义的历史字符串，不是可直接汇总的实时数字；预算字段也是历史规划记录。若做自动预算，需要先独立整理币种、数量、规格和优惠条件。

## layout

```js
walls: [[x1, z1, x2, z2, thickness], ...]
windows: [{x1, z1, x2, z2}, ...]
doors: [{x1, z1, x2, z2}, ...]
floor: [[x, z], ...] // 主体多边形，院子另有定制逻辑
placed: [{id, pid, x, y, z, rotation}, ...]
bathTub: true // 当前浴室有独立浴缸，移除旧淋浴底座
```

X/Z 为地面坐标，Y 向上，单位米。`rotation` 为绕 Y 轴的角度。`product(p,M)` 将模型底部归到 Y=0，包络对应 `dims`，最终正面朝 +Z。`placed.id` 是实例 ID，`placed.pid` 引用商品 ID；同一商品可以有多个实例。

`house.js` 还包含厨房、窗台、院墙、门扇等固定陈设；`collision.js` 与 `finishes.js` 有本户型边界/分房逻辑。换户型必须一起检查这些文件。

## 保存与导入

导出格式：

```json
{"format":"bauhaus-layout-v2","items":[],"finishes":{},"created":"ISO时间"}
```

`items` 结构同 `placed`，`finishes` 为房间键到十六进制颜色。支持 v1/v2 布局导入；最多 500 个实例，限制有限坐标与合法商品 ID。导入方案不是导入商品库。页面保留原 localStorage 键，但不同端口、浏览器和 file URL 的存储互不保证共享。

## 添加专用模型

1. 查看商品图片及规格；新增稳定 `id` 与 `photo`。
2. 在 `photo-models-v5.js` 或单独新模块写模型函数，并接入 `objects.js`。
3. 避免把用于展示的装饰物算作商品随附物；记录组合与推定部分。
4. `npm run check` 通过包络检查后再视觉对照。归一化会掩盖某些比例错误，不能用测试结果替代看图。
5. 新建样板实例，明确安装高度与朝向，检查通路、门扇与台面支撑。

