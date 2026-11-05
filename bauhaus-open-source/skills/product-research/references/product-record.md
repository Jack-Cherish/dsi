# 商品记录格式

现有 `catalog` 最小新增记录示例（示例是结构模板，不是真实在售商品）：

```json
{
  "id": "B_NEW_SKU",
  "sourceKey": "JD_SKU",
  "style": "B",
  "slot": 0,
  "name": "商品名与已选规格",
  "brand": "已核品牌；未知时明确未核",
  "brandIntro": "与产品直接相关的品牌/材质/配置说明",
  "seller": "店铺名",
  "room": "起居",
  "group": "起居",
  "category": "休闲椅",
  "categoryKey": "NEW_CHAIR",
  "kind": "armchair",
  "mount": "floor",
  "dims": [0.82, 0.72, 0.72],
  "estimated": true,
  "sizeText": "说明来源、单位，以及哪些维度待核",
  "price": "YYYY-MM-DD 页面所示金额与优惠条件；非结算报价",
  "url": "https://item.jd.com/实际SKU.html",
  "review": "未读取评论正文",
  "photo": "products/NEW_SKU.jpg",
  "photoSource": "原图片URL（确实取得时填写）",
  "photoStatus": "已查看正面；背面待核",
  "modelNote": "图片近似模型；列出未公开细节",
  "status": "待核规格",
  "note": "与参考图相似，未证实同款"
}
```

辅助证据保存到 `docs/research/<批次>.json`，不要求塞进 UI 文案：

```json
{
  "productId": "B_NEW_SKU",
  "observedAt": "实际采集时间，含时区",
  "evidenceLevel": "product-page",
  "matchLevel": "similar",
  "selectedVariant": "颜色/尺寸/单件或组合",
  "queries": ["实际使用的查询词"],
  "visibleFeatures": ["钢管外框", "三块独立厚软垫"],
  "modelChecks": ["检查扶手前沿", "脚架离地高度"],
  "unknowns": ["背面连接方式", "皮革成分"],
  "reviewEvidence": {"bodyRead": false, "sampleCount": 0},
  "imageRights": "第三方商品图；再分发许可未核"
}
```

删除无法证明的字段比填入猜测更好。商品时间必须是实际采集时间；不能把打包/写文档的时间作为旧报价的核验时间。
