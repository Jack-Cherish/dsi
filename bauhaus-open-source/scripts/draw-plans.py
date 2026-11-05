from pathlib import Path
import html, json

OUT=Path(__file__).resolve().parents[1]/'docs'/'plans'
OUT.mkdir(parents=True,exist_ok=True)
FONT='Microsoft YaHei, Noto Sans CJK SC, sans-serif'
S=.094
OX,OY=130,470
parts=[]
PLAN_GROUPS={}
def add(s): parts.append(s)
def rect(x,y,w,h,fill='none',stroke='none',sw=12,rx=0,cls=''):
    add(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def line(x1,y1,x2,y2,color='#383b39',w=12,dash=''):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" stroke-dasharray="{dash}"/>')
def path(d,stroke='#555',w=12,fill='none',dash='',extra=''):
    add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" stroke-dasharray="{dash}" {extra}/>')
def text(x,y,t,size=210,color='#303934',anchor='middle',weight=400):
    add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(t)}</text>')
def circle(x,y,r,fill='none',stroke='#777',sw=12):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def wall(x1,y1,x2,y2): line(x1,y1,x2,y2,'#525953',120)
def zone(x,y,w,h,fill): rect(x,y,w,h,fill)
def label(x,y,name,sub=''):
    text(x,y,name,245,weight=600)
    if sub:text(x,y+260,sub,155,'#647069')
def slider(x1,y1,x2,y2):
    line(x1,y1,x2,y2,'#fff',132)
    if x1==x2:
        line(x1-35,y1,x2-35,y2,'#6c827c',20);line(x1+35,y1+(y2-y1)/2,x2+35,y2+(y2-y1)/2,'#6c827c',20)
    else:
        line(x1,y1-35,x2,y2-35,'#6c827c',20);line(x1+(x2-x1)/2,y1+35,x2+(x2-x1)/2,y2+35,'#6c827c',20)
def window(x1,y1,x2,y2):
    line(x1,y1,x2,y2,'#fafdfd',100)
    if x1==x2:
        for d in [-37,0,37]:line(x1+d,y1,x2+d,y2,'#829997',12)
    else:
        for d in [-37,0,37]:line(x1,y1+d,x2,y2+d,'#829997',12)
def bed(x,y,w,h,vertical=True,fill='#e0d4bf'):
    rect(x,y,w,h,fill,'#938b7f',16,40)
    if vertical:
        rect(x+55,y+60,w/2-80,330,'#f5f1e9','#aaa190',10,35);rect(x+w/2+25,y+60,w/2-80,330,'#f5f1e9','#aaa190',10,35)
        line(x,y+460,x+w,y+460,'#aaa190')
    else:
        rect(x+60,y+55,330,h/2-80,'#f5f1e9','#aaa190',10,35);rect(x+60,y+h/2+25,330,h/2-80,'#f5f1e9','#aaa190',10,35)
        line(x+460,y,x+460,y+h,'#aaa190')
def sofa(x,y,w,h,orient='right',color='#c3b8a6'):
    rect(x,y,w,h,color,'#8f8a80',15,80)
    if orient=='right':
        rect(x+50,y+100,160,h-200,'#ded7ca','#8f8a80',10,30)
        line(x+230,y+h/2,x+w-80,y+h/2,'#8f8a80')
    else:
        rect(x+w-210,y+100,160,h-200,'#ded7ca','#8f8a80',10,30)
        line(x+80,y+h/2,x+w-230,y+h/2,'#8f8a80')
def wardrobe(x,y,w,h):
    rect(x,y,w,h,'#d9d1c2','#9f9889',14)
    for xx in range(int(x+500),int(x+w),500):line(xx,y,xx,y+h,'#b0a897',9)
    text(x+w/2,y+h*.65,'衣柜',145,'#736d62')
def kitchen(kind='front'):
    if kind=='front':
        rect(215,1000,600,2450,'#ddd5c7','#8f958e',16)
        line(215,1600,815,1600,'#8f958e');line(215,2400,815,2400,'#8f958e')
        text(515,1350,'冰箱',140)
        rect(300,1770,430,450,'#f7f8f4','#8f958e',12,40)
        for yy in [2790,3120]:circle(515,yy,115,'none','#606b65',16)
        label(1530,2750,'一字厨房','台面深 600')
    else:
        rect(215,2800,600,1900,'#ded4c0','#969084',16)
        rect(300,2930,430,450,'#faf8ee','#969084',12,35)
        for yy in [4000,4400]:circle(515,yy,110,'none','#79786b',16)
        rect(2100,2800,600,600,'#ded4c0','#969084',16)
        text(2400,3140,'冰箱',140)
        label(1500,3950,'半开放厨房','操作 / 通行共用开敞区')
def banquet(x,y):
    rect(x,y,450,1200,'#c6bba6','#999183',14,35)
    rect(x+635,y,700,1200,'#e9dfca','#a4987f',16,60)
    for yy in [y+150,y+700]:rect(x+1340,yy,420,400,'#d2c6ae','#9d947f',12,55)
def desk(x,y,w=1300):
    rect(x,y,w,550,'#dbcdb5','#aa997b',14)
    rect(x+160,y+60,500,270,'#e7ece9','#8b9890',10,20)
    rect(x+w/2-180,y+620,360,360,'#e7e0d2','#a49780',12,70)
def guestbed(x,y):
    rect(x,y,1900,900,'#e4ddcf','#aaa08e',16,55)
    rect(x+70,y+80,300,700,'#f6f1e6','#aaa08e',10,35)
    text(x+1100,y+530,'客卧 900 × 1900',135,'#857a67')
def arrow(d,color='#3c8e87'):
    path(d,color,36,dash='95 70',extra='marker-end="url(#arr)"')
def dim(x1,y1,x2,y2,labeltext):
    line(x1,y1,x2,y2,'#8b9690',10)
    if y1==y2:
        line(x1,y1-65,x1,y1+65,'#8b9690',10);line(x2,y2-65,x2,y2+65,'#8b9690',10)
        text((x1+x2)/2,y1-85,labeltext,170,'#6f7b74')
    else:
        line(x1-65,y1,x1+65,y1,'#8b9690',10);line(x2-65,y2,x2+65,y2,'#8b9690',10)
        add(f'<text transform="translate({x1-105},{(y1+y2)/2}) rotate(-90)" text-anchor="middle" font-size="170" fill="#6f7b74">{html.escape(labeltext)}</text>')
def bathroom():
    zone(5550,4914,1800,3261,'#dce8e7')
    wall(5490,4914,5490,5115);wall(5490,5915,5490,8175)
    wall(5490,5995,5900,5995);wall(6700,5995,7350,5995);slider(5900,5995,6700,5995)
    rect(6750,4914,600,600,'#eff5f3','#819794',15)
    rect(6840,5040,410,330,'#fafcfb','#819794',12,80)
    label(6310,5660,'转折干区')
    label(6440,6500,'卫生间','约 1800 × 2120')
    rect(6450,7175,900,1000,'#c5dddc','#7c9693',15)
    line(6450,7175,7350,8175,'#9cb8b3',10)
    text(6900,7770,'淋浴',150)
    rect(5700,7275,420,630,'#fbfcf9','#8b9990',14,165)
    rect(5690,7175,440,190,'#edf2ed','#8b9990',12,20)
    text(6370,8510,'右后侧湿区：排污 / 排风待核实',155,'#6b7770')
def frontmaster():
    zone(3070,1754,2675,3036,'#eee5d5')
    wall(3010,1754,3010,3750);wall(3010,4550,3010,4850);slider(3010,3750,3010,4550)
    wall(3010,4850,5745,4850)
    bed(3220,2150,2000,1500,False)
    wardrobe(4050,4190,1695,600)
    label(4420,1980,'主卧')
    text(4400,3990,'约 2675 × 3035',150,'#7e7565')
def rearstudy(width=2500,soft=False):
    right=610+width
    zone(610,5960,width,2215,'#e6eadf')
    wall(right+60,5900,right+60,8175)
    if soft:
        line(610,5900,right+60,5900,'#7c9282',35,dash='120 50')
        slider(right-800,5900,right,5900)
    else:
        wall(610,5900,right-800,5900);wall(right,5900,right+60,5900);slider(right-800,5900,right,5900)
    desk(710,5960,1200)
    guestbed(730,7175)
    text(2200,6850,'茶室 / 书房' if soft else '书房兼客卧',185,weight=600)
    text(2200,7060,f'{width} × 2215',140,'#6e7c68')
def base():
    rect(0,-1600,5450,1600,'#eef1e7')
    rect(3150,0,2300,1554,'#eef1e7')
    path('M 0 0 L 0 -1600 L 3950 -1600 M 4750 -1600 L 5450 -1600 L 5450 1554','#89978d',65)
    path('M 3950 -1600 L 4750 -2400 M 3950 -1600 A 800 800 0 0 1 4750 -2400','#89978d',12)
    label(2600,-700,'前院','保留开放院落，不计室内面积')
    rect(395,8380,3875,3570,'#edf0e6','#8e9b91',55)
    label(2350,10000,'后院','保留庭院与室内的直接联系')
    for x,y,r in [(850,11100,260),(1350,11250,170),(3700,10800,260),(3850,9000,170)]:
        circle(x,y,r,'#d2ddc5','#a5b59b',12)
    # Approximate outer/inner boundary reconstructed from original raster and horizontal dimensions.
    path('M 0 0 L 3150 0 L 3150 1554 L 5940 1554 L 5940 4714 L 7510 4714 L 7510 8380 L 395 8380 L 395 5130 L 0 5130 Z',fill='#56605a',stroke='none')
    path('M 215 200 L 2950 200 L 2950 1754 L 5745 1754 L 5745 4914 L 7350 4914 L 7350 8175 L 610 8175 L 610 4790 L 215 4790 Z',fill='#f7f5ef',stroke='none')
def outline_details():
    # Entry and existing openings remain recognizable.
    window(1490,100,2950,100);window(3050,350,3050,1250)
    window(3960,1654,5440,1654);window(610,8277,1490,8277)
    # Rear right existing window converted to a door, explicitly a proposal.
    line(3555,8277,4270,8277,'#fff',210)
    path('M 4270 8277 L 4270 8992 M 3555 8277 A 715 715 0 0 0 4270 8992','#bc8455',20)
    line(3555,8180,4270,8180,'#bc8455',25,dash='60 45')
    text(4440,9200,'原窗位拟改院门',160,'#a36c43','start')
    path('M 4320 9040 L 4650 9040 L 4650 8970','#b88a65',13)
    line(400,100,1200,100,'#fff',210)
    path('M 400 100 L 400 900 M 1200 100 A 800 800 0 0 1 400 900','#6c7b71',18)
    text(1730,-140,'原入户位置',155,'#566b61')
    rect(1490,200,960,350,'#d9d1c2','#9f9889',14)
    text(1970,440,'低鞋柜',140,'#736d62')
    text(2100,1050,'短玄关',155,'#647168')
    dim(0,-2800,7510,-2800,'7510  ·  原图总宽')
    dim(-550,0,-550,8380,'8380  ·  原图左侧纵向标注')
    dim(215,12330,2950,12330,'2735  ·  原标注')
    dim(3510,12330,7350,12330,'3845  ·  原标注（取整映射）')
    text(3720,12750,'单位 mm ｜ 房间尺寸为方案拟定值 ｜ 上方为前院方向，非指北',155,'#6a756e')
def circulation(which):
    add('<g class="routes">')
    if which=='A':
        arrow('M 4350 -1730 L 4350 -1100 L 1000 -1100 L 1000 600 L 1500 1500 L 1450 2500 L 2450 3500 L 2450 5100 L 4550 5400 L 4550 7750 L 3900 7950 L 3900 9450')
        arrow('M 2450 4100 L 3250 4100');arrow('M 4550 5500 L 5800 5500')
    elif which=='B':
        arrow('M 4350 -1730 L 4350 -1100 L 1000 -1100 L 1000 600 L 1450 2500 L 2450 3500 L 2450 4450 L 4200 4450 L 4200 7850 L 3900 9450')
        arrow('M 4200 5500 L 5800 5500');arrow('M 2800 4450 L 2800 5300')
    else:
        arrow('M 4350 -1730 L 4350 -1100 L 1000 -1100 L 1000 600 L 1600 900 L 1600 2500 L 1450 3500 L 2350 4500 L 2350 5350 L 4900 5400 L 4900 7700 L 3900 7950 L 3900 9450')
        arrow('M 4900 5500 L 5800 5500');arrow('M 2350 4100 L 3250 4100')
    add('</g>')
DATA={
'A':dict(title='意式 · 前厨餐，后会客',accent='#7d715e',sub='完整卧室 + 独立书房｜用收纳界面组织空间',
notes=[('布局顺序','前院 → 短玄关 → 厨餐区','→ 后段客厅 → 后院'),('厨餐关系','一字厨房与卡座餐位连续布置；','买菜入门即收纳，端菜不穿客厅。'),('会客与休息','主卧借前院窗，书房借后院窗；','客厅留在后段，书房保持实体隔断。'),('卫浴处理','卫生间移至右后角；入口先进入','转折洗手区，与餐位不共墙。'),('风格落在布局上','整墙柜、窄框移门、清晰分区；','采用小体量双人沙发，避免硬塞岛台。'),('家具校核','主卧床 1500 × 2000，横向摆放；','客卧床 900 × 1900，书桌独立。')]),
'B':dict(title='包豪斯 · 可开合的共享客厅',accent='#276e8b',sub='主卧后置 + 前侧弹性书房｜让空间随使用变化',
notes=[('布局顺序','前院 → 厨餐区 → 中央共享客厅','→ 后院；主卧在后院左侧。'),('最主要的改动','主卧与书房交换位置；前侧书房','可向客厅打开，扩大日常公共活动区。'),('厨餐关系','厨房紧接餐位，保留约 1 m','餐侧通行带，不做独立长走廊。'),('卫浴处理','右后侧干湿区保留转折入口；','从客厅侧抵达，不穿主卧和书房。'),('风格落在布局上','可移动家具、可开合玻璃隔断；','书房白天工作，客人留宿时闭合。'),('使用取舍','主卧更大，客厅本体偏紧凑；','舒适性依靠书房日常开放获得。')]),
'C':dict(title='侘寂 · 从前院起居到后院用餐',accent='#7d845d',sub='前客厅 + 后餐区｜书房兼茶室，减少固定隔断',
notes=[('布局顺序','前院 → 起居区 → 半开放厨房','→ 后侧餐区 / 茶室 → 后院'),('厨餐关系','厨房处于公共动线中段，餐位','沿后侧布置；厨房到餐桌转一次弯。'),('书房与庭院','后院侧书房兼茶室，用透光移门','开合；客人留宿时可独立关闭。'),('卫浴处理','餐桌与卫生间之间隔有通行带','及实墙，厕门朝独立干区转入。'),('风格落在布局上','前侧弱化电视中心，以坐谈为主；','后侧卡座用餐，茶室与庭院相连。'),('使用取舍','书房宽约 2200，适合单人客卧；','保留低家具和留白，避免厚重造型墙。')])
}
def build(which):
    global parts
    parts=[];d=DATA[which]
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1860" viewBox="0 0 1600 1860" style="font-family:{FONT}">')
    add('<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#3c8e87"/></marker></defs>')
    rect(0,0,1600,1860,'#fbfaf6')
    rect(60,55,66,54,d['accent'],rx=8);text(93,94,which,31,'#fff',weight=650)
    text(148,98,d['title'],38,anchor='start',weight=650)
    text(150,137,d['sub'],21,'#6b746b','start')
    text(1510,85,'布局研究 / 01',16,'#788277','end');text(1510,113,'概念分割 · 非施工放线图',16,'#788277','end')
    line(60,170,1540,170,'#d5d9d0',1)
    add(f'<g transform="translate({OX},{OY}) scale({S})">')
    plan_start=len(parts)
    base()
    add('<g class="furniture">')
    if which=='A':
        zone(215,2200,2735,2590,'#f3eee3')
        frontmaster();rearstudy();bathroom();kitchen()
        banquet(215,3500);label(1500,5100,'厨餐一体','餐侧主通行约 1000')
        sofa(3230,5850,800,1400);rect(5250,6500,180,1300,'#777d75')
        label(4400,5100,'后段客厅');text(4350,8050,'通后院',140,'#6b7b70')
    elif which=='B':
        zone(3070,1754,2675,2200,'#e5eadd')
        wall(3010,1754,3010,3954)
        line(3010,4014,5745,4014,'#628b97',34,dash='150 60')
        text(4430,4190,'可开合隔断',135,'#527582')
        rect(3070,1854,550,1100,'#dbcdb5','#aa997b',14)
        rect(3150,1984,300,450,'#e7ece9','#8b9890',10,20)
        rect(3700,2000,360,360,'#e7e0d2','#a49780',12,70)
        # Single guest bed runs perpendicular to the front window; entry stays on its left.
        rect(4745,1854,900,1900,'#e4ddcf','#aaa08e',16,55)
        rect(4825,1924,740,300,'#f6f1e6','#aaa08e',10,35)
        text(5195,2850,'客卧',145,'#857a67')
        text(5195,3070,'900 × 1900',125,'#857a67')
        text(3935,3230,'书房兼客卧',205,weight=600)
        text(3935,3460,'约 2675 × 2200',145,'#647069')
        zone(610,5060,2800,3115,'#eee4d2')
        wall(3470,5000,3470,8175);wall(610,5000,2410,5000);wall(3210,5000,3470,5000);slider(2410,5000,3210,5000)
        wardrobe(610,5060,1550,600);bed(810,6200,2000,1500,False)
        text(2000,5900,'主卧',245,weight=600);text(2000,6100,'约 2800 × 3115',150,'#7e7565')
        kitchen();banquet(215,3450)
        bathroom();sofa(4630,6200,800,1550,'left','#adbac1');rect(3530,6400,150,1250,'#586971')
        label(4390,4930,'共享客厅','书房打开后联用')
    else:
        frontmaster();rearstudy(2200,True);bathroom()
        sofa(215,1000,800,1500,'right','#d5cdbb')
        label(1530,2650,'前院起居','坐谈 / 阅读')
        kitchen('middle')
        # Rear dining bay keeps a separate 900 mm route along the bathroom wall.
        rect(2930,6000,450,1200,'#d0c3a8','#a89a80',14,45)
        rect(3380,6000,700,1200,'#e6d9bd','#a89a80',16,65)
        for yy in [6110,6670]:rect(4100,yy,420,400,'#dbceb0','#a89a80',12,60)
        label(4040,7620,'后院餐区','餐侧通行目标约 900')
    add('</g>')
    outline_details();circulation(which)
    PLAN_GROUPS[which]=''.join(parts[plan_start:])
    add('</g>')
    # Right hand editorial notes.
    line(950,220,950,1645,'#dce0d7',1)
    text(990,245,'空间与动线',27,d['accent'],'start',600)
    yy=295
    for i,(h,a,b) in enumerate(d['notes'],1):
        text(990,yy,f'{i:02d}',17,d['accent'],'start',650)
        text(1030,yy,h,21,'#35443a','start',600)
        text(1030,yy+37,a,19,'#697568','start')
        text(1030,yy+69,b,19,'#697568','start')
        yy+=143
    text(990,1205,'图例',23,d['accent'],'start',600)
    for j,(c,t) in enumerate([('#eee5d5','主卧'),('#e6eadf','书房 / 客卧'),('#dce8e7','干区 / 卫浴')]):
        rect(990,1235+j*40,22,22,c,'#aab2a6',1,3);text(1027,1253+j*40,t,18,'#647065','start')
    line(990,1380,1045,1380,'#3c8e87',3,'8 5');text(1062,1387,'主要通行路径',18,'#647065','start')
    line(990,1425,1045,1425,'#525953',9);text(1062,1432,'拟新增隔墙',18,'#647065','start')
    line(990,1470,1045,1470,'#bc8455',3,'6 4');text(1062,1477,'原窗位拟改后院门',18,'#647065','start')
    text(990,1550,'读图约定',21,d['accent'],'start',600)
    text(990,1587,'未指定指北方向；前后以原图为准。',17,'#697568','start')
    text(990,1617,'黑色构件按设想取消；实拆需核实结构。',17,'#697568','start')
    line(60,1700,1540,1700,'#d5d9d0',1)
    text(70,1740,'尺寸与实施前提',22,'#405442','start',600)
    text(70,1777,'① 原图右侧分段合计约 8169，与左侧 8380 差约 211；本图按可辨轮廓和总宽重建，局部待复测。',18,'#717b70','start')
    text(70,1808,'② 后院门利用原右侧窗位，图示净口约 715，偏窄；需核实能否扩至约 800–900，不能直接按本图施工。',18,'#717b70','start')
    text(70,1839,'③ 厨卫排污点、排风、承重墙属性均未提供；三方案均以右后侧湿区可实施为前提，未将院落并入室内。',18,'#717b70','start')
    add('</svg>')
    return ''.join(parts)

for key in DATA:
    (OUT/f'方案{key}.svg').write_text(build(key),encoding='utf-8')

parts=[]
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="2100" height="1530" viewBox="0 0 2100 1530" style="font-family:{FONT}">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#3c8e87"/></marker></defs>')
rect(0,0,2100,1530,'#fbfaf6')
text(50,55,'同一户型 · 三种空间组织',33,anchor='start',weight=650)
text(2050,53,'1 主卧 + 1 书房兼客卧 ｜ 动线优先的概念分割',20,'#69756b','end')
titles=['A  意式：前厨餐，后会客','B  包豪斯：可开合的共享客厅','C  侘寂：前起居，后用餐']
subs=['主卧靠前院，书房靠后院；公共空间分区清晰。','交换主卧与书房，活动隔断扩展公共空间。','起居与餐区分置两端，书房兼茶室借后院光。']
for i,k in enumerate(DATA):
    x=80+i*680
    if i:line(x-60,95,x-60,1400,'#d8ddd3',1)
    text(x-10,120,titles[i],26,DATA[k]['accent'],'start',650)
    text(x-10,157,subs[i],17,'#69756b','start')
    add(f'<g transform="translate({x},425) scale(.075)">{PLAN_GROUPS[k]}</g>')
line(50,1420,2050,1420,'#d5d9d0',1)
text(50,1460,'总览仅作方案比较；单张详图包含尺寸、家具、动线及实施前提。原图纵向尺寸不闭合，需复测后深化。',21,'#5d6f5e','start')
text(50,1496,'共同待核实条件：右后侧湿区排污 / 排风、承重墙属性、原窗位改后院门。前后院均保留室外用途。',20,'#7e897c','start')
add('</svg>')
(OUT/'三方案总览.svg').write_text(''.join(parts),encoding='utf-8')

svgs={k:(OUT/f'方案{k}.svg').read_text(encoding='utf-8') for k in DATA}
doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三套户型布局方案</title><style>body{margin:0;background:#e9ebe5;color:#344238;font-family:"Microsoft YaHei",sans-serif}header{position:sticky;top:0;background:#f9faf6ee;backdrop-filter:blur(10px);padding:16px 24px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:3;border-bottom:1px solid #d4d9cf}button{padding:10px 17px;border:1px solid #c3cbbd;border-radius:8px;background:white;color:#40503e;cursor:pointer}button.active{background:#435b4a;color:white}label{margin-left:auto;font-size:14px;cursor:pointer}main{max-width:1500px;margin:22px auto}section{display:none}section.active{display:block}svg{width:100%;height:auto;box-shadow:0 5px 30px #35412f19}body.hide-routes .routes{display:none}.intro{padding:20px 30px;max-width:1100px;margin:auto;line-height:1.8}.intro p{margin:6px 0}a{color:#456d59}</style><header><strong>三套布局研究</strong>'''
for k in DATA:doc+=f'<button onclick="show(\'{k}\')" id="b{k}" class="{"active" if k=="A" else ""}">{k} · {DATA[k]["title"].split(" · ")[0]}</button>'
doc+='''<label><input type="checkbox" checked onchange="document.body.classList.toggle('hide-routes',!this.checked)"> 显示动线</label></header><div class="intro"><p><b>一间主卧 + 一间书房兼客卧，保留前后院。</b> 三套均先处理厨餐联系、院落通行与卫浴遮挡，再采用不同的公共空间组织。</p><p>这是比例布局研究。原图纵向尺寸不闭合，未获得排污、承重及门窗实测资料；右后侧卫生间和后院窗改门是共同的待核实条件。箭头代表通行顺序，家具及过道需随复测尺寸调整。</p></div><main>'''
for k,svg in svgs.items():doc+=f'<section id="p{k}" class="{"active" if k=="A" else ""}">{svg}</section>'
doc+='''</main><script>function show(k){document.querySelectorAll('section,button').forEach(e=>e.classList.remove('active'));document.getElementById('p'+k).classList.add('active');document.getElementById('b'+k).classList.add('active')}</script></html>'''
(OUT/'三套户型布局.html').write_text(doc,encoding='utf-8')
print('Saved three vector plans and comparison viewer.')
