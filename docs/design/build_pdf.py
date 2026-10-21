import json,re,os
from pathlib import Path
from xml.sax.saxutils import escape
from datetime import datetime,timezone
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).parent
os.environ['SOURCE_DATE_EPOCH']=str(int(datetime(2026,10,20,12,tzinfo=timezone.utc).timestamp()))
PAGES=json.loads((ROOT/'design_pages.json').read_text())
OUT=ROOT/'知选ChoiceAgent_项目设计书_2026-10-20.pdf'
pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('LatinBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
W,H=A4;M=40;CW=W-M*2
NAVY=colors.HexColor('#123246');TEAL=colors.HexColor('#087E83');INK=colors.HexColor('#263D4B');GRAY=colors.HexColor('#5A707E');LIGHT=colors.HexColor('#EDF4F5');LINE=colors.HexColor('#D4E1E5');GOLD=colors.HexColor('#D7AA67')
c=canvas.Canvas(str(OUT),pagesize=A4,pageCompression=1)
c.setTitle('知选 ChoiceAgent｜搜广推与 Agent 双方向项目设计书')
c.setAuthor('个人化项目设计 · 2026-10-20')
c.setSubject('商品搜索推荐、工具Agent、A6000本地模型与可选DeepSeek；面向2028届首次实习')
y=0;page_num=0;bounds=[];missing=set()
def runs(s):
    out=[]
    for ch in s:
        f='CJK' if ord(ch)>=0x2e80 and ord(ch) in pdfmetrics.getFont('CJK').face.charToGlyph else 'Latin'
        if ch not in '\n\r\t' and ord(ch) not in pdfmetrics.getFont(f).face.charToGlyph: missing.add(ch)
        if out and out[-1][0]==f:out[-1]=(f,out[-1][1]+ch)
        else:out.append((f,ch))
    return out

def plainrich(s):
    return ''.join('<font name="'+f+'">'+escape(v)+'</font>' for f,v in runs(s))
def rich(s):
    pieces=[];pos=0
    for mt in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)',s):
        pieces.append(plainrich(s[pos:mt.start()]))
        label,url=mt.groups()
        if url.startswith('http'):pieces.append('<link href="'+escape(url)+'" color="#087E83">'+plainrich(label)+'</link>')
        else:pieces.append(plainrich(label))
        pos=mt.end()
    pieces.append(plainrich(s[pos:]));return ''.join(pieces).replace('\n','<br/>')
def para(s,size=10,color=INK,leading=None):
    return Paragraph(rich(s),ParagraphStyle('p',fontName='CJK',fontSize=size,leading=leading or size*1.53,textColor=color,wordWrap='CJK',splitLongWords=True))
def drawtext(s,x,yy,size=10,color=INK):
    c.setFillColor(color)
    for f,v in runs(s):
        c.setFont(f,size);c.drawString(x,yy,v);x+=pdfmetrics.stringWidth(v,f,size)
def p(s,size=11,gap=11,color=INK,width=CW,x=M):
    global y
    pp=para(s,size,color);_,hh=pp.wrap(width,H)
    if y-hh<54:raise ValueError(f'OVERFLOW page={page_num} y={y:.1f} h={hh:.1f} text={s[:80]}')
    pp.drawOn(c,x,y-hh);y-=hh+gap

def table(b):
    global y
    widths=b.get('widths') or [CW/len(b['head'])]*len(b['head'])
    widths=[z*CW/sum(widths) for z in widths]
    data=[[para(str(z),9.5,colors.white,14.3) for z in b['head']]]
    data += [[para(str(z),9.4,INK,14.3) for z in row] for row in b['rows']]
    t=Table(data,colWidths=widths)
    style=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]
    for i in range(1,len(data)):
        if i%2:style.append(('BACKGROUND',(0,i),(-1,i),LIGHT))
        style.append(('LINEBELOW',(0,i),(-1,i),.35,LINE))
    t.setStyle(TableStyle(style));_,hh=t.wrap(CW,H)
    if y-hh<54:raise ValueError(f'TABLE OVERFLOW page={page_num} y={y} h={hh}')
    t.drawOn(c,M,y-hh);y-=hh+12

def code(s):
    global y
    pp=para(s,8.6,INK,13.1);_,hh=pp.wrap(CW-22,H)
    hh+=20
    if y-hh<54:raise ValueError(f'CODE OVERFLOW page={page_num}')
    c.setFillColor(LIGHT);c.roundRect(M,y-hh,CW,hh,6,fill=1,stroke=0)
    c.setFillColor(TEAL);c.rect(M,y-hh,3,hh,fill=1,stroke=0)
    pp.drawOn(c,M+11,y-hh+10);y-=hh+12

def box(x,top,w,h,title,body='',accent=False):
    c.setFillColor(NAVY if accent else LIGHT);c.setStrokeColor(LINE)
    c.roundRect(x,top-h,w,h,6,fill=1,stroke=0)
    titlep=para(title,10.2,colors.white if accent else NAVY,15)
    _,ht=titlep.wrap(w-18,h);titlep.drawOn(c,x+9,top-10-ht)
    if body:
        bp=para(body,8.7,colors.HexColor('#DCEBED') if accent else GRAY,12.8)
        _,hb=bp.wrap(w-18,h);bp.drawOn(c,x+9,top-16-ht-hb)

def arrow(x1,y1,x2,y2):
    import math
    c.setStrokeColor(TEAL);c.setLineWidth(1.15);c.line(x1,y1,x2,y2)
    angle=math.atan2(y2-y1,x2-x1);al=5
    for sign in [-1,1]:c.line(x2,y2,x2-al*math.cos(angle+sign*.5),y2-al*math.sin(angle+sign*.5))

def diagram(kind):
    global y
    top=y
    if kind=='architecture':
        gap=22;bw=(CW-2*gap)/3;x=[M,M+bw+gap,M+2*(bw+gap)]
        box(x[0],top,bw,65,'原始数据','商品元信息 / 评论')
        box(x[1],top,bw,65,'离线构建','清洗、切分、编码、CF')
        box(x[2],top,bw,65,'版本化产物','SQLite / FAISS / 模型')
        arrow(x[0]+bw,top-32,x[1]-3,top-32);arrow(x[1]+bw,top-32,x[2]-3,top-32)
        top2=top-98
        box(x[0],top2,bw,82,'普通搜索页面','搜索 / 推荐 / 反馈')
        box(x[1],top2,bw,82,'共享检索排序服务','过滤 → 召回 → 排序\n返回 IDs 与证据',True)
        box(x[2],top2,bw,82,'Agent 工具层','搜索 / 比较 / 偏好\n有界循环与校验')
        arrow(x[0]+bw,top2-40,x[1]-3,top2-40);arrow(x[2],top2-40,x[1]+bw+3,top2-40)
        arrow(x[2]+bw*.5,top-65,x[1]+bw*.65,top2+4)
        top3=top2-115
        box(x[0],top3,bw,65,'会话与反馈','显式偏好 / 候选清单')
        box(x[1],top3,bw,65,'结果与轨迹','证据 / 指标 / 失败原因')
        box(x[2],top3,bw,65,'模型提供方','本地 Qwen / DeepSeek')
        arrow(x[1]+bw*.5,top2-82,x[1]+bw*.5,top3+4)
        arrow(x[2]+bw*.5,top3+3,x[2]+bw*.5,top2-86)
        y=top3-65-14
    elif kind=='split':
        bw=(CW-24)/3
        labels=[('训练 ≤ T0','目录 / 画像 / 模型'),('验证 T0—T1','选参数 / checkpoint'),('测试 > T1','锁定后正式评测')]
        for i,(t,b) in enumerate(labels):box(M+i*(bw+12),top,bw,62,t,b,i==0)
        arrow(M+12,top-81,W-M-12,top-81)
        drawtext('事件时间 →   全局切分；同时间戳归同一区间',M+14,top-103,9,GRAY)
        y=top-120
    elif kind=='agent':
        bw=(CW-44)/3;xs=[M,M+bw+22,M+2*(bw+22)]
        box(xs[0],top,bw,61,'加载上下文','当前约束 / 偏好 / 状态')
        box(xs[1],top,bw,61,'模型选择动作','澄清 / 工具 / 回答',True)
        box(xs[2],top,bw,61,'校验并执行','参数 / ID / 权限 / 预算')
        arrow(xs[0]+bw,top-30,xs[1]-3,top-30);arrow(xs[1]+bw,top-30,xs[2]-3,top-30)
        top2=top-94
        box(xs[0],top2,bw,61,'返回与结束','有据答案 / 未完成原因')
        box(xs[1],top2,bw,61,'输出验证','约束 / 字段 / 事实引用')
        box(xs[2],top2,bw,61,'更新状态与证据','工具结果 → 再次决策')
        arrow(xs[2]+bw*.5,top-61,xs[2]+bw*.5,top2+3)
        arrow(xs[2],top2-30,xs[1]+bw+3,top2-30);arrow(xs[1],top2-30,xs[0]+bw+3,top2-30)
        c.setStrokeColor(GOLD);c.setLineWidth(1)
        c.line(xs[2]+bw*.3,top2+4,xs[1]+bw*.7,top-68)
        drawtext('达到调用数或总时限即停止；修复最多一次',M+12,top2-85,9,GRAY)
        y=top2-102

def start(pg,i):
    global y,page_num
    if i:c.showPage()
    page_num=i+1
    c.setFillColor(TEAL);c.rect(M,H-33,25,3,fill=1,stroke=0)
    drawtext('CHOICEAGENT / 项目设计',M+35,H-35,8,GRAY)
    drawtext('2026.10.20 · v1.0',W-M-103,H-35,8,GRAY)
    drawtext(pg['title'],M,H-72,18,NAVY)
    c.setStrokeColor(LINE);c.line(M,42,W-M,42)
    drawtext('设计方案 · 数值目标尚未实测 · 面向 2028 届首份实习',M,27,7.4,GRAY)
    c.setFont('Latin',8);c.setFillColor(GRAY);c.drawRightString(W-M,27,f'{page_num} / {len(PAGES)}')
    key=f'p{page_num}';c.bookmarkPage(key);c.addOutlineEntry(pg['title'],key,0)
    y=H-94
    if pg['kicker']:p(pg['kicker'],8.8,15,GRAY)

def cover(pg):
    global y,page_num
    page_num=1
    c.setFillColor(NAVY);c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(TEAL);c.rect(M,H-69,75,5,fill=1,stroke=0)
    drawtext('SEARCH · RECOMMENDATION · AGENT',M,H-105,10,colors.HexColor('#A7D7D7'))
    drawtext('知选',M,H-183,38,colors.white)
    drawtext('ChoiceAgent',M+105,H-183,31,colors.white)
    drawtext('个性化商品搜索、推荐与决策助手',M,H-231,20,colors.white)
    drawtext('项目设计书',M,H-272,24,colors.HexColor('#A7D7D7'))
    y=H-320
    for b in pg['blocks'][:5]:p(b['text'],10.3,17,colors.HexColor('#D5E5EA'))
    c.setFillColor(TEAL);c.roundRect(M,126,CW,48,7,fill=1,stroke=0)
    drawtext('两周可演示',M+17,145,15,colors.white)
    drawtext('六周完成求职版本',M+228,145,15,colors.white)
    y=106;p(pg['blocks'][-1]['text'],8.5,5,colors.HexColor('#B3CAD4'))
    c.bookmarkPage('cover');c.addOutlineEntry('封面', 'cover',0)

for i,pg in enumerate(PAGES):
    if i==0:cover(pg)
    else:
        start(pg,i)
        for b in pg['blocks']:
            typ=b['type']
            if typ=='p':p(b['text'])
            elif typ=='note':p(b['text'],9.1,10,GRAY)
            elif typ=='h':
                y-=3;p(b['text'],11.3,8,TEAL)
            elif typ=='table':table(b)
            elif typ=='code':code(b['text'])
            elif typ=='diagram':diagram(b['kind'])
    bounds.append({'page':page_num,'bottom_content':round(y,2),'title':pg['title']})
c.save()
(ROOT/'layout_audit.json').write_text(json.dumps({'page_count':len(PAGES),'bounds':bounds,'missing_glyphs':sorted(missing)},ensure_ascii=False,indent=2))
print(OUT)
print('pages',len(PAGES),'min_bottom',min(x['bottom_content'] for x in bounds),'missing_glyphs',sorted(missing))
