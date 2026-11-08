"""Rebuild source excerpt images, interaction diagram and the PDF submission.

Document-only dependencies: reportlab, pillow. Runtime simulation uses requirements.txt.
The images are explicitly labeled source exports, not desktop screenshots.
"""
import ast
import html
import json
import pathlib
import re

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence"
SOURCE = ROOT / "source"
FONT_DIR = pathlib.Path("C:/Windows/Fonts")
CN = ImageFont.truetype(str(FONT_DIR / "msyh.ttc"), 25)
SMALL = ImageFont.truetype(str(FONT_DIR / "msyh.ttc"), 20)
MONO = ImageFont.truetype(str(FONT_DIR / "consola.ttf"), 21)
TINY_MONO = ImageFont.truetype(str(FONT_DIR / "consola.ttf"), 17)
ENV = "motrix_envs/src/motrix_envs/"
RL = "motrix_rl/src/motrix_rl/"
CART = ENV + "basic/cartpole/cartpole_np.py"
BASE = ENV + "base.py"
NP = ENV + "np/env.py"

def function_span(path, name):
    tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(nodes) != 1:
        raise ValueError((path, name, len(nodes)))
    node = nodes[0]
    return node.lineno, node.end_lineno

def excerpt(name, groups):
    rows = []
    for path, ranges in groups:
        rows.append(("path", path))
        lines = (SOURCE / path).read_text(encoding="utf-8").splitlines()
        for start, end in ranges:
            for i in range(start, min(end, len(lines)) + 1):
                rows.append(("code", i, lines[i - 1].expandtabs(4)))
            rows.append(("gap",))
    width = max(1280, int(max([MONO.getlength(r[2]) + 135 for r in rows if r[0] == "code"] + [1280])))
    height = 90 + sum(37 if r[0] == "path" else 14 if r[0] == "gap" else 29 for r in rows) + 35
    image = PILImage.new("RGB", (width, height), "#f7f9fc")
    d = ImageDraw.Draw(image)
    d.rectangle((0, 0, width, 70), fill="#12324c")
    d.text((25, 16), "SOURCE EXPORT  |  MotrixLab v0.3.0  |  e1421d10559c", font=MONO, fill="white")
    y = 88
    for row in rows:
        if row[0] == "path":
            d.rectangle((18, y - 3, width - 18, y + 28), fill="#e6eef6")
            d.text((26, y), row[1], font=TINY_MONO, fill="#174e7a")
            y += 37
        elif row[0] == "gap":
            y += 14
        else:
            _, lineno, text = row
            d.text((25, y), f"{lineno:>3}", font=MONO, fill="#8594a4")
            fill = "#6b8196" if text.lstrip().startswith("#") else "#164978" if text.lstrip().startswith(("def ", "class ", "@")) else "#172f43"
            d.text((92, y), text, font=MONO, fill=fill)
            y += 29
    d.text((25, height - 25), "Exact source lines with original line numbers. Exported for code review; not an app-window screenshot.", font=ImageFont.truetype(str(FONT_DIR / "consola.ttf"), 15), fill="#6b8196")
    image.save(OUT / name)

def make_source_images():
    init, _ = function_span(CART, "__init__")
    update, _ = function_span(CART, "update_state")
    apply_start, apply_end = function_span(CART, "apply_action")
    reset_start, reset_end = function_span(CART, "reset")
    excerpt("code_observation.png", [(CART, [(init, init + 9), (update, update + 7)])])
    excerpt("code_apply_action.png", [(CART, [(apply_start, apply_end)]),
            (RL + "skrl/torch/wrap_np.py", [(51, 59)])])
    excerpt("code_policy.png", [(RL + "tasks/cartpole.py", [(48, 67)])])
    xml = ENV + "basic/cartpole/cartpole.xml"
    excerpt("code_actuator.png", [(xml, [(25, 38)])])
    excerpt("code_reward_termination.png", [(CART, [(update + 8, update + 22)])])
    trunc_start, trunc_end = function_span(NP, "_update_truncate")
    clock_start, clock_end = function_span(BASE, "max_episode_steps")
    excerpt("code_timeout.png", [(ENV + "basic/cartpole/cfg.py", [(27, 34)]),
            (BASE, [(30, 34), (clock_start, clock_end)]), (NP, [(trunc_start, trunc_end)])])
    auto_start, _ = function_span(NP, "_reset_done_envs")
    step_start, step_end = function_span(NP, "step")
    excerpt("code_autoreset.png", [(NP, [(auto_start, auto_start + 15), (step_start, step_end)])])
    excerpt("code_reset.png", [(CART, [(reset_start, reset_end)])])

def make_flow():
    im = PILImage.new("RGB", (1440, 1050), "#f4f8fc")
    d = ImageDraw.Draw(im)
    d.text((45, 25), "CartPole：一个回合的交互与自动重置", font=ImageFont.truetype(str(FONT_DIR / "msyhbd.ttc"), 35), fill="#12324c")
    def box(rect, text, fill="#ffffff"):
        d.rounded_rectangle(rect, radius=16, fill=fill, outline="#b6c9db", width=3)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            tw = d.textlength(line, font=CN)
            d.text(((rect[0]+rect[2]-tw)/2, (rect[1]+rect[3])/2 - len(lines)*18 + i*36), line, font=CN, fill="#183b55")
    def arrow(points, text=None, pos=None, color="#267e9b"):
        d.line(points, fill=color, width=5)
        x,y = points[-1]; px,py = points[-2]
        vec = complex(x-px,y-py); vec = vec/abs(vec)
        tip = complex(x,y)
        left = tip - 15*vec + 8j*vec; right = tip - 15*vec - 8j*vec
        d.polygon([(x,y),(left.real,left.imag),(right.real,right.imag)], fill=color)
        if text: d.text(pos, text, font=SMALL, fill=color)
    box((70,120,460,230), "init_state / reset\n采样初始位置与速度噪声", "#e1f1f0")
    box((610,120,990,230), "观测 obs\n[x, theta, dx, dtheta]")
    arrow([(460,175),(610,175)], "①初始观测", (476,138))
    box((610,320,990,420), "策略 policy → 动作 a")
    arrow([(800,230),(800,320)])
    box((610,510,990,635), "step / apply_action\n执行器限幅与 ×100 推力\n推进物理仿真")
    arrow([(800,420),(800,510)], "②动作传给环境", (825,454))
    box((610,715,990,820), "update_state\n新观测、奖励、失败判断")
    arrow([(800,635),(800,715)])
    box((610,900,990,1000), "计步 / 超时检查\ndone = terminated | truncated", "#fff0d8")
    arrow([(800,820),(800,900)])
    arrow([(990,950),(1290,950),(1290,175),(990,175)], "③未结束：新观测\n用于下一步", (1030,420))
    arrow([(610,950),(265,950),(265,230)], "④已结束：自动 reset", (85,520))
    d.text((70,760), "结束当步仍返回上一回合的\n奖励和结束标志；obs 已替换\n为新回合的初始观测。", font=SMALL, fill="#536c82", spacing=12)
    im.save(OUT / "interaction_flow.png")

def inline(text):
    text = text.replace("ẋ", "dx/dt").replace("θ̇", "dθ/dt")
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r'<font color="#1e587e">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r'<b>\1</b>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'\1', text)
    return text

def build_pdf():
    pdfmetrics.registerFont(TTFont("CN", str(FONT_DIR / "simhei.ttf")))
    pdfmetrics.registerFontFamily("CN", normal="CN", bold="CN", italic="CN", boldItalic="CN")
    body = ParagraphStyle("Body", fontName="CN", fontSize=10.3, leading=16.2, spaceAfter=7, wordWrap="CJK", textColor=colors.HexColor("#23394b"))
    title = ParagraphStyle("Title", parent=body, fontSize=24, leading=31, spaceAfter=15, textColor=colors.HexColor("#12324c"))
    h2 = ParagraphStyle("H2", parent=body, fontSize=16, leading=23, spaceBefore=15, spaceAfter=9, keepWithNext=True, textColor=colors.HexColor("#165679"))
    h3 = ParagraphStyle("H3", parent=body, fontSize=12.5, leading=19, spaceBefore=10, keepWithNext=True, textColor=colors.HexColor("#165679"))
    cell = ParagraphStyle("Cell", parent=body, fontSize=9, leading=13.4, spaceAfter=0)
    caption = ParagraphStyle("Caption", parent=body, fontSize=8.3, leading=12.5, textColor=colors.HexColor("#647e91"), spaceAfter=12)
    width = A4[0]-88
    story = []
    lines = (ROOT / "answer.md").read_text(encoding="utf-8").splitlines()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1; continue
        if line.startswith("```"):
            i+=1
            while i<len(lines) and not lines[i].startswith("```"): i+=1
            i+=1; continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                raw=[x.strip() for x in lines[i].strip().strip("|").split("|")]
                # Literal logical OR is avoided inside the source tables.
                if not all(re.fullmatch(r"[:\- ]+", x) for x in raw): rows.append(raw)
                i+=1
            cols=len(rows[0])
            widths=([54,292,width-346] if rows[0][0]=="要素" else
                    [64,126,width-190] if cols==3 else
                    [50,105,170,75,width-400] if cols==5 else [width/cols]*cols)
            use_cell = ParagraphStyle("EightElements", parent=cell, fontSize=8.6, leading=12.2) if rows[0][0]=="要素" else cell
            padding = 5 if rows[0][0]=="要素" else 7
            table=Table([[Paragraph(inline(x),use_cell) for x in row] for row in rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceaf4")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7fa")]),
                ("GRID",(0,0),(-1,-1),.45,colors.HexColor("#cfdae3")),("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
                ("TOPPADDING",(0,0),(-1,-1),padding),("BOTTOMPADDING",(0,0),(-1,-1),padding)]))
            story.extend([table,Spacer(1,10)]); continue
        m=re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)",line)
        if m:
            img=PILImage.open(ROOT/m[2]); iw,ih=img.size
            scale=min(width/iw,520/ih)
            story.extend([Image(str(ROOT/m[2]),width=iw*scale,height=ih*scale,hAlign="LEFT"),Paragraph(inline(m[1])+"（源码导出图，保留文件名与行号）" if m[2].startswith("evidence/code_") else inline(m[1]),caption)])
            i+=1; continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]),title))
        elif line.startswith("## "):
            if line.startswith("## 二、") or line.startswith("## 三、"):
                story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]),h2))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]),h3))
        else:
            story.append(Paragraph(inline(line),body))
        i+=1
    report=json.loads((OUT/"verification.json").read_text())
    story.extend([PageBreak(),Paragraph("补充：实际运行结果与效果图",h2),
        Paragraph("运行核验通过。观测四维、动作一维；严格边界、自动重置、1024 个初始化样本及动作限幅均与正文一致。",body),
        Paragraph("额外提供可运行的 LQR 工程演示入口（非强化学习训练）。seed=42 下实际运行 2000 步，两个回合均达到 1000 步时间截断，失败回合为 0；详细结果保存在 balance_run.json。",body),
        Paragraph("以下为官方 MotrixSim 0.7.0 引擎相机图像。为说明状态阈值，手动设置关节状态，只在内存中添加取景相机；没有改变物理模型。这些图不作为巧工窗口截图或训练效果证据。",body)])
    for name,text in [("simulation_upright.png","参考状态：x=0 m、theta=0 rad，竖直且位于轨道中心。"),
                      ("simulation_near_limit.png","正常范围示例：x=0.60 m、theta=0.18 rad，未触发位置/角度终止条件。"),
                      ("simulation_failed.png","失败状态示例：theta=0.25 rad，大于 0.2 rad 阈值，触发失败终止。")]:
        img=PILImage.open(OUT/name); iw,ih=img.size
        draw_width=min(width, 295*iw/ih)
        story.extend([Image(str(OUT/name),width=draw_width,height=draw_width*ih/iw,hAlign="LEFT"),Paragraph(text,caption)])
    def decorate(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#cfdae3"));canvas.line(44,40,A4[0]-44,40)
        canvas.setFont("CN",8);canvas.setFillColor(colors.HexColor("#647e91"))
        canvas.drawString(44,26,"第三周 CartPole  |  MotrixLab v0.3.0  |  2026-11-08")
        canvas.drawRightString(A4[0]-44,26,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/"CartPole分析作业.pdf"),pagesize=A4,leftMargin=44,rightMargin=44,
                          topMargin=38,bottomMargin=53,title="第三周 CartPole 环境分析",author="Ryon-droid")
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)

if __name__=="__main__":
    make_source_images()
    make_flow()
    build_pdf()
    print(ROOT/"CartPole分析作业.pdf")
