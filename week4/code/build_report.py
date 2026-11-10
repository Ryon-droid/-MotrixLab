"""Build the Chinese PDF from answer.md, embedding every evidence screenshot."""
import pathlib

import markdown
from playwright.sync_api import sync_playwright
from runtime import ROOT


def main():
    source=(ROOT/'answer.md').read_text(encoding='utf-8')
    body=markdown.markdown(source,extensions=['tables','fenced_code'])
    markup='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第四周作业</title>
    <style>
    @page { size:A4; margin:15mm 13mm 17mm; }
    * { box-sizing:border-box; }
    body { font:10pt/1.65 "Noto Sans CJK SC",sans-serif; color:#23384a; margin:0; }
    h1 { font-size:24pt; line-height:1.35; color:#12344d; margin:0 0 18pt; }
    h2 { font-size:17pt; color:#13617e; margin:24pt 0 12pt; break-after:avoid; }
    h3 { font-size:13pt; color:#18516d; margin:16pt 0 9pt; break-after:avoid; }
    h4 { font-size:11pt; color:#18516d; margin:13pt 0 6pt; break-after:avoid; }
    p { margin:7pt 0; orphans:3; widows:3; }
    a { color:#17678a; text-decoration:none; overflow-wrap:anywhere; }
    code { font-family:"DejaVu Sans Mono","Noto Sans CJK SC",monospace; font-size:8.3pt; color:#18516d; overflow-wrap:anywhere; }
    pre { white-space:pre-wrap; overflow-wrap:anywhere; padding:9pt; background:#edf4f8; border-left:3pt solid #3f8ca5; font-size:8pt; line-height:1.65; break-inside:avoid; }
    pre code { font-size:8pt; }
    table { width:100%; border-collapse:collapse; table-layout:fixed; font-size:8pt; line-height:1.6; margin:10pt 0; }
    table code { font-size:7.2pt; }
    th { background:#d9eaf2; color:#113a52; text-align:left; }
    td,th { border:.5pt solid #c5d5e0; padding:5pt; vertical-align:top; overflow-wrap:anywhere; }
    tr { break-inside:avoid; }
    thead { display:table-header-group; }
    tbody tr:nth-child(even) { background:#f2f6f9; }
    img { display:block; max-width:100%; max-height:225mm; width:auto; height:auto; margin:12pt auto 6pt; object-fit:contain; }
    p:has(>img) { break-inside:avoid; text-align:center; }
    ul { margin:8pt 0; padding-left:20pt; }
    li { margin:4pt 0; }
    h2:nth-of-type(n+2) { break-before:page; }
    </style><body>'''+body+'</body></html>'
    path=ROOT/'第四周作业.html'
    path.write_text(markup,encoding='utf-8')
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        page=browser.new_page()
        page.goto(path.as_uri(),wait_until='networkidle')
        page.evaluate('document.fonts.ready')
        missing=page.locator('img').evaluate_all('(es)=>es.filter(e=>!e.complete||e.naturalWidth===0).map(e=>e.src)')
        if missing: raise RuntimeError(missing)
        page.pdf(path=str(ROOT/'第四周作业.pdf'),print_background=True,prefer_css_page_size=True,display_header_footer=True,
                 header_template='<div></div>',footer_template='<div style="font-family:Arial;font-size:8px;width:100%;padding:0 48px;color:#647789;display:flex;justify-content:space-between"><span>Week 4 | MotrixLab v0.3.0 | Ryon-droid | 2026-11-10</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',outline=True)
        browser.close()
    print(ROOT/'第四周作业.pdf')


if __name__=='__main__':
    main()
