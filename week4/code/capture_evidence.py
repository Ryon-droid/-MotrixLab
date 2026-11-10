"""Browser screenshots of exact source/log text and real TensorBoard dashboards."""
import ast
import csv
import html
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.request

from playwright.sync_api import sync_playwright
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from runtime import ROOT, SOURCE

OUT = ROOT / "evidence"
ENV = "motrix_envs/src/motrix_envs/"


def screenshot_text(page, name, title, path, numbered_lines, highlight=()):
    rows=''.join(f'<div class="line {"changed" if n in highlight else ""}"><span>{n:>4}</span> {html.escape(t)}</div>' for n,t in numbered_lines)
    markup=f'''<!doctype html><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#10283e;color:#e8eef6;font:19px "DejaVu Sans Mono",monospace}}
    header{{padding:22px 28px;color:#aee8d9;background:#173a54;line-height:1.7}}small{{font-size:16px;color:#b6c6d7}}
    main{{padding:22px 20px;background:#142331;width:max-content;min-width:100%}}.line{{white-space:pre;line-height:1.55;padding:0 10px}}
    .line span{{color:#7894ae}}.changed{{background:#674b13;color:#fff1b9}}
    footer{{padding:15px 28px;font-size:15px;color:#adbed0}}
    </style><header>{html.escape(title)}<br><small>{html.escape(path)}</small></header><main>{rows}</main><footer>Actual file text | MotrixLab v0.3.0 e1421d10559c | Browser screenshot, original line numbers</footer>'''
    file=OUT/'panels'/f'{name}.html'
    file.parent.mkdir(exist_ok=True)
    file.write_text(markup,encoding='utf-8')
    page.goto(file.as_uri())
    width=page.locator('main').evaluate('(e)=>e.scrollWidth')
    page.set_viewport_size({'width':max(1200,min(int(width),2400)),'height':900})
    page.screenshot(path=str(OUT/f'{name}.png'),full_page=True)


def source_panel(page,name,path,functions=None,ranges=None,highlight=()):
    file=SOURCE/path if not path.startswith('code/') else ROOT/path
    lines=file.read_text().splitlines()
    if functions:
        nodes=[n for n in ast.walk(ast.parse(file.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in functions]
        ranges=sorted((n.lineno,n.end_lineno) for n in nodes)
    selected=[]
    for start,end in ranges:
        if selected: selected.append((0,'...'))
        selected.extend((n,lines[n-1]) for n in range(start,min(end,len(lines))+1))
    screenshot_text(page,name,'SOURCE | '+path.split('/')[-1],path,selected,highlight)


def dashboard(browser,name):
    logdir=ROOT/'runs'/name
    log=(OUT/f'{name}_tensorboard.log').open('w')
    proc=subprocess.Popen([str(ROOT/'.venv/bin/tensorboard'),'--logdir_spec',f'{name}:{logdir}','--host','127.0.0.1','--port','6007','--load_fast=false'],stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen('http://127.0.0.1:6007/data/plugin/scalars/tags',timeout=1) as response:
                    if json.load(response): break
            except Exception:
                pass
            time.sleep(.2)
        page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page.goto('http://127.0.0.1:6007/#scalars',wait_until='networkidle')
        page.locator('input').nth(3).fill('Reward / Total reward')
        page.wait_for_timeout(1800)
        # Wait for reward cards to be drawn, with task identity in the sidebar.
        page.screenshot(path=str(OUT/f'{name}_tensorboard.png'),full_page=True)
        page.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()


def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
        for task in ['pendulum','acrobot','reacher']:
            path=ENV+f'basic/{task}/{task}_np.py'
            obsfn='_get_obs' if task!='pendulum' else None
            if obsfn: source_panel(page,f'code_{task}_observation',path,[obsfn,'_init_obs_space'] if task=='reacher' else [obsfn])
            else: source_panel(page,'code_pendulum_observation',path,ranges=[(61,68)])
            source_panel(page,f'code_{task}_action',path,['apply_action'])
            source_panel(page,f'code_{task}_reward',path,['update_state'] if task!='pendulum' else None,ranges=[(70,103)] if task=='pendulum' else None)
        source_panel(page,'code_pendulum_energy_B','code/pendulum_no_energy.py',ranges=[(70,95)],highlight=[88])
        source_panel(page,'code_pendulum_xml',ENV+'basic/pendulum/pendulum.xml',ranges=[(23,38)])
        source_panel(page,'code_acrobot_actuator',ENV+'basic/acrobot/acrobot.xml',ranges=[(36,55)])
        source_panel(page,'code_reacher_reset',ENV+'basic/reacher/reacher_np.py',['reset'])
        source_panel(page,'code_reacher_xml',ENV+'basic/reacher/reacher.xml',ranges=[(29,62)])
        source_panel(page,'code_timeout_base',ENV+'np/env.py',['_update_truncate','step'])
        source_panel(page,'code_acrobot_timeout',ENV+'basic/acrobot/acrobot_np.py',ranges=[(112,125),(145,152)])
        for name in ['cartpole','pendulum_A','pendulum_B','acrobot']:
            lines=(OUT/f'{name}_train.log').read_text().splitlines()
            # Show the actual summary and completion marker without truncating values.
            clean=[(i,re.sub(r'\x1b\[[0-9;]*m','',line)) for i,line in enumerate(lines,1) if 'TRAINING_COMPLETED' in line or any('"'+k+'"' in line for k in ['task','group','seed','vector_steps','num_envs','environment_transitions','device','torch','skrl','motrixsim'])]
            expanded=[]
            for n,line in clean:
                if line.startswith('TRAINING_COMPLETED'):
                    words=line.split();expanded.extend((n,' '.join(words[i:i+4])) for i in range(0,len(words),4))
                else: expanded.append((n,line))
            screenshot_text(page,f'{name}_train_end','TRAINING LOG | '+name,f'evidence/{name}_train.log',expanded)
        page.close()
        for name in ['cartpole','pendulum_A','pendulum_B','acrobot']:
            dashboard(browser,name)
        browser.close()
    rows=[];summary={}
    for file in (ROOT/'runs').glob('*/seed42/events*'):
        name=file.parents[1].name
        ea=EventAccumulator(str(file));ea.Reload()
        summary[name]={}
        for tag in ea.Tags()['scalars']:
            values=ea.Scalars(tag)
            if values: summary[name][tag]={'first':values[0].value,'last':values[-1].value,'last_step':values[-1].step,'count':len(values)}
            rows.extend([name,tag,v.step,v.value,v.wall_time] for v in values)
    with (OUT/'scalars.csv').open('w') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['run','tag','vector_step','value','wall_time']);writer.writerows(rows)
    (OUT/'scalar_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('EVIDENCE_SCREENSHOTS_COMPLETED')


if __name__=='__main__':
    main()
