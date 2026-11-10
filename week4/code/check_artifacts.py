"""Check submission links, real video frames, checkpoint hashes and complete episodes."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from pypdf import PdfReader
from runtime import ROOT


def main():
    answer=(ROOT/'answer.md').read_text()
    report={'document':'第四周作业.pdf','checks':{},'videos':{},'sha256':{}}
    links=re.findall(r'\]\(([^)]+)\)',answer)
    local=[link for link in links if not re.match(r'https?://',link)]
    for file in local:
        assert (ROOT/file).is_file(),file
        with Image.open(ROOT/file) as im:
            im.verify()
    report['checks']['embedded_image_references']=len(local)
    pdf=PdfReader(ROOT/'第四周作业.pdf')
    assert len(pdf.pages)>10
    text=unicodedata.normalize('NFKC','\n'.join(p.extract_text() for p in pdf.pages))
    for required in ['一、三个 Demo','二、单摆能量','三、双关节','0.04','6.38','2.83','target_x']:
        assert unicodedata.normalize('NFKC',required) in text,required
    urls=[]
    for page in pdf.pages:
        for annotation in page.get('/Annots',[]):
            obj=annotation.get_object()
            url=obj.get('/A',{}).get('/URI','')
            if url: urls.append(str(url))
    expected_videos={p.name for p in (ROOT/'videos').glob('*.mp4')}
    assert len(expected_videos)==8
    assert all(any(name in url for url in urls) for name in expected_videos)
    report['checks']['pdf_pages']=len(pdf.pages)
    report['checks']['pdf_video_links_present']=True
    source=json.loads((ROOT/'evidence/source_manifest.json').read_text())
    for file,h in source['sha256'].items():
        assert hashlib.sha256((ROOT/'source'/file).read_bytes()).hexdigest()==h
    report['checks']['source_hashes_verified']=len(source['sha256'])
    a=json.loads((ROOT/'runs/pendulum_A/manifest.json').read_text())
    b=json.loads((ROOT/'runs/pendulum_B/manifest.json').read_text())
    assert a['rl_config']==b['rl_config'] and a['env_config']==b['env_config']
    report['checks']['ab_configs_identical']=True
    for name in ['cartpole','pendulum_A','pendulum_B','acrobot']:
        m=json.loads((ROOT/'runs'/name/'manifest.json').read_text())
        assert m['completed']
        assert m['num_envs']==m['rl_config']['num_envs']==128
        assert m['vector_steps']==m['rl_config']['runner']['trainer']['timesteps']
        policy=ROOT/m['final_checkpoint']
        actual=hashlib.sha256(policy.read_bytes()).hexdigest()
        assert actual==m['final_checkpoint_sha256']
        replay=ROOT/'evidence/replay'/name
        entries=json.loads((replay/'episodes.json').read_text())['episodes']
        cap=json.loads((replay/'capture_manifest.json').read_text())
        for entry in entries:
            seed=entry['seed']
            assert entry['policy_sha256']==actual
            assert entry['terminal_state_captured_before_autoreset']
            data=np.load(replay/f'seed{seed}.npz')
            assert len(data['qpos'])==entry['steps']+1
            assert entry['terminated'] or entry['truncated']
            video=ROOT/'videos'/f'{name}_seed{seed}.mp4'
            c=next(c for c in cap['episodes'] if c['seed']==seed)
            with imageio.get_reader(str(video)) as reader:
                meta=reader.get_meta_data()
                frames=reader.count_frames()
                first=reader.get_data(0)
                last=reader.get_data(frames-1)
            assert frames==len(c['frame_indices'])
            assert c['frame_indices'][0]==0 and c['frame_indices'][-1]==entry['steps']
            assert first.shape==last.shape==(736,960,3)
            # H.264 is lossy; allow compression noise while checking stored image identity.
            first_ref=np.asarray(Image.open(replay/f'seed{seed}_initial.png').convert('RGB'))
            last_ref=np.asarray(Image.open(replay/f'seed{seed}_terminal.png').convert('RGB'))
            first_error=float(np.mean(abs(first.astype(float)-first_ref)))
            last_error=float(np.mean(abs(last.astype(float)-last_ref)))
            assert max(first_error,last_error)<8,(name,seed,first_error,last_error)
            report['videos'][video.name]={'frames':frames,'fps':meta['fps'],'encoded_duration_seconds':meta['duration'],'simulation_duration_seconds':entry['duration_seconds'],'first_frame_mean_absolute_error':first_error,'terminal_frame_mean_absolute_error':last_error,'terminal_frame_before_reset':True}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '.venv' in p.parts or '__pycache__' in p.parts or p.name=='artifact_manifest.json':
            continue
        assert p.stat().st_size<95*1024*1024,str(p)
        report['sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['all_checks_passed']=True
    target=ROOT/'evidence/artifact_manifest.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'all_checks_passed':True,'pdf_pages':len(pdf.pages),'videos':len(report['videos']),'files_hashed':len(report['sha256'])},indent=2))


if __name__=='__main__':
    main()
