"""只读汇总阶段诊断，生成预测热图、时序图和JSON；不改原实验。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize_episode(path, figures=None):
    predictions = [json.loads(line) for line in (path/'predictions.jsonl').read_text().splitlines()]
    states = [json.loads(line) for line in (path/'physical_states.jsonl').read_text().splitlines()]
    rows = list(csv.DictReader((path/'steps.csv').open()))
    result = json.loads((path/'result.json').read_text())
    if len(rows) != result['steps'] or len(states) != result['steps']:
        raise ValueError('诊断步数不完整')
    if not predictions:
        raise ValueError('未记录模型预测')
    offsets, selected, full_negative, postponement, overlaps = [], [], [], [], []
    for j,p in enumerate(predictions):
        a = np.asarray(p['clipped'])
        pos = np.flatnonzero(a[:,6]>0)
        offsets.append(int(pos[0]) if len(pos) else None)
        full_negative.append(bool(np.all(a[:,6]<0)))
        selected.append(bool(np.any(a[p['selected_offsets'],6]>0)))
        # Verify saved selected ACT rows against the actually executed actions.
        for k in p['selected_offsets']:
            row = rows[p['t']+k]
            if row['action_source']=='act':
                np.testing.assert_array_equal(a[k], [float(row[f'action_{i}']) for i in range(7)])
        if j:
            old = predictions[j-1]; delta=p['t']-old['t']
            old_a=np.asarray(old['clipped'])
            overlap=min(len(a),len(old_a)-delta)
            if overlap>0:
                overlaps.extend((np.sign(a[:overlap,6]) != np.sign(old_a[delta:delta+overlap,6])).tolist())
            if (offsets[j] is not None and offsets[j-1] is not None
                    and offsets[j] >= p['H'] and offsets[j-1] >= old['H']):
                # Only compare still-unexecuted closing events; continuous closing
                # after phase transition trivially shifts the first-positive time.
                postponement.append(delta+offsets[j]-offsets[j-1])
    act = [s for s in states if s['action_source']=='act']
    g=np.array([s['action'][6] for s in act]); contacts={k:sum(s['after']['contacts'][k] for s in states) for k in states[0]['after']['contacts']}
    both_steps=[s['step'] for s in states if s['after']['contacts']['left_fingerpad_cube'] and s['after']['contacts']['right_fingerpad_cube']]
    table_steps=[s['step'] for s in states if s['after']['contacts']['left_finger_table'] or s['after']['contacts']['right_finger_table']]
    both=len(both_steps)
    late_blocks = [(p,o) for p,o in zip(predictions,offsets) if 30 <= p['t'] <= 80]
    rg=[{'t':p['t'],'r_g':o,'predicted_step':p['t']+o+1 if o is not None else None} for p,o in zip(predictions,offsets)]
    out=dict(episode_id=result['episode_id'], predictions=len(predictions), full_block_all_negative=sum(full_negative),
             blocks_with_closing=sum(o is not None for o in offsets), blocks_with_executed_closing=sum(selected),
             blocks_closing_only_beyond_H=sum(o is not None and o>=p['H'] for p,o in zip(predictions,offsets)),
             late_blocks_30_to_80=len(late_blocks),
             late_blocks_with_closing=sum(o is not None for p,o in late_blocks),
             late_blocks_closing_beyond_H=sum(o is not None and o>=p['H'] for p,o in late_blocks),
             first_ACT_positive_step=next((s['step'] for s in act if s['action'][6]>0), None),
             positive_ACT_steps=int(sum(g>0)), ACT_sign_reversals=int(sum(g[1:]*g[:-1]<0)),
             contact_step_counts=contacts, both_fingerpads_cube_steps=both,
             first_both_fingerpads_cube_step=both_steps[0] if both_steps else None,
             first_finger_table_step=table_steps[0] if table_steps else None,
             last_finger_table_step=table_steps[-1] if table_steps else None,
             same_future_gripper_sign_conflict_fraction=float(np.mean(overlaps)) if overlaps else None,
             deferred_closing_comparisons=len(postponement),
             positive_deferred_closing_postponements=sum(v>0 for v in postponement),
             first_lift_step=result['first_lift_step'],hold_status=result['first_lift_hold_status'], closing_offsets=rg)
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(4,1,figsize=(10,11),constrained_layout=True)
        grid=np.stack([p['clipped'] for p in predictions])[:,:,6]
        axs[0].imshow(grid.T,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='coolwarm',extent=[0,len(predictions),-.5,grid.shape[1]-.5]);axs[0].axhline(predictions[0]['H']-.5,color='black');axs[0].set(xlabel='Prediction index',ylabel='Future offset k',title=f"Episode {result['episode_id']}: gripper predictions (red=close)")
        ts=[p['t'] for p in predictions];axs[1].plot(ts,[np.nan if o is None else o for o in offsets],'.-');axs[1].axhline(predictions[0]['H'],color='black',ls='--');axs[1].set(ylabel='First positive offset',xlabel='Executed steps before prediction')
        x=[s['step'] for s in states]
        axs[2].plot(x,[s['action'][6] for s in states],label='gripper command');axs[2].plot(x,[s['after']['gripper_target'][0] for s in states],label='internal target[0]');axs[2].plot(x,[abs(s['after']['gripper_qpos'][0]-s['after']['gripper_qpos'][1])*10 for s in states],label='aperture (dm)');axs[2].legend()
        axs[3].plot(x,[s['after']['eef_position'][2] for s in states],label='eef z (m)');axs[3].plot(x,[s['after']['cube_position'][2] for s in states],label='cube z (m)');axs[3].plot(x,[float(s['after']['contacts']['left_fingerpad_cube'] and s['after']['contacts']['right_fingerpad_cube']) for s in states],label='both pads-cube (0/1)');axs[3].legend();axs[3].set(xlabel='Control step')
        fig.savefig(figures/f"episode_{result['episode_id']:04d}.png",dpi=130);plt.close(fig)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='可选历史结果目录；逐字段核对CSV与result，不改原文件')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    seen=set()
    verified=[]
    for root in args.input:
        meta=json.loads((root/'config.json').read_text())
        for episode_id in meta['episode_ids']:
            if not (root/f'episode_{episode_id:04d}'/'result.json').exists():
                raise ValueError(f'计划诊断轮次未完成: {episode_id}')
        for ep in sorted(root.glob('episode_*')):
            if (ep/'predictions.jsonl').exists():
                if ep.name in seen:
                    raise ValueError('输入中包含重复初态编号')
                seen.add(ep.name)
                records.append(summarize_episode(ep,args.output))
                if args.reference:
                    reference=args.reference/ep.name
                    with (ep/'steps.csv').open() as f:
                        actual=list(csv.DictReader(f))
                    with (reference/'steps.csv').open() as f:
                        expected=list(csv.DictReader(f))
                    if actual != expected or json.loads((ep/'result.json').read_text()) != json.loads((reference/'result.json').read_text()):
                        raise ValueError(f'与参考轨迹或结果不一致: {ep.name}')
                    verified.append(ep.name)
    (args.output/'summary.json').write_text(json.dumps({'episodes':records,'reference_identical_episodes':verified,'note':'Selected diagnostic cases; step-boundary contacts only; not population success rates.'},indent=2))
    print('Analyzed episodes:',len(records))


if __name__=='__main__':
    main()
