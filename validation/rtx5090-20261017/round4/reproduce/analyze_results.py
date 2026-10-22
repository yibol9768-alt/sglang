"""Check all paired results before permitting publication."""
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'remote-results/round4'
if not DATA.is_dir():
    DATA=ROOT.parent/'evidence'


def load(name):
    return json.loads((DATA/name).read_text())


def model(tag):
    return load('sglang-fp8-extended-r4-'+tag+'.json')


def cells(r):
    return {(c['prefix_tokens'],c['batch']):c for c in r['fixed_work_performance']}


def main():
    current=[model('graph-current'),model('graph-current-2')]
    candidate=[model('graph-candidate-1'),model('graph-candidate-2')]
    eager_current=model('eager-current')
    eager_candidate=model('eager-candidate')
    upstream=model('graph-upstream')
    paired_cache=0
    fixed_work_outputs=0
    for a,b in [*zip(current,candidate),(eager_current,eager_candidate)]:
        for x,y in zip(a['cache_cases'],b['cache_cases'],strict=True):
            assert x['runs']==y['runs']
            assert x['cold_warm_max_abs']==y['cold_warm_max_abs']==0
            assert x['warm_repeat_max_abs']==y['warm_repeat_max_abs']==0
            assert all(z==0 for z in y['cached_tokens'][0])
            assert all(z>0 for run in y['cached_tokens'][1:] for z in run)
            paired_cache+=1
        for key,x in cells(a).items():
            y=cells(b)[key]
            assert x['outputs']==y['outputs'],key
            for repetition in y['outputs']:
                assert all(o['completion_tokens']==32 for o in repetition)
                fixed_work_outputs+=len(repetition)
    qa=load('quality-r4-graph-current-2.json')
    qb=load('quality-r4-graph-candidate-1.json')
    assert len(qa['rows'])==len(qb['rows'])==128
    assert qa['source_sha256']==qb['source_sha256']
    assert qa['rows']==qb['rows']
    equality=load('pipeline-equality.json')
    assert len(equality['cases'])==288 and equality['all_exact']
    tests=ET.parse(DATA/'full-fp8-candidate.xml').getroot()
    assert len(tests.findall('.//testcase'))==137
    assert not tests.findall('.//failure') and not tests.findall('.//error')
    thermal={}
    for p in sorted(DATA.glob('*-thermal.json')):
        data=json.loads(p.read_text())
        assert data['exit_code']==0 and data['interrupted'] is None,p
        assert data['power_limit_w']==400 and data['max_temperature_c']<72,p
        thermal[p.name]={k:data[k] for k in ['max_temperature_c','exit_code','interrupted']}
    assert len(thermal)==10,len(thermal)
    rows=[]
    for key in cells(current[0]):
        aa=[cells(r)[key] for r in current]
        bb=[cells(r)[key] for r in candidate]
        av=[v for r in aa for v in r['seconds']]
        bv=[v for r in bb for v in r['seconds']]
        a,b=statistics.median(av),statistics.median(bv)
        ea,eb=cells(eager_current)[key],cells(eager_candidate)[key]
        up=cells(upstream)[key]
        rows.append({'prefix':key[0],'batch':key[1],
            'current_boot_medians_s':[r['median_s'] for r in aa],
            'candidate_boot_medians_s':[r['median_s'] for r in bb],
            'current_samples_s':av,'candidate_samples_s':bv,
            'current_pooled_median_s':a,'candidate_pooled_median_s':b,
            'graph_wall_time_reduction_percent':100*(1-b/a),
            'candidate_vs_current_sample_ranges_disjoint':max(bv)<min(av),
            'upstream_median_s':up['median_s'],
            'candidate_extra_wall_time_vs_upstream_percent':100*(b/up['median_s']-1),
            'eager_current_median_s':ea['median_s'],'eager_candidate_median_s':eb['median_s'],
            'eager_wall_time_reduction_percent':100*(1-eb['median_s']/ea['median_s'])})
    report={'status':'all_required_validation_passed','change':'SM12x deterministic FP8 GEMM num_stages 2 -> 3 only; N64/K128, row tiles16/32/128, warps4 and arithmetic unchanged',
        'runtime':'Official SGLang0.5.23.dev20261020+g55b2aef7ad, Torch2.14.1, Triton3.8.0, FlashInfer0.7.2, sglang-kernel0.4.10; complete files, normal imports',
        'model':'nvidia/Qwen3-8B-FP8@2cebc4c89e25abc17668c81b01dceaf3d8b914d5',
        'native_tests_passed':137,'extra_exact_tensor_comparisons':288,
        'paired_cache_cases_exact':paired_cache,'fixed_work_outputs_exact':fixed_work_outputs,
        'quality_regression':{'count':128,'exact_complete_texts':128,'exact_token_sequences':128,'correct_each_arm':qa['correct'],'missing_final_each_arm':qa['missing_final'],'reused_subset':True},
        'graph_boot_order':'current A1, candidate B1 (+quality), candidate B2, current A2 (+quality), then upstream reference',
        'timing':rows,'thermal':thermal,'max_temperature_c':max(t['max_temperature_c'] for t in thermal.values()),
        'limits':['One RTX5090 at400W, TP1, WSL2; no other GPU, TP2 or torch.compile validation.',
                  'Two boots per PR arm, five measured samples per cell per boot; pooled medians are bounded observations, not broad performance guarantees.',
                  'Upstream Graph reference and eager PR arms have one boot each.',
                  'Cooling <=55C is outside timings; 32 forced output tokens per request, one warmup per cell, cache flushed.',
                  'The128 questions reuse earlier data and are an output-preservation regression, not independent accuracy evidence.',
                  'Official-nightly runtime is not a latest-main rebuild or upstream CI.']}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','native_tests_passed','extra_exact_tensor_comparisons','paired_cache_cases_exact','fixed_work_outputs_exact','quality_regression','max_temperature_c']},indent=2))
    for row in rows:
        print(row['prefix'],row['batch'],'Graph reduction %',round(row['graph_wall_time_reduction_percent'],2),'remaining cost %',round(row['candidate_extra_wall_time_vs_upstream_percent'],2),'Eager reduction %',round(row['eager_wall_time_reduction_percent'],2))


if __name__=='__main__':
    main()
