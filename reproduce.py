"""Recompute the paper's numerical results from final saved predictions."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss, balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss, accuracy_score

ROOT = Path(__file__).resolve().parent
SEEDS = range(42, 47)
METHODS = ['CDLD', 'responseSVD', 'exposureSVD']
CLASSIFIERS = ['frequency', 'B', 'CDLD', 'B_CDLD', 'responseSVD', 'B_responseSVD', 'exposureSVD', 'B_exposureSVD']

def summarize(rows, methods, metrics):
    return {m: {k: dict(mean=float(np.mean(v := [r[k] for r in rows if r['method'] == m])), sd=float(np.std(v, ddof=1)), min=min(v), max=max(v)) for k in metrics} for m in methods}

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def paired_intervals(loss, present, pairs, random_seed, batch_size):
    n = present.shape[1]
    rng = np.random.default_rng(random_seed)
    draws = {pair: [] for pair in pairs}
    for start in range(0, 5000, batch_size):
        weights = rng.multinomial(n, np.full(n, 1/n), size=batch_size).astype(float)
        denominator = weights @ present.T
        for a, b in pairs:
            draws[a, b].extend(np.mean((weights @ (loss[b]-loss[a]).T)/denominator, axis=1).tolist())
    return {a+'_vs_'+b: np.quantile(v, [.025,.975]).tolist() for (a,b),v in draws.items()}

def classification(dataset, mode, output):
    runs = [dict(np.load(ROOT/'predictions'/f'{dataset}_{mode}_{s}.npz')) for s in SEEDS]
    union = np.unique(np.concatenate([z['groups'] for z in runs])); n=len(union)
    present=np.zeros((5,n)); losses={m:np.zeros((5,n)) for m in CLASSIFIERS}; rows=[]; domain={m:[] for m in CLASSIFIERS}
    for r,(seed,z) in enumerate(zip(SEEDS,runs)):
        pos=np.searchsorted(union,z['groups']); y=z['y']; present[r]=np.bincount(pos,minlength=n)
        for m in CLASSIFIERS:
            prob=z[m]; pred=prob.argmax(1)
            rows.append(dict(seed=seed,method=m,n=len(y),logloss=log_loss(y,prob),balanced_accuracy=balanced_accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',zero_division=0)))
            losses[m][r]=np.bincount(pos,weights=-np.log(np.clip(prob[np.arange(len(y)),y],np.finfo(float).eps,1)),minlength=n)
            if dataset=='Eedi':
                cm=np.zeros((4,4),dtype=int);np.add.at(cm,(y,pred),1); domain[m].append(cm)
    summary=summarize(rows,CLASSIFIERS,['logloss','balanced_accuracy','macro_f1'])
    pairs=[('B_CDLD','B'),('B_CDLD','B_responseSVD'),('B_CDLD','B_exposureSVD'),('CDLD','responseSVD'),('CDLD','exposureSVD')]
    intervals=paired_intervals(losses,present,pairs,20261023,50)
    contrasts={a+'_vs_'+b:dict(logloss_reduction=summary[b]['logloss']['mean']-summary[a]['logloss']['mean'],conditional_95_CI=intervals[a+'_vs_'+b]) for a,b in pairs}
    write_csv(output/f'{dataset}_{mode}_runs.csv',rows)
    domains={}
    if dataset=='Eedi':
        for m in CLASSIFIERS[1:]:
            cm=domain[m]; ns=np.array([x.sum(1) for x in cm]); mean=np.mean([x/x.sum(1)[:,None] for x in cm],axis=0)
            domains[m]=dict(mean_row_normalized_confusion=mean.tolist(),recall=np.diag(mean).tolist(),n_min=ns.min(0).tolist(),n_max=ns.max(0).tolist(),mean_prevalence=np.mean(ns/ns.sum(1)[:,None],axis=0).tolist())
    return dict(methods=summary,contrasts=contrasts),domains

def correctness(dataset,output):
    runs=[dict(np.load(ROOT/'predictions'/f'{dataset}_correctness_{s}.npz')) for s in SEEDS]
    methods=[k for k in runs[0] if k not in ('people','y')];rows=[]
    union=np.unique(np.concatenate([z['people'] for z in runs]));present=np.zeros((5,len(union)));losses={m:np.zeros_like(present) for m in methods}
    for r,(seed,z) in enumerate(zip(SEEDS,runs)):
        pos=np.searchsorted(union,z['people']);present[r,pos]=1;y=z['y']
        for m in methods:
            prob=np.clip(z[m].astype(float),np.finfo(float).eps,1-np.finfo(float).eps)
            loss=-(y*np.log(prob)+(1-y)*np.log1p(-prob));losses[m][r,pos]=loss.mean(1)
            rows.append(dict(seed=seed,method=m,logloss=float(loss.mean()),auc=roc_auc_score(y.ravel(),prob.ravel()),brier=brier_score_loss(y.ravel(),prob.ravel()),accuracy=accuracy_score(y.ravel(),prob.ravel()>=.5)))
    summary=summarize(rows,methods,['logloss','auc','brier','accuracy']);pairs=[]
    for kind in ('MLP','bilinear'):
        pairs += [(f'{kind}_CDLD_rate',f'{kind}_{m}_rate') for m in METHODS[1:]]
        pairs += [(f'{kind}_{m}_rate',f'{kind}_{m}_original') for m in METHODS]
        pairs += [(f'{kind}_{m}_rate','item_rate') for m in METHODS]
    intervals=paired_intervals(losses,present,pairs,20261027,100)
    contrasts={a+'_vs_'+b:dict(logloss_reduction=summary[b]['logloss']['mean']-summary[a]['logloss']['mean'],conditional_95_CI=intervals[a+'_vs_'+b]) for a,b in pairs}
    write_csv(output/f'{dataset}_correctness_runs.csv',rows)
    return dict(methods=summary,contrasts=contrasts)

def bilinear_sensitivity(dataset,output):
    runs=[dict(np.load(ROOT/'predictions'/f'{dataset}_bilinear_sensitivity_{s}.npz')) for s in SEEDS]
    methods=[k for k in runs[0] if k not in ('people','y')];rows=[]
    union=np.unique(np.concatenate([z['people'] for z in runs]));present=np.zeros((5,len(union)));losses={m:np.zeros_like(present) for m in methods}
    for r,(seed,z) in enumerate(zip(SEEDS,runs)):
        pos=np.searchsorted(union,z['people']);present[r,pos]=1;y=z['y']
        for m in methods:
            prob=np.clip(z[m].astype(float),np.finfo(float).eps,1-np.finfo(float).eps)
            loss=-(y*np.log(prob)+(1-y)*np.log1p(-prob));losses[m][r,pos]=loss.mean(1)
            rows.append(dict(seed=seed,method=m,logloss=float(loss.mean()),auc=roc_auc_score(y.ravel(),prob.ravel()),brier=brier_score_loss(y.ravel(),prob.ravel()),accuracy=accuracy_score(y.ravel(),prob.ravel()>=.5)))
    summary=summarize(rows,methods,['logloss','auc','brier','accuracy']);pairs=[]
    pairs = [('CDLD_logit', m+'_logit') for m in METHODS[1:]]
    pairs += [(m+'_logit', m+'_raw') for m in METHODS]
    pairs += [(m+'_logit', 'item_rate') for m in METHODS]
    intervals=paired_intervals(losses,present,pairs,20261027,100)
    contrasts={a+'_vs_'+b:dict(logloss_reduction=summary[b]['logloss']['mean']-summary[a]['logloss']['mean'],conditional_95_CI=intervals[a+'_vs_'+b]) for a,b in pairs}
    write_csv(output/f'{dataset}_bilinear_sensitivity_runs.csv',rows)
    return dict(methods=summary,contrasts=contrasts)

def activity(dataset):
    z=np.load(ROOT/'predictions'/f'{dataset}_response_counts.npz');a=z['counts'];q=[z[f'seed{s}'] for s in SEEDS]
    def stat(x):
        return dict(n=len(x),mean=float(x.mean()),p25=float(np.quantile(x,.25)),median=float(np.median(x)),p75=float(np.quantile(x,.75)),min=int(x.min()),max=int(x.max()))
    return dict(all=stat(a),eligible=stat(a[a>=80]),eligible_percent=float((a>=80).mean()*100),evaluation_unique=stat(a[np.unique(np.concatenate(q))]),per_run={str(s):stat(a[u]) for s,u in zip(SEEDS,q)})

def check(actual, expected, path=''):
    if isinstance(actual,dict):
        for k,v in actual.items():check(v,expected[k],path+'/'+k)
    elif isinstance(actual,(list,tuple)):
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10,err_msg=path)
    elif isinstance(actual,(float,int)):
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10,err_msg=path)

def figure(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#168C89','#C47B35','#6569A9'];fig,axs=plt.subplots(2,2,figsize=(9,6.7),layout='constrained')
    for col,ds in enumerate(['EdNet','Eedi']):
        c=results[f'{ds}_item'];d=results[f'{ds}_correctness'];ax=axs[0,col]
        for j,m in enumerate(METHODS):ax.scatter(j,c['methods'][m]['balanced_accuracy']['mean'],s=65,color=colors[j])
        ax.set_ylabel('Classification balanced accuracy');ax.set_ylim(0,1.05);ax.set_title(ds);ax=axs[1,col]
        for k,marker,offset in [('MLP','o',-.09),('bilinear','s',.09)]:
            for j,m in enumerate(METHODS):ax.scatter(j+offset,d['methods'][f'{k}_{m}_rate']['auc']['mean'],s=55,marker=marker,color=colors[j],label=k if j==0 else None)
        ax.set_ylabel('Correctness AUC (with item rate)');ax.set_ylim(.68,.81);ax.legend(frameon=False,fontsize=9)
    for ax in axs.flat:
        ax.set_xticks(range(3),['CDLD','Response\nSVD','Exposure\nSVD']);ax.set_xlim(-.45,2.45);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.savefig(output/'figure4.pdf');fig.savefig(output/'figure4.png',dpi=300);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'output');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    results={}
    for ds,mode in [('EdNet','item'),('EdNet','bundle'),('Eedi','item')]:
        results[f'{ds}_{mode}'],domain=classification(ds,mode,args.output)
        if domain:results['domains']=domain
        print(ds,mode,'complete',flush=True)
    for ds in ['EdNet','Eedi']:
        results[f'{ds}_correctness']=correctness(ds,args.output);print(ds,'correctness complete',flush=True)
    results['samples']={ds:activity(ds) for ds in ['EdNet','Eedi']}
    check(results,json.loads((ROOT/'expected.json').read_text()))
    (args.output/'results.json').write_text(json.dumps(results,indent=2));figure(results,args.output)
    sensitivity={ds:bilinear_sensitivity(ds,args.output) for ds in ['EdNet','Eedi']}
    check(sensitivity,json.loads((ROOT/'sensitivity_expected.json').read_text()))
    (args.output/'bilinear_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
    import subprocess,sys
    subprocess.run([sys.executable,str(ROOT/'draw_figures.py'),'--output',str(args.output.resolve())],check=True)
    (args.output/'verification.json').write_text(json.dumps(dict(passed=True,bootstrap_draws=5000,absolute_tolerance=1e-10)))
    print('DONE: all recomputed statistics match the archived reference',flush=True)
if __name__=='__main__':main()
