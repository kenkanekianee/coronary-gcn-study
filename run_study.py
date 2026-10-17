"""Run the Multimodal GCN study on your own computer (same code as the Colab notebook).

Usage (from a terminal / Command Prompt in this folder):
    python run_study.py data        # generate the dataset + quality-control report only (no PyTorch needed)
    python run_study.py smoke       # whole pipeline on 300 graphs, a few minutes: run this first
    python run_study.py benchmark   # time one GCN trial / final fit / RF tuning and project the total hours
    python run_study.py full        # the approved study specification (can take very long on a CPU)
    python run_study.py fast        # lighter settings (a deviation from the spec: report it)
Results are saved in ./results/<profile>/ and every run RESUMES from what is already saved.
"""
import os, sys, time, pickle, shutil, warnings
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results')
FULL = dict(rf_thr=True, rf_trees=[200, 500], rf_depth=[None, 8, 16], rf_leaf=[1, 3, 5],
            gcn_trials=20, gcn_space=dict(layers=[2, 3], hidden=[32, 64, 128], dropout=[0.3, 0.5], lr=[1e-3, 5e-4], lam=[0.5, 1.0, 2.0], pat=[16, 32]),
            gcn_search_epochs=60, gcn_search_patience=10, gcn_epochs=200, gcn_patience=20, log_trials=True)
FAST = dict(FULL, rf_trees=[200], rf_depth=[None, 16], rf_leaf=[1, 3], gcn_trials=10, gcn_search_epochs=40, gcn_search_patience=8, gcn_epochs=120, gcn_patience=15)
SMOKE = dict(rf_thr=True, rf_trees=[30], rf_depth=[None], rf_leaf=[1],
             gcn_trials=1, gcn_space=dict(layers=[2], hidden=[16], dropout=[0.3], lr=[1e-3], lam=[1.0], pat=[8]),
             gcn_search_epochs=4, gcn_search_patience=2, gcn_epochs=6, gcn_patience=3)
DETERMINISTIC = True   # False = faster on GPU but not bit-for-bit reproducible

def get_dataset():
    import gen
    os.makedirs(OUT, exist_ok=True); pk = os.path.join(OUT, 'synthetic_dataset.pkl')
    if os.path.exists(pk): print('Loaded cached dataset'); return pickle.load(open(pk, 'rb'))
    t0 = time.time(); D = gen.build_dataset(2000); pickle.dump(D, open(pk, 'wb')); print('Generated 2000 graphs in %.0f s' % (time.time()-t0)); return D

def qc_report(D):
    y = pd.DataFrame([d['y'] for d in D]); sev = y.sev.value_counts(normalize=True).sort_index().round(3)
    print('Severity (0 low / 1 moderate / 2 high):', sev.to_dict()); print('Plaque primary: %.3f | relational variant: %.3f' % (y.plaque.mean(), y.plaque.mean() if False else y.plaque_rel.mean()))
    assert (sev >= 0.15).all() and 0.30 <= y.plaque.mean() <= 0.60, 'Prevalence outside the design targets'
    rows = [a for d in D for _, a in d['G'].nodes(data=True) if a['plaque']]
    print('HRP feature prevalence among plaque segments: LAP %.3f PR %.3f SC %.3f NRS %.3f' % tuple(np.mean([r[k] for r in rows]) for k in ('lap', 'pr', 'sc', 'nrs')))
    bm = pd.DataFrame([d['bm'] for d in D]); print('Lp(a) >= 50 mg/dL: %.3f' % (bm.Lpa >= 50).mean())

def device_and_torch():
    try: import torch
    except ImportError: sys.exit('PyTorch is not installed. See README.txt (pip install torch torch_geometric shap).')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DETERMINISTIC:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'; torch.use_deterministic_algorithms(True, warn_only=True)
    print('torch', torch.__version__, '| device', dev)
    if dev == 'cpu': print('NOTE: no GPU found. The full study on a CPU can take days; run "benchmark" first and consider "fast".')
    return dev

def run_profile(D, name, cfg, dev, n=None):
    import pipe
    d = os.path.join(OUT, name); data = D if n is None else D[:n]
    res, pred = pipe.run_pipeline(data, dict(cfg), d, pipe.TorchGCN(dev))
    summ, tests = pipe.analyze(res, d); pipe.plot_all(res, pred, summ, d)
    print('\nDone. Tables and figures are in', d); return res, summ, tests

def benchmark(D, dev):
    import gen, pipe
    cfg = dict(FULL, log_trials=False); Db = D[:1500]; RAWb = [pipe.raw_graph(x) for x in Db]
    yb = dict(pl=np.array([x['y']['plaque'] for x in Db]), rel=np.array([x['y']['plaque_rel'] for x in Db]), sv=np.array([x['y']['sev'] for x in Db]))
    trb, vab, teb = np.arange(857), np.arange(857, 1000), np.arange(1000, 1500); g = pipe.TorchGCN(dev); rng = np.random.default_rng(0)
    def timed(c): t = time.time(); g.fit_eval(RAWb, yb, trb, vab, teb, 4, 1, c, rng); return time.time()-t
    t_search = timed(dict(cfg, gcn_trials=2, gcn_epochs=1, gcn_patience=1)) / 2
    t_final = timed(dict(cfg, gcn_trials=1, gcn_search_epochs=1, gcn_search_patience=1))
    Xb = pd.DataFrame([gen.rf_features(x, thr_feats=True, level=4) for x in Db]).values.astype(float)
    t0 = time.time(); pipe.rf_fit_eval(Xb, yb, trb, vab, teb, dict(cfg, _rf_names={4: list(range(Xb.shape[1]))}), 1, 4); t_rf = time.time()-t0
    for nm, c in (('full', FULL), ('fast', FAST)):
        print('Projected total for %-4s: about %.1f hours (upper-bound style estimate)' % (nm, 40*(c['gcn_trials']*t_search + t_final + t_rf)/3600))
    print('GCN search trial %.0f s | GCN final fit %.0f s | RF %.0f s' % (t_search, t_final, t_rf))

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ''
    if mode not in ('data', 'smoke', 'benchmark', 'full', 'fast'): sys.exit(__doc__)
    warnings.filterwarnings('ignore'); sys.path.insert(0, HERE)
    D = get_dataset(); qc_report(D)
    if mode == 'data': return
    dev = device_and_torch()
    if mode == 'smoke':
        shutil.rmtree(os.path.join(OUT, 'smoke'), ignore_errors=True); run_profile(D, 'smoke', SMOKE, dev, n=300); print('SMOKE TEST PASSED')
    elif mode == 'benchmark': benchmark(D, dev)
    else: run_profile(D, mode, FULL if mode == 'full' else FAST, dev)

if __name__ == '__main__':      # this guard is required on Windows (the Random Forest uses worker processes)
    main()
