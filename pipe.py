"""Evaluation pipeline: 5x2cv, GCN (PyTorch Geometric) vs Random Forest, four-model biomarker hierarchy, statistics."""
import os, json, time, pickle, itertools, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score,
                             average_precision_score, brier_score_loss, confusion_matrix)
import gen

SEGN = list(gen.SEG.keys()); ARTS = ['LM','LAD','LCX','RCA']; PTYPES = ['none','calcified','noncalcified','mixed']
NODE_C = ['d_prox','d_dist','length','tort','curv','p0x','p0y','p0z','p1x','p1y','p1z','sten_obs','burden_obs','les_len',
          'les_pos','hu_obs','ri_obs','spot_obs','e_angle','e_dratio']
NODE_B = ['plaque','nrs_obs','prox','has_hu','e_side','has_parent']+['art_'+a for a in ARTS]+['pt_'+p for p in PTYPES]+['seg_'+s for s in SEGN]
IMG_C = ['log_vol_total','log_vol_calc','log_vol_nc','log_vol_lap','pfai']; IMG_B = ['has_pfai']
PAT_CLIN = ['age','male','smoke_former','smoke_current','diabetes','htn']
PAT_KEYS = ['LDL','HDL','TG','HbA1c','eGFR','Lpa','hsCRP','IL6','hsTnT','NTproBNP']
PAT_CIRC = ['LDL','HDL','logTG','HbA1c','eGFR','logLpa','logCRP','logIL6','logTnT','logBNP']
PAT_MISS = [k+'_missing' for k in PAT_KEYS]

# ------------------------------------------------------------------ raw tensors per graph (observed values only)
def raw_graph(d):
    G = d['G']; names = list(G.nodes); idx = {n:i for i,n in enumerate(names)}; N = len(names)
    xc = np.full((N,len(NODE_C)), np.nan); xb = np.zeros((N,len(NODE_B))); xic = np.full((N,5), np.nan); xib = np.zeros((N,1))
    for n in names:
        a = G.nodes[n]; i = idx[n]; ine = list(G.in_edges(n, data=True)); ea = ine[0][2] if ine else None
        row = dict(d_prox=a['d_prox'], d_dist=a['d_dist'], length=a['length'], tort=a['tort'], curv=a['curv'],
                   p0x=a['p0'][0], p0y=a['p0'][1], p0z=a['p0'][2], p1x=a['p1'][0], p1y=a['p1'][1], p1z=a['p1'][2],
                   sten_obs=a['sten_obs'], burden_obs=a['burden_obs'], les_len=a['les_len'] if a['plaque'] else np.nan,
                   les_pos=a['les_pos'], hu_obs=a['hu_obs'], ri_obs=a['ri_obs'], spot_obs=a['spot_obs'] if a['plaque'] else np.nan,
                   e_angle=ea['angle'] if ea else np.nan, e_dratio=ea['d_ratio'] if ea else np.nan)
        xc[i] = [row[k] for k in NODE_C]
        b = {'plaque':a['plaque'], 'nrs_obs':a['nrs_obs'], 'prox':a['prox'], 'has_hu':int(a['hu_obs']==a['hu_obs']),
             'e_side':ea['side'] if ea else 0, 'has_parent':int(ea is not None)}
        for A in ARTS: b['art_'+A] = int(a['art']==A)
        for P in PTYPES: b['pt_'+P] = int(a['ptype']==P)
        for S in SEGN: b['seg_'+S] = int(n==S)
        xb[i] = [b[k] for k in NODE_B]
        im = d['img'][n]; xic[i] = [im['vol_total'], im['vol_calc'], im['vol_nc'], im['vol_lap'], im['pfai']]; xib[i,0] = int(im['pfai']==im['pfai'])
    xic[:,:4] = np.log1p(xic[:,:4])
    ei = np.array([[idx[u] for u,v in G.edges],[idx[v] for u,v in G.edges]], dtype=np.int64).reshape(2,-1)
    b = d['bm']; m = d['miss']
    clin = np.array([b['age'], b['male'], int(b['smoke']==1), int(b['smoke']==2), b['diabetes'], b['htn']], float)
    circ_raw = [b['LDL'],b['HDL'],np.log(b['TG']),b['HbA1c'],b['eGFR'],np.log(b['Lpa']),np.log(b['hsCRP']),np.log(b['IL6']),np.log(b['hsTnT']),np.log(b['NTproBNP'])]
    circ = np.array([np.nan if m[k] else v for k,v in zip(PAT_KEYS, circ_raw)], float)
    miss = np.array([m[k] for k in PAT_KEYS], float)
    return dict(xc=xc, xb=xb, xic=xic, xib=xib, ei=ei, clin=clin, circ=circ, miss=miss, cac=np.array([np.log1p(d['cac'])]))

def pat_names(level):
    n = [] if level<2 else list(PAT_CLIN)
    if level>=3: n += PAT_CIRC + PAT_MISS
    if level>=4: n += ['log_CAC']
    return n
def node_names(level): return NODE_C + NODE_B + (IMG_C + IMG_B if level>=4 else [])

def pat_matrix(RAW, idx, level):
    if level<2: return np.zeros((len(idx),0))
    parts = [np.stack([RAW[i]['clin'] for i in idx])]
    if level>=3: parts += [np.stack([RAW[i]['circ'] for i in idx]), np.stack([RAW[i]['miss'] for i in idx])]
    if level>=4: parts.append(np.stack([RAW[i]['cac'] for i in idx]))
    return np.concatenate(parts,1)
def pat_cont_mask(level):
    m = [True]+[False]*5
    if level>=3: m += [True]*10 + [False]*10
    if level>=4: m += [True]
    return np.array(m) if level>=2 else np.zeros(0,bool)
def fit_pat(X, level):
    if X.shape[1]==0: return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); med = np.nanmedian(X,0)
    med = np.where(np.isnan(med),0,med); Xi = np.where(np.isnan(X),med,X)
    mu = Xi.mean(0); sd = Xi.std(0); c = pat_cont_mask(level)
    mu = np.where(c,mu,0.0); sd = np.where(c,np.where(sd<1e-6,1.0,sd),1.0)
    return dict(med=med, mu=mu, sd=sd)
def apply_pat(X, st):
    if st is None: return X
    return (np.where(np.isnan(X),st['med'],X)-st['mu'])/st['sd']
def fit_node(RAW, idx, level):
    xc = np.concatenate([RAW[i]['xc'] for i in idx])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mu = np.nanmean(xc,0); sd = np.nanstd(xc,0)
        st = dict(mu=np.nan_to_num(mu), sd=np.where(np.nan_to_num(sd)<1e-6,1.0,np.nan_to_num(sd)))
        if level>=4:
            xi = np.concatenate([RAW[i]['xic'] for i in idx]); m2 = np.nan_to_num(np.nanmean(xi,0)); s2 = np.nan_to_num(np.nanstd(xi,0))
            st.update(imu=m2, isd=np.where(s2<1e-6,1.0,s2))
    return st

# ------------------------------------------------------------------ metrics
def metrics_bin(y, p):
    yh = (p>=0.5).astype(int); tn,fp,fn,tp = confusion_matrix(y,yh,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,yh), precision=precision_score(y,yh,zero_division=0), recall=recall_score(y,yh,zero_division=0),
                f1=f1_score(y,yh,zero_division=0), auroc=roc_auc_score(y,p), auprc=average_precision_score(y,p),
                brier=brier_score_loss(y,p), specificity=tn/max(tn+fp,1))
def metrics_sev(y, P):
    yh = P.argmax(1)
    return dict(accuracy=accuracy_score(y,yh), macro_f1=f1_score(y,yh,average='macro',zero_division=0),
                auroc=roc_auc_score(y,P,multi_class='ovr',average='macro'), qwk=cohen_kappa_score(y,yh,weights='quadratic'),
                mae=float(np.mean(np.abs(y-yh))), cost_acc=float(1-np.mean(np.abs(y-yh))/2))
def val_score(y_pl, p_pl, y_sv, P_sv):
    return 0.5*(roc_auc_score(y_pl,p_pl)+cohen_kappa_score(y_sv,P_sv.argmax(1),weights='quadratic'))
def all_metrics(y, pr, idx):
    return dict(pl=metrics_bin(y['pl'][idx],pr['pl']), rel=metrics_bin(y['rel'][idx],pr['rel']), sv=metrics_sev(y['sv'][idx],pr['sv']))

# ------------------------------------------------------------------ Random Forest backend
def rf_fit_eval(X, y, tr, va, te, run, seed, level, explain=False):
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X[tr])
    Xtr, Xva, Xte = imp.transform(X[tr]), imp.transform(X[va]), imp.transform(X[te])
    best = None
    for nt, md, ml in itertools.product(run['rf_trees'], run['rf_depth'], run['rf_leaf']):
        kw = dict(n_estimators=nt, max_depth=md, min_samples_leaf=ml, class_weight='balanced', random_state=seed, n_jobs=-1)
        a = RandomForestClassifier(**kw).fit(Xtr, y['pl'][tr]); b = RandomForestClassifier(**kw).fit(Xtr, y['sv'][tr])
        sc = val_score(y['pl'][va], a.predict_proba(Xva)[:,1], y['sv'][va], b.predict_proba(Xva))
        if best is None or sc>best[0]: best = (sc, kw, dict(trees=nt, depth=md, leaf=ml))
    sc, kw, hp = best
    m_pl = RandomForestClassifier(**kw).fit(Xtr, y['pl'][tr]); m_rel = RandomForestClassifier(**kw).fit(Xtr, y['rel'][tr]); m_sv = RandomForestClassifier(**kw).fit(Xtr, y['sv'][tr])
    pr = dict(pl=m_pl.predict_proba(Xte)[:,1], rel=m_rel.predict_proba(Xte)[:,1], sv=m_sv.predict_proba(Xte))
    out = dict(metrics=all_metrics(y, pr, te), val=float(sc), hp=hp, preds=pr)
    if explain:
        out['explain'] = {}
        try:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(m_pl, Xte, y['pl'][te], n_repeats=3, random_state=seed, scoring='roc_auc')
            out['explain']['rf_perm'] = dict(zip(run['_rf_names'][level], map(float, pi.importances_mean)))
        except Exception as e: out['explain']['rf_perm_error'] = repr(e)
        try:
            import shap
            sv = shap.TreeExplainer(m_pl).shap_values(Xte[:300])
            sv = sv[1] if isinstance(sv, list) else (sv[:,:,1] if np.ndim(sv)==3 else sv)
            out['explain']['rf_shap'] = dict(zip(run['_rf_names'][level], map(float, np.abs(sv).mean(0))))
        except Exception as e: out['explain']['rf_shap_error'] = repr(e)
    return out

# ------------------------------------------------------------------ GCN backend (PyTorch Geometric; needs torch)
def _build_net(in_dim, P, hp):
    import torch, torch.nn as nn
    from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.P = P; s.drop = nn.Dropout(hp['dropout'])
            s.convs = nn.ModuleList([RGCNConv(in_dim if i==0 else hp['hidden'], hp['hidden'], num_relations=2) for i in range(hp['layers'])])
            if P>0: s.pat = nn.Sequential(nn.Linear(P,hp['pat']), nn.ReLU(), nn.Dropout(hp['dropout']), nn.Linear(hp['pat'],hp['pat']), nn.ReLU())
            hin = 2*hp['hidden'] + (hp['pat'] if P>0 else 0)
            s.shared = nn.Sequential(nn.Linear(hin,hp['hidden']), nn.ReLU(), nn.Dropout(hp['dropout']))
            s.h_pl = nn.Linear(hp['hidden'],1); s.h_rel = nn.Linear(hp['hidden'],1); s.h_sv = nn.Linear(hp['hidden'],3)
        def forward(s, x, edge_index, edge_type, batch, patient=None):
            for c in s.convs: x = s.drop(torch.relu(c(x, edge_index, edge_type)))
            h = torch.cat([global_mean_pool(x,batch), global_max_pool(x,batch)], 1)
            if s.P>0: h = torch.cat([h, s.pat(patient)], 1)
            h = s.shared(h)
            return s.h_pl(h).squeeze(-1), s.h_rel(h).squeeze(-1), s.h_sv(h)
    return Net()

def _make_data(RAW, y, idx, level, nst, pst, P):
    import torch
    from torch_geometric.data import Data
    out = []
    for k, i in enumerate(idx):
        r = RAW[i]; xc = np.nan_to_num((r['xc']-nst['mu'])/nst['sd'], nan=0.0); feats = [xc, r['xb']]
        if level>=4: feats += [np.nan_to_num((r['xic']-nst['imu'])/nst['isd'], nan=0.0), r['xib']]
        ei = torch.tensor(r['ei'], dtype=torch.long); n_e = ei.shape[1]
        d = Data(x=torch.tensor(np.concatenate(feats,1), dtype=torch.float32), edge_index=torch.cat([ei, ei.flip(0)],1),
                 edge_type=torch.cat([torch.zeros(n_e,dtype=torch.long), torch.ones(n_e,dtype=torch.long)]),
                 y_pl=torch.tensor([float(y['pl'][i])]), y_rel=torch.tensor([float(y['rel'][i])]), y_sv=torch.tensor([int(y['sv'][i])]))
        d.patient = torch.tensor(P[k:k+1], dtype=torch.float32)
        out.append(d)
    return out

def _predict(model, data, device):
    import torch
    from torch_geometric.loader import DataLoader
    model.eval(); a, b, c = [], [], []
    with torch.no_grad():
        for bt in DataLoader(data, batch_size=128, shuffle=False):
            bt = bt.to(device); lp, lr, ls = model(bt.x, bt.edge_index, bt.edge_type, bt.batch, bt.patient)
            a.append(torch.sigmoid(lp).cpu().numpy()); b.append(torch.sigmoid(lr).cpu().numpy()); c.append(torch.softmax(ls,1).cpu().numpy())
    return dict(pl=np.concatenate(a), rel=np.concatenate(b), sv=np.concatenate(c))

def _train(tr, va, in_dim, P, hp, seed, max_epochs, patience, device, y_tr):
    import torch, torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed); np.random.seed(seed)
    model = _build_net(in_dim, P, hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp['lr']); sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    def pw(v): v = np.asarray(v); return torch.tensor(max((v==0).sum(),1)/max((v==1).sum(),1), dtype=torch.float32, device=device)
    pw_pl, pw_rel = pw(y_tr['pl']), pw(y_tr['rel'])
    cnt = np.bincount(y_tr['sv'], minlength=3).astype(float); svw = torch.tensor(len(y_tr['sv'])/(3*np.maximum(cnt,1)), dtype=torch.float32, device=device)
    def loss_fn(out, b):
        lp, lr, ls = out
        return (F.binary_cross_entropy_with_logits(lp, b.y_pl, pos_weight=pw_pl) + 0.5*F.binary_cross_entropy_with_logits(lr, b.y_rel, pos_weight=pw_rel)
                + hp['lam']*F.cross_entropy(ls, b.y_sv, weight=svw))
    ltr = DataLoader(tr, batch_size=32, shuffle=True); lva = DataLoader(va, batch_size=128, shuffle=False)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        for b in ltr:
            b = b.to(device); opt.zero_grad(); loss = loss_fn(model(b.x,b.edge_index,b.edge_type,b.batch,b.patient), b); loss.backward(); opt.step()
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for b in lva:
                b = b.to(device); tot += loss_fn(model(b.x,b.edge_index,b.edge_type,b.batch,b.patient), b).item()*b.num_graphs; n += b.num_graphs
        vl = tot/n; sched.step(vl)
        if vl < best-1e-4: best, bad = vl, 0; best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model

def _integrated_gradients(model, data, device, steps=32):
    """Integrated gradients with a zero (= training-mean) baseline, computed directly in PyTorch so graph batching is handled correctly."""
    import torch
    from torch_geometric.loader import DataLoader
    model.eval(); b = next(iter(DataLoader(data, batch_size=len(data)))).to(device)
    gx = torch.zeros_like(b.x); gp = torch.zeros_like(b.patient)
    for k in range(1, steps+1):
        al = k/steps; x = (al*b.x).clone().requires_grad_(True); p = (al*b.patient).clone().requires_grad_(True)
        lp, _, _ = model(x, b.edge_index, b.edge_type, b.batch, p); lp.sum().backward()
        gx += x.grad/steps; gp += (p.grad/steps if p.grad is not None else 0)
    return (b.x*gx).abs().mean(0).cpu().numpy(), (b.patient*gp).abs().mean(0).cpu().numpy() if b.patient.shape[1]>0 else np.zeros(0)

class TorchGCN:
    def __init__(self, device): self.device = device
    def fit_eval(self, RAW, y, tr, va, te, level, seed, run, rng, explain=False):
        nst = fit_node(RAW, tr, level); pst = fit_pat(pat_matrix(RAW, tr, level), level)
        Pm = {k: apply_pat(pat_matrix(RAW, ix, level), pst) for k, ix in (('tr',tr),('va',va),('te',te))}
        D = {k: _make_data(RAW, y, ix, level, nst, pst, Pm[k]) for k, ix in (('tr',tr),('va',va),('te',te))}
        in_dim = D['tr'][0].x.shape[1]; P = Pm['tr'].shape[1]; ytr = {k: y[k][tr] for k in y}
        grid = [dict(layers=a, hidden=b, dropout=c, lr=d, lam=e, pat=f) for a,b,c,d,e,f in itertools.product(*[run['gcn_space'][k] for k in ('layers','hidden','dropout','lr','lam','pat')])]
        pick = rng.choice(len(grid), size=min(run['gcn_trials'], len(grid)), replace=False)
        best = None; t_fit = time.time()
        for t, gi in enumerate(pick):
            hp = grid[gi]; m = _train(D['tr'], D['va'], in_dim, P, hp, seed+t, run['gcn_search_epochs'], run['gcn_search_patience'], self.device, ytr)
            pr = _predict(m, D['va'], self.device); sc = val_score(y['pl'][va], pr['pl'], y['sv'][va], pr['sv'])
            if best is None or sc>best[0]: best = (sc, hp)
            if run.get('log_trials'): print(f"    Model {level} GCN trial {t+1}/{len(pick)}: val {sc:.3f} ({time.time()-t_fit:.0f}s since start of this fit)", flush=True)
        sc, hp = best
        model = _train(D['tr'], D['va'], in_dim, P, hp, seed+999, run['gcn_epochs'], run['gcn_patience'], self.device, ytr)
        pr = _predict(model, D['te'], self.device)
        out = dict(metrics=all_metrics(y, pr, te), val=float(sc), hp=hp, preds=pr)
        if explain:
            out['explain'] = {}
            try:
                nx_, px_ = _integrated_gradients(model, D['te'][:200], self.device)
                out['explain']['gcn_ig_node'] = dict(zip(node_names(level), map(float, nx_)))
                out['explain']['gcn_ig_patient'] = dict(zip(pat_names(level), map(float, px_)))
            except Exception as e: out['explain']['gcn_ig_error'] = repr(e)
        return out

# ------------------------------------------------------------------ orchestration (resumable)
def run_pipeline(D, run, out_dir, gcn, verbose=True):
    os.makedirs(out_dir, exist_ok=True); rp = os.path.join(out_dir,'results.json')
    results = json.load(open(rp)) if os.path.exists(rp) else {}
    pp = os.path.join(out_dir,'preds_model4.pkl'); PRED = pickle.load(open(pp,'rb')) if os.path.exists(pp) else {}
    RAW = [raw_graph(d) for d in D]; n = len(D)
    y = dict(pl=np.array([d['y']['plaque'] for d in D]), rel=np.array([d['y']['plaque_rel'] for d in D]), sv=np.array([d['y']['sev'] for d in D]))
    strat = y['sv']*2 + y['pl']
    RFX = {L: pd.DataFrame([gen.rf_features(d, thr_feats=run['rf_thr'], level=L) for d in D]) for L in (1,2,3,4)}
    run['_rf_names'] = {L: list(RFX[L].columns) for L in RFX}
    RFX = {L: v.values.astype(float) for L, v in RFX.items()}
    t0 = time.time()
    for rep in range(5):
        for half, (trall, te) in enumerate(StratifiedKFold(2, shuffle=True, random_state=rep).split(np.zeros(n), strat)):
            tr, va = train_test_split(trall, test_size=0.143, stratify=strat[trall], random_state=rep*10+half)
            for level in (1,2,3,4):
                key = f"{rep}|{half}|{level}"
                if key in results: continue
                seed = 100000*rep + 10000*half + 1000*level; rng = np.random.default_rng(seed); expl = (rep==0 and half==0 and level==4)
                g = gcn.fit_eval(RAW, y, tr, va, te, level, seed, run, rng, explain=expl)
                r = rf_fit_eval(RFX[level], y, tr, va, te, run, seed, level, explain=expl)
                results[key] = dict(gcn=dict(metrics=g['metrics'], val=g['val'], hp=g['hp']), rf=dict(metrics=r['metrics'], val=r['val'], hp=r['hp']))
                if expl: results['explain'] = {**g.get('explain',{}), **r.get('explain',{})}
                if level==4:
                    PRED[f"{rep}|{half}"] = dict(y_pl=y['pl'][te], y_sv=y['sv'][te], gcn=g['preds'], rf=r['preds'])
                    pickle.dump(PRED, open(pp,'wb'))
                json.dump(results, open(rp,'w'))
                if verbose: print(f"[{(time.time()-t0)/60:6.1f} min] rep {rep} half {half} Model {level}: GCN val {g['val']:.3f} | RF val {r['val']:.3f}", flush=True)
    return results, PRED

# ------------------------------------------------------------------ analysis
def collect(results, algo, level, task, metric):
    a = np.full((5,2), np.nan)
    for rep in range(5):
        for half in range(2):
            r = results.get(f"{rep}|{half}|{level}")
            if r: a[rep,half] = r[algo]['metrics'][task][metric]
    return a

def test_row(name, diffs, family):
    t = gen.cv52_tests(diffs); s2 = ((diffs-diffs.mean(1,keepdims=True))**2).sum(1)
    return dict(family=family, comparison=name, mean_diff=float(np.mean(diffs)), se_5x2cv=float(np.sqrt(np.mean(s2))),
                t=t['t'], p_t=t['p_t'], F=t['F'], p_F=t['p_F'])

def analyze(results, out_dir):
    rows = []
    for algo in ('gcn','rf'):
        for level in (1,2,3,4):
            for task, mets in (('pl',['accuracy','precision','recall','f1','auroc','auprc','specificity','brier']), ('rel',['auroc','f1']), ('sv',['accuracy','macro_f1','auroc','qwk','mae','cost_acc'])):
                for m in mets:
                    v = collect(results, algo, level, task, m).ravel()
                    rows.append(dict(algorithm=algo.upper(), model=level, outcome=task, metric=m, mean=np.nanmean(v), sd=np.nanstd(v, ddof=1)))
    summ = pd.DataFrame(rows); summ.to_csv(os.path.join(out_dir,'summary_metrics.csv'), index=False)
    tests = []
    A = [test_row(f"plaque {m}: GCN - RF (Model 4)", collect(results,'gcn',4,'pl',m)-collect(results,'rf',4,'pl',m), 'A: plaque-feature class') for m in ['accuracy','precision','recall','f1','auroc']]
    B = [test_row(f"severity {m}: GCN - RF (Model 4)", collect(results,'gcn',4,'sv',m)-collect(results,'rf',4,'sv',m), 'B: stenosis-severity class') for m in ['accuracy','macro_f1','auroc','qwk','mae','cost_acc']]
    C = [test_row(f"{al.upper()} {tk} {m}: Model 4 - Model 1", collect(results,al,4,tk,m)-collect(results,al,1,tk,m), 'C: incremental value')
         for al,tk,m in (('gcn','pl','auroc'),('rf','pl','auroc'),('gcn','sv','qwk'),('rf','sv','qwk'))]
    for fam in (A,B,C):
        adj = gen.holm([r['p_F'] for r in fam])
        for r, a in zip(fam, adj): r['p_F_holm'] = float(a); r['significant_at_.05'] = bool(a<0.05)
    X = [test_row(f"{tk} {m}: GCN - RF (Model {L})", collect(results,'gcn',L,tk,m)-collect(results,'rf',L,tk,m), 'X: exploratory (uncorrected)')
         for L in (1,2,3) for tk,m in (('pl','auroc'),('sv','qwk'))]
    X += [test_row(f"relational plaque {m}: GCN - RF (Model 4)", collect(results,'gcn',4,'rel',m)-collect(results,'rf',4,'rel',m), 'X: exploratory (uncorrected)') for m in ('auroc','f1')]
    for r in X: r['p_F_holm'] = np.nan; r['significant_at_.05'] = None
    for r, h in zip(A, ['H01','H02','H03','H04','H05']): r['hypothesis'] = h
    for r, h in zip(B, ['H06','H07','H08','H09','H010','H011']): r['hypothesis'] = h
    for r, h in zip(C, ['H012','H013','H014','H015']): r['hypothesis'] = h
    for r in X: r['hypothesis'] = 'exploratory'
    tests = pd.DataFrame(A+B+C+X); tests = tests[['hypothesis']+[c for c in tests.columns if c!='hypothesis']]; tests.to_csv(os.path.join(out_dir,'hypothesis_tests.csv'), index=False)
    return summ, tests

def plot_all(results, PRED, summ, out_dir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    for a, (task, met, ttl) in zip(ax, (('pl','auroc','Plaque-feature class: AUROC'), ('sv','qwk','Stenosis-severity class: QWK'))):
        for al in ('GCN','RF'):
            s = summ[(summ.algorithm==al)&(summ.outcome==task)&(summ.metric==met)].sort_values('model')
            a.errorbar(s.model, s['mean'], yerr=s.sd, marker='o', capsize=3, label=al)
        a.set_xticks([1,2,3,4]); a.set_xlabel('Model (1 anatomy; 2 +clinical; 3 +circulating; 4 +imaging)'); a.set_title(ttl); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'performance_by_model.png'), dpi=150); plt.close()
    fig, ax = plt.subplots(1,2, figsize=(9,4))
    for a, al in zip(ax, ('gcn','rf')):
        yt = np.concatenate([v['y_pl'] for v in PRED.values()]); pp = np.concatenate([v[al]['pl'] for v in PRED.values()])
        fr, mp = calibration_curve(yt, pp, n_bins=10, strategy='quantile'); a.plot(mp, fr, 'o-'); a.plot([0,1],[0,1],'k--'); a.set_title(f'{al.upper()} plaque calibration (Model 4)'); a.set_xlabel('Predicted'); a.set_ylabel('Observed')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'calibration_model4.png'), dpi=150); plt.close()
    ex = results.get('explain', {})
    for key, ttl in (('rf_shap','RF SHAP (mean |value|)'),('rf_perm','RF permutation importance (AUROC drop)'),('gcn_ig_node','GCN integrated gradients: node features'),('gcn_ig_patient','GCN integrated gradients: patient features')):
        if key in ex and ex[key]:
            s = pd.Series(ex[key]).sort_values(ascending=False).head(15)[::-1]; plt.figure(figsize=(6,4)); s.plot.barh(); plt.title(ttl); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, key+'.png'), dpi=150); plt.close()
