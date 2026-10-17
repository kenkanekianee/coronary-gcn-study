"""Synthetic coronary-tree graph generator (Multimodal GCN proposal, Part 1).
Every PARAM tagged STUDY_DEFINED is a modeling assumption; NEEDS_SOURCE marks values to replace with cited ones.
"""
import numpy as np, networkx as nx
from scipy.special import expit

# ---- segment table: name -> (parent, artery group, base prox diameter mm, base length mm, proximal?) -- NEEDS_SOURCE
SEG = {
 'RCA-p':('', 'RCA',3.4,22,1), 'RCA-m':('RCA-p','RCA',3.0,28,0), 'RCA-d':('RCA-m','RCA',2.6,30,0),
 'R-PDA':('RCA-d','RCA',2.0,35,0), 'R-PLB':('RCA-d','RCA',1.9,30,0),
 'LM':('', 'LM',4.2,12,1),
 'LAD-p':('LM','LAD',3.5,18,1), 'LAD-m':('LAD-p','LAD',2.9,25,0), 'LAD-d':('LAD-m','LAD',2.0,30,0),
 'D1':('LAD-p','LAD',2.2,35,0), 'D2':('LAD-m','LAD',1.8,30,0),
 'LCX-p':('LM','LCX',3.1,20,1), 'LCX-d':('LCX-p','LCX',2.4,32,0),
 'OM1':('LCX-p','LCX',2.1,35,0), 'OM2':('LCX-d','LCX',1.8,30,0),
 'L-PL':('LCX-d','LCX',1.9,28,0), 'L-PDA':('LCX-d','LCX',1.8,30,0), 'RAMUS':('LM','LCX',2.3,35,0)}
CONT = {('RCA-p','RCA-m'),('RCA-m','RCA-d'),('LM','LAD-p'),('LAD-p','LAD-m'),('LAD-m','LAD-d'),('LCX-p','LCX-d')}
ART_OFF = {'LM':-0.3,'LAD':0.15,'LCX':-0.1,'RCA':0.05}   # STUDY_DEFINED segment-propensity offsets

CFG = dict(
 p_dom=(0.80,0.10,0.10),                     # right/left/co-dominant  STUDY_DEFINED, NEEDS_SOURCE
 p_branch={'D1':0.9,'D2':0.7,'OM1':0.9,'OM2':0.5,'RAMUS':0.3,'R-PLB':0.8,'L-PL':0.7},
 w_burden=0.75,        # share of segment propensity driven by the patient-level latent burden B
 t_plaque=0.2,         # plaque present if propensity > t_plaque
 g0=-1.1, g1=0.9, kappa=8.0,   # stenosis mean = expit(g0+g1*(s - t_plaque)), Beta concentration kappa
 hu_mu0=48.0, hu_slope=-9.0, hu_sd=18.0,       # true minimum plaque attenuation (HU) for non-calcified/mixed
 ri_mu0=1.0, ri_slope=0.035, ri_sd=0.07,       # true remodeling index (diameter ratio)
 p_small_ca=(0.15,0.12), p_nrs=0.25,           # P(small calcified fragment)=a+b*expit(s); P(NRS | LAP, non-calc)
 noise=dict(hu=10.0, ri=0.03, sten=4.0, spot=0.3, burden=0.05, nrs_sens=0.90, nrs_spec=0.97, spot_sens=0.90, spot_fp=0.03),
 rel_rule='cluster_single_features',   # or 'paper_original' (redundant with the primary label; see notes)
)

def _unit(v): return v/np.linalg.norm(v)
def _rot(d, ang_deg, rng):
    a=np.deg2rad(ang_deg); r=rng.normal(size=3); r-=r.dot(d)*d; r=_unit(r)
    return _unit(np.cos(a)*d+np.sin(a)*r)

def make_graph(seed, cfg=CFG):
    rng = np.random.default_rng(seed)
    dom = rng.choice(['right','left','co'], p=cfg['p_dom'])
    names = []
    for n,(par,art,d,L,prox) in SEG.items():
        if n in ('R-PDA',) and dom=='left': continue
        if n=='L-PDA' and dom=='right': continue
        if n in cfg['p_branch'] and rng.random()>cfg['p_branch'][n]:
            continue
        if n=='R-PLB' and dom=='left': continue
        if n=='L-PL' and dom=='right' and rng.random()<0.5: continue
        names.append(n)
    size = rng.normal(1.0,0.07)                # patient-level vessel size factor
    B = rng.normal()                            # patient-level latent disease burden
    G = nx.DiGraph(); pos={}; dirs={}
    for n in names:
        par,art,d0,L0,prox = SEG[n]
        if par=='' : # root
            d = _unit(rng.normal(size=3)); p0=np.zeros(3)+ (0 if n=='LM' else rng.normal(scale=15,size=3))
            dp = d0*size*rng.lognormal(0,0.08)
        else:
            ang = rng.normal(15,8) if (par,n) in CONT else rng.normal(60,15)
            ang = float(np.clip(ang, 5 if (par,n) in CONT else 30, 120))
            d = _rot(dirs[par], ang, rng); p0 = pos[par][1]
            pd = G.nodes[par]['d_dist']
            dp = min(d0*size*rng.lognormal(0,0.08), (0.97 if (par,n) in CONT else 0.8)*pd)
        L = L0*rng.lognormal(0,0.12); tort = 1+abs(rng.normal(0,0.06)); chord=L/tort
        p1 = p0+d*chord; dd = dp*rng.uniform(0.80,0.95)
        pos[n]=(p0,p1); dirs[n]=d
        G.add_node(n, art=art, prox=prox, d_prox=dp, d_dist=dd, length=L, tort=tort,
                   curv=np.sqrt(max(tort**2-1,0))/L, p0=p0, p1=p1)
        if par!='':
            G.add_edge(par,n, angle=ang, d_ratio=dd/ G.nodes[par]['d_dist'] if False else dp/G.nodes[par]['d_dist'],
                       cl_dist=G.nodes[par]['length'], side=int((par,n) not in CONT))
    # ---- disease ----
    nz = cfg['noise']
    for n in G.nodes:
        a = G.nodes[n]
        s = cfg['w_burden']*B + np.sqrt(1-cfg['w_burden']**2)*rng.normal() + ART_OFF[a['art']] + 0.25*a['prox']
        a['s']=s
        has = s>cfg['t_plaque']
        a.update(plaque=int(has), sten=0., burden=0., ptype='none', hu=np.nan, ri=np.nan, small_ca=0, spot=np.nan, nrs=0,
                 les_len=0., les_pos=np.nan, calc_frac=0., lap=0, pr=0, sc=0, hrp_n=0)
        if not has: continue
        m = expit(cfg['g0']+cfg['g1']*(s-cfg['t_plaque']))
        sten = 100*rng.beta(m*cfg['kappa'],(1-m)*cfg['kappa'])
        sten = float(np.clip(sten,5,99))
        burden = float(np.clip(0.25+0.5*sten/100+rng.normal(0,0.08),0.1,0.95))
        pt = rng.choice(['calcified','noncalcified','mixed'], p=np.array([0.25+0.1*expit(-s),0.35+0.1*expit(s),0.40])/ (1.0+0.1*expit(-s)+0.1*expit(s)))
        a.update(sten=sten, burden=burden, ptype=pt, les_len=float(np.clip(rng.lognormal(np.log(12),0.4),4,40)), les_pos=float(rng.uniform(0.1,0.9)))
        a['calc_frac']={'calcified':rng.uniform(0.8,1.0),'noncalcified':0.0,'mixed':rng.uniform(0.25,0.6)}[pt]
        if pt!='calcified':
            a['hu']=float(rng.normal(cfg['hu_mu0']+cfg['hu_slope']*s, cfg['hu_sd']))
        a['ri']=float(rng.normal(cfg['ri_mu0']+cfg['ri_slope']*s, cfg['ri_sd']))
        if pt!='noncalcified' and rng.random()<cfg['p_small_ca'][0]+cfg['p_small_ca'][1]*expit(s):
            a['small_ca']=1; a['spot']=float(rng.uniform(0.8,3.0))
        a['lap']=int(pt!='calcified' and a['hu']<30)
        if a['lap'] and pt=='noncalcified' and rng.random()<cfg['p_nrs']: a['nrs']=1
        a['pr']=int(a['ri']>1.1); a['sc']=int(a['small_ca']==1 and a['spot']<=3.0)
        a['hrp_n']=a['lap']+a['pr']+a['sc']+a['nrs']
        # observed (noisy) measurements: what the models see
    for n in G.nodes:
        a=G.nodes[n]
        a['sten_obs']=float(np.clip(a['sten']+rng.normal(0,nz['sten']),0,100)) if a['plaque'] else float(np.clip(rng.normal(0,nz['sten']/2),0,100))
        a['burden_obs']=float(np.clip(a['burden']+rng.normal(0,nz['burden']),0,1)) if a['plaque'] else 0.0
        a['hu_obs']=a['hu']+rng.normal(0,nz['hu']) if a['plaque'] and not np.isnan(a['hu']) else np.nan
        a['ri_obs']=a['ri']+rng.normal(0,nz['ri']) if a['plaque'] else np.nan
        det = (rng.random()<nz['spot_sens']) if a['sc'] else (rng.random()<nz['spot_fp'] and a['plaque'] and a['ptype']!='noncalcified')
        a['spot_obs']=float(max(0.3,(a['spot'] if a['sc'] else rng.uniform(0.8,3.0))+rng.normal(0,nz['spot']))) if det else 0.0
        a['nrs_obs']=int((rng.random()<nz['nrs_sens']) if a['nrs'] else (rng.random()>nz['nrs_spec']))
    return G, dict(B=B, dom=dom, size=size, seed=seed)

def labels(G, cfg=CFG):
    maxs = max(G.nodes[n]['sten'] for n in G.nodes)
    lm = G.nodes['LM']['sten'] if 'LM' in G.nodes else 0
    sev = 2 if (maxs>=70 or lm>50) else (1 if maxs>=50 else 0)
    prim = int(any(G.nodes[n]['hrp_n']>=2 for n in G.nodes))
    # relational label: primary OR adjacent (parent-daughter) pair of segments that each carry exactly ONE (different) HRP feature
    def feats(a): return {k for k,v in (('lap',a['lap']),('pr',a['pr']),('sc',a['sc']),('nrs',a['nrs'])) if v}
    rel = prim
    if cfg['rel_rule']=='cluster_single_features':
        for u,v in G.edges:
            fu,fv=feats(G.nodes[u]),feats(G.nodes[v])
            if len(fu)==1 and len(fv)==1 and fu!=fv: rel=1; break
    else:  # paper_original: adjacent pair with each >=1 and combined >=4 (always implies primary => redundant)
        for u,v in G.edges:
            a,b=G.nodes[u]['hrp_n'],G.nodes[v]['hrp_n']
            if a>=1 and b>=1 and a+b>=4: rel=1; break
    return dict(sev=sev, plaque=prim, plaque_rel=rel, max_sten=maxs)

def qc(G):
    ok=True; msg=[]
    roots=[n for n in G if G.in_degree(n)==0]
    if set(roots)-{'LM','RCA-p'}: ok=False; msg.append('bad root')
    for comp in nx.weakly_connected_components(G):
        if not nx.is_tree(G.subgraph(comp).to_undirected()): ok=False; msg.append('cycle')
    for n,a in G.nodes(data=True):
        if not (a['d_prox']>0 and a['d_dist']>0 and a['d_dist']<=a['d_prox']): ok=False; msg.append('taper')
    for u,v,e in G.edges(data=True):
        if not (e['d_ratio']<=1.0): ok=False; msg.append('d_ratio')
        if not (5<=e['angle']<=120): ok=False; msg.append('angle')
    pts={n:G.nodes[n]['p1'] for n in G}
    for u in G:
        for v in G:
            if u<v and not G.has_edge(u,v) and not G.has_edge(v,u) and np.linalg.norm(pts[u]-pts[v])<1.0:
                ok=False; msg.append('intersect')
    return ok, msg

# ======================= biomarkers (generated AFTER anatomy/plaque; own RNG stream) =======================
BM_MISS = {'LDL':.03,'HDL':.03,'TG':.03,'HbA1c':.03,'eGFR':.03,'hsCRP':.08,'IL6':.30,'hsTnT':.15,'NTproBNP':.30,'Lpa':.25}
# name: (primary latent, secondary latent, sign_primary, scale: total log-SD or SD)  STUDY_DEFINED
BM_LOG = {'TG':('lipid','trad',+1,np.log(1.6),np.log(112)), 'Lpa':('lipid','plaque',+1,1.5,np.log(14)),
          'hsCRP':('infl','plaque',+1,1.0,np.log(1.9)), 'IL6':('infl','trad',+1,0.8,np.log(1.45)),
          'hsTnT':('inj','plaque',+1,0.7,np.log(7.0)), 'NTproBNP':('inj','trad',+1,1.0,np.log(80))}

def biomarkers(G, meta, seed):
    r = np.random.default_rng([seed, 1])           # separate stream: never alters anatomy/labels
    age = float(np.clip(r.normal(58,10),40,80)); male = int(r.random()<0.525)
    smoke = int(r.choice([0,1,2], p=[.5,.25,.25]))   # never/former/current
    zage=(age-58)/10
    trad_raw = 0.5*zage+0.3*male+0.3*(smoke>0)
    diab = int(r.random()<expit(-1.3+0.4*zage+0.5*trad_raw))
    htn  = int(r.random()<expit(-0.1+0.4*zage+0.4*trad_raw))
    trad = (trad_raw+0.3*diab+0.3*htn+r.normal(0,0.6))/0.9
    tot_burden = sum(G.nodes[n]['burden'] for n in G if G.nodes[n]['plaque'])
    L = {'trad':trad, 'lipid':r.normal(), 'infl':0.3*trad+r.normal(0,0.95),
         'plaque': (tot_burden-2.97)/2.24, 'inj':0.3*zage+r.normal(0,0.95)}
    out = dict(age=age, male=male, smoke=smoke, diabetes=diab, htn=htn)
    ageeff, sexeff = 0.1*zage, 0.1*(1 if male else -1)
    def lat(bm, sd_total, sgn=+1):
        p,s=BM_LOG[bm][0],BM_LOG[bm][1]
        return sd_total*(sgn*0.6*L[p]+0.3*L[s]+0.74*r.normal()+ageeff*(bm!='Lpa')+ sexeff*(bm not in ('Lpa',)))
    for bm,(p,s,sg,sd,mu) in BM_LOG.items():
        out[bm]=float(np.exp(mu+lat(bm,sd)))
    out['hsCRP']=float(min(out['hsCRP'],10.0))
    out['LDL']=float(np.clip(120+35*(0.6*L['lipid']+0.3*L['trad']+0.74*r.normal()),40,250))
    mu_h=45 if male else 55; sd_h=11 if male else 13
    out['HDL']=float(np.clip(mu_h+sd_h*(-0.6*L['lipid']-0.3*L['trad']*0+0.74*r.normal())-3*diab,15,110))
    out['HbA1c']=float(np.clip(7.5+1.2*r.normal(),6.5,14)) if diab else float(np.clip(5.5+0.4*r.normal(),4.0,6.4))
    out['eGFR']=float(np.clip(95-25*(age-45)/30-4*diab-3*htn+15*(0.6*(-L['inj'])+0.3*(-L['trad'])+0.74*r.normal()),15,130))
    # missing completely at random; indicator recorded; imputation happens inside each CV fold
    miss = {k:int(r.random()<p) for k,p in BM_MISS.items()}
    # lesion-level imaging biomarkers (true, then measured with small noise)
    calc_vol=0.; seg_img={}
    for n in G:
        a=G.nodes[n]
        if not a['plaque']: seg_img[n]=dict(vol_total=0.,vol_calc=0.,vol_nc=0.,vol_lap=0.,pfai=np.nan); continue
        dmid=0.5*(a['d_prox']+a['d_dist']); vol=a['burden_obs']*np.pi*(dmid/2)**2*a['les_len']
        vc=vol*a['calc_frac']; vnc=vol-vc; vlap=vnc*float(np.clip(0.15+0.25*(a['hu_obs']<30)+r.normal(0,0.05),0,1)) if a['hu_obs']==a['hu_obs'] else 0.
        pf=np.nan
        if a['prox'] and a['art'] in ('RCA','LAD','LCX'):
            pf=float(np.clip(-78+8*r.normal()+3*L['infl']+2*a['burden_obs'],-190,-30))
        seg_img[n]=dict(vol_total=vol,vol_calc=vc,vol_nc=vnc,vol_lap=vlap,pfai=pf); calc_vol+=vc
    cac = 0. if calc_vol<=0 else float(6.0*calc_vol*np.exp(r.normal(0,0.3)))   # STUDY_DEFINED mapping, patient total
    return out, miss, seg_img, cac

# ======================= dataset assembly =======================
def build_dataset(n=2000, seed0=1000, cfg=CFG, max_tries=10):
    data=[]
    for i in range(n):
        for t in range(max_tries):                      # reject-and-regenerate on QC failure (new seed)
            seed=seed0+i+100000*t
            G,meta=make_graph(seed,cfg); ok,_=qc(G)
            if ok: break
        lab=labels(G,cfg); bm,miss,img,cac=biomarkers(G,meta,seed)
        data.append(dict(seed=seed,G=G,meta=meta,y=lab,bm=bm,miss=miss,img=img,cac=cac))
    return data

# ======================= Random Forest feature vector (observed measurements only) =======================
ART=['LM','LAD','LCX','RCA']
def rf_features(d, thr_feats=True, level=4):
    """Aggregated anatomy (+ biomarkers up to `level`). thr_feats: CAD-RADS thresholds applied to OBSERVED (noisy) values."""
    G=d['G']; N=list(G.nodes(data=True)); f={}
    f['n_seg']=len(N)
    for k,key in (('diam','d_prox'),('len','length'),('tort','tort'),('curv','curv'),('sten','sten_obs')):
        v=np.array([a[key] for _,a in N]); f[k+'_mean']=v.mean(); f[k+'_max']=v.max(); f[k+'_sd']=v.std()
    pl=[a for _,a in N if a['plaque']]
    f['n_plaque']=len(pl); f['burden_sum']=sum(a['burden_obs'] for a in pl); f['les_len_sum']=sum(a['les_len'] for a in pl)
    for t in ('calcified','noncalcified','mixed'): f['frac_'+t]=sum(a['ptype']==t for a in pl)/max(len(pl),1)
    for g in ART:
        v=[a['sten_obs'] for _,a in N if a['art']==g]; f[g+'_sten_max']=max(v) if v else 0; f[g+'_sten_mean']=np.mean(v) if v else 0; f[g+'_n']=len(v)
    f['n_sten50']=sum(a['sten_obs']>=50 for _,a in N); f['n_sten70']=sum(a['sten_obs']>=70 for _,a in N)
    f['angle_mean']=np.mean([e['angle'] for *_,e in G.edges(data=True)]) if G.number_of_edges() else 0
    if thr_feats:                       # engineered from observed values (a reader applying CAD-RADS thresholds)
        cnt=[]
        for _,a in N:
            lap=(a['hu_obs']<30) if a['hu_obs']==a['hu_obs'] else False
            pr=(a['ri_obs']>1.1) if a['ri_obs']==a['ri_obs'] else False
            sc=(0<a['spot_obs']<=3.0); c=int(lap)+int(pr)+int(sc)+a['nrs_obs']; cnt.append(c)
        f['n_hrp_ge2']=sum(c>=2 for c in cnt); f['hrp_cnt_max']=max(cnt); f['n_hrp_ge1']=sum(c>=1 for c in cnt)
    else:
        hv=[a['hu_obs'] for _,a in N if a['hu_obs']==a['hu_obs']]; rv=[a['ri_obs'] for _,a in N if a['ri_obs']==a['ri_obs']]; f['hu_min']=min(hv) if hv else 100.0; f['ri_max']=max(rv) if rv else 1.0; f['nrs_n']=sum(a['nrs_obs'] for _,a in N); f['spot_n']=sum(a['spot_obs']>0 for _,a in N)
    if level>=2:
        b=d['bm']; f.update(age=b['age'],male=b['male'],smoke=b['smoke'],diabetes=b['diabetes'],htn=b['htn'])
    if level>=3:
        for k in ('LDL','HDL','TG','HbA1c','eGFR','Lpa','hsCRP','IL6','hsTnT','NTproBNP'):
            f[k]=np.nan if d['miss'][k] else b[k]; f[k+'_miss']=d['miss'][k]
    if level>=4:
        f['CAC']=d['cac']; f['vol_calc']=sum(v['vol_calc'] for v in d['img'].values()); f['vol_nc']=sum(v['vol_nc'] for v in d['img'].values())
        f['vol_lap']=sum(v['vol_lap'] for v in d['img'].values())
        pf=[v['pfai'] for v in d['img'].values() if v['pfai']==v['pfai']]; f['pfai_max']=max(pf) if pf else np.nan
    return f

# ======================= 5x2cv statistics =======================
from scipy import stats as _st
def cv52_tests(diffs):
    """diffs: array (5,2) of paired fold differences (model A - model B) for one metric.
    Returns Dietterich (1998) t (df=5) and Alpaydin (1999) combined F (df=10,5) with p-values."""
    d=np.asarray(diffs,float); pbar=d.mean(1); s2=((d-pbar[:,None])**2).sum(1)
    den=s2.sum()
    if den<=0: return dict(t=np.nan,p_t=np.nan,F=np.nan,p_F=np.nan)
    t=d[0,0]/np.sqrt(den/5); F=(d**2).sum()/(2*den)
    return dict(t=t,p_t=2*_st.t.sf(abs(t),5),F=F,p_F=_st.f.sf(F,10,5))
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,idx in enumerate(o): run=max(run,(m-i)*p[idx]); adj[idx]=min(run,1)
    return adj
