"""Side lines of a round, report only, never gating, read by no gate, card or fit of the run (coordinator 11-07 after the fusion trial).
Beside ds_agree: the whole-row scale form ds_agree_scale, s_i = 1/(1+(|B_i|-1) rho) and beta_m,c on the A1 base, shrunk to beta_m. FIT held out by picture, and SELECT.
The functions are fus_trial/fus_lib.py verbatim; only Cx reads the driver's arrays in place of the trial's frozen cache, and the trial's block grouping (opts part) is gone.
fus_side.py IN.npz OUT.json"""
import json, os, sys, time, zlib
C_ = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [C_, os.path.join(os.path.dirname(C_), "pb"), os.path.dirname(C_)]
import numpy as np
from scipy.optimize import minimize
import res_fuse as rf
from vrsi27.boost.calibrate import log_softmax_masked


class Cx:
    """One Table from the driver: V [M,N,2], rows, classes."""
    def __init__(s, z):
        s.V, s.y, s.mask, s.other = z["V"], z["y"].copy(), z["mask"], z["other"]
        s.names, s.ids, s.bench, s.cluster = [str(x) for x in z["names"]], [str(x) for x in z["ids"]], z["bench"].astype(str), z["cluster"].astype(str)
        s.mc, s.cells, s.fit, s.select = z["mc"], z["cells"], z["fit"], z["select"]
        s.K, s.N, s.L, s.nc = s.mask.shape[1], len(s.ids), s.mask.sum(1), int(s.cells.max()) + 1
        s.fold = np.array([zlib.crc32(c.encode()) % 5 for c in s.cluster])
        s.ifold = np.array([zlib.crc32(("in|" + c).encode()) % 4 for c in s.cluster])
        s.tie = rf.tie_break(s.ids, s.K)


def tables(V,y,mc,cells,K,nc,tr,kp=10.):
    """res_fuse.ds_letters equations for clamped rows tr."""
    M=V.shape[0]; N_=np.zeros((M,nc,K,K)); R_=np.zeros((M,nc,2)); pi=np.ones(K); tr=np.asarray(tr,int)
    tm=tr[mc[tr]]; to=tr[~mc[tr]]; np.add.at(pi,y[tm],1)
    for m in range(M):
        for t in (0,1):
            v=V[m,:,t]; r=tm[v[tm]>=0]; np.add.at(N_[m],(cells[r],y[r],v[r]),1)
            r=to[v[to]>=0]; np.add.at(R_[m],(cells[r],(v[r]!=y[r]).astype(int)),1)
    NG,RG=N_.sum(1),R_.sum(1); pri=np.ones((K,K))+np.eye(K); br=np.array([2.,1.])
    A=N_+kp*((NG+pri)/(NG+pri).sum(2,keepdims=True))[:,None]; LE=np.log(A/A.sum(3,keepdims=True))
    B=R_+kp*((RG+br)/(RG+br).sum(1,keepdims=True))[:,None]; LR=np.log(B/B.sum(2,keepdims=True))
    return LE,LR,np.log(pi/pi.sum())

def evidence(V,tb,cx):
    """lp [N,K], E [M,N,K]: the two samples of a member summed; abstention adds nothing."""
    LE,LR,lpi=tb; M=V.shape[0]; E=np.zeros((M,cx.N,cx.K)); mc,cells,mask=cx.mc,cx.cells,cx.mask
    for m in range(M):
        for t in (0,1):
            v=V[m,:,t]; ok=v>=0
            r=np.flatnonzero(ok&mc); E[m,r]+=LE[m,cells[r],:,v[r]]
            r=np.flatnonzero(ok&~mc)
            base=LR[m,cells[r],1]-np.log(np.maximum(cx.L[r]-1,1)); E[m,r]+=base[:,None]*mask[r]; E[m,r,v[r]]+=LR[m,cells[r],0]-base
    lp=np.where(mc[:,None],lpi[None,:],0.)*mask
    return lp,E*mask[None]

def llof(lp,E,b):
    return lp+(b[:,:,None]*E).sum(0)

def pair_corr(x,valid,strata,rows):
    """Within-stratum pooled correlation of x over rows both members answered. Returns rho, cov, n (residual df). Strata with one row add nothing."""
    M=x.shape[0]; rho=np.full((M,M),np.nan); n=np.zeros((M,M)); cov=np.zeros((M,M))
    ns=int(strata.max())+1 if len(strata) else 1
    for a in range(M):
        for b in range(a,M):
            ok=rows[valid[a,rows]&valid[b,rows]]
            if len(ok)<3: continue
            st=strata[ok]; c=np.bincount(st,minlength=ns).astype(float); c_=np.maximum(c,1)
            xa=x[a,ok]; xb=x[b,ok]; ma=np.bincount(st,xa,ns)/c_; mb=np.bincount(st,xb,ns)/c_
            da=xa-ma[st]; db=xb-mb[st]; df=len(ok)-(c>0).sum()
            if df<2: continue
            va,vb=(da*da).sum(),(db*db).sum(); n[a,b]=n[b,a]=df; cov[a,b]=cov[b,a]=(da*db).sum()/df
            if va>1e-12 and vb>1e-12: rho[a,b]=rho[b,a]=(da*db).sum()/np.sqrt(va*vb)
    return rho,cov,n

def shrink_pair(rho,n,k0=20.):
    M=len(rho); iu=np.triu_indices(M,1); ok=~np.isnan(rho[iu])
    g=float(np.average(rho[iu][ok],weights=n[iu][ok])) if ok.any() and n[iu][ok].sum()>0 else 0.
    r=np.where(np.isnan(rho),0.,rho); s=(n*r+k0*g)/(n+k0); np.fill_diagonal(s,1.)
    return s,g

def err_flags(V,y):
    """Share of a member's valid samples that miss gold, and which members answered."""
    ok=V>=0; hit=(V==y[None,:,None])&ok; nv=ok.sum(2)
    return 1-hit.sum(2)/np.maximum(nv,1),nv>0

def rho_err(cx,V,rows,cond,kappa):
    e,valid=err_flags(V,cx.y)
    st=cx.cells*cx.K+cx.y if cond else cx.cells
    _,st=np.unique(np.where(np.isin(np.arange(cx.N),rows),st,0),return_inverse=True)
    if cond: rp,_,np_=pair_corr(e,valid,st,rows)
    else:    rp,_,np_=pair_corr(e,valid,np.zeros(cx.N,int),rows)
    sp,g=shrink_pair(rp,np_)
    M=V.shape[0]; R=np.repeat(sp[None],cx.nc,0)
    if np.isfinite(kappa):
        for c in np.unique(cx.cells[rows]):
            rc=rows[cx.cells[rows]==c]; rr,_,nn=pair_corr(e,valid,st,rc)
            rr=np.where(np.isnan(rr),sp,rr); R[c]=(nn*rr+kappa*sp)/(nn+kappa); np.fill_diagonal(R[c],1.)
    return R,sp,g

def b_one(cx,V,tr_rows,Ein,lpin,dtr,T0,opts):
    return (lambda valid: np.ones(valid.shape)),{}

def b_equi(cx,V,tr_rows,Ein,lpin,dtr,T0,opts):
    """One number: rho bar. beta_i = 1/(1+(|B_i|-1) rho). Rescales a whole row: changes the answer only through the prior term."""
    rows=tr_rows if opts['rows']=='all' else dtr
    _,_,g=rho_err(cx,V,rows,opts.get('cond',False),np.inf)
    if 'rho' in opts: g=opts['rho']
    g=max(g,0.)
    return (lambda valid: np.where(valid,1/(1+(valid.sum(0,keepdims=True)-1)*g),0.)),dict(rho_bar=g)

def score_rows(ll,y,mask,T): 
    return -log_softmax_masked(ll/T,mask)[np.arange(len(y)),y]

def choose_lambda(cx,V,Ein,lpin,dtr,grid,make):
    """make(rows,lam) -> beta function fit on rows. Held-out CE over inner folds; T refit on the other inner folds."""
    best,scores=None,[]
    valid=(V>=0).any(2)
    for lam in grid:
        tot=0.
        for g in range(4):
            te=dtr[cx.ifold[dtr]==g]; tr=dtr[cx.ifold[dtr]!=g]
            if not len(te) or not len(tr): continue
            bf=make(tr,lam); b=bf(valid); ll=llof(lpin,Ein,b)
            T=rf.fit_temperature(ll[tr],cx.y[tr],cx.mask[tr])
            tot+=score_rows(ll[te]+cx.tie[te],cx.y[te],cx.mask[te],T).sum()
        scores.append(tot/len(dtr))
    return grid[int(np.argmin(scores))],scores

def fit_member_class(lp,E,y,mask,tie,T0,lam,lamc,rows,valid,cells,nc):
    """beta_m,c = beta_m + d_m,c, shrunk to beta_m. Bounds [0,2] on the sum are enforced by clipping at use; penalty lam on beta_m-1, lamc on d."""
    M=E.shape[0]; R=np.arange(len(rows)); y_=y[rows]; m_=mask[rows]; c_=cells[rows]
    E_=E[:,rows]; ml=lp[rows]+tie[rows]; EE=np.where(m_[None],E_,0.)
    def f(z):
        bm=z[:M]; d=z[M:].reshape(M,nc); bb=bm[:,None]+d[:,c_]
        F=(ml+(bb[:,:,None]*E_).sum(0))/T0
        ls=log_softmax_masked(F,m_); p=np.where(m_,np.exp(np.where(m_,ls,0.)),0.); res=p.copy(); res[R,y_]-=1
        gr=(res[None]*EE).sum(2)/(T0*len(rows))   # [M,n]
        gm=gr.sum(1)+lam*(bm-1); gd=np.zeros((M,nc))
        for m in range(M): np.add.at(gd[m],c_,gr[m])
        gd+=lamc*d
        return -ls[R,y_].mean()+lam/2*((bm-1)**2).sum()+lamc/2*(d**2).sum(),np.r_[gm,gd.ravel()]
    z0=np.r_[np.ones(M),np.zeros(M*nc)]
    r=minimize(f,z0,jac=True,method='L-BFGS-B',bounds=[(0,2)]*M+[(-1,1)]*(M*nc),options=dict(maxiter=300,ftol=1e-9))
    return r.x[:M],r.x[M:].reshape(M,nc)

def level_ids(cx,level):
    if level=='bench': return np.unique(cx.bench,return_inverse=True)[1]
    if level=='kindbench': return np.unique(np.char.add(np.char.add(cx.bench,'|'),cx.kind),return_inverse=True)[1]
    return cx.cells

def ng_range(n): return range(n)

def eff_df(lp,XG,bm,d,cx,rows,T,lam,ids,nl):
    """Effective parameters at the solution: tr[H (H+lam)^-1] of the penalised Hessian, row-mean scale."""
    G=XG.shape[0]; p=G+G*nl; H=np.zeros((p,p)); y=cx.y[rows]; m_=cx.mask[rows]; c_=ids[rows]
    bb=np.clip(bm[:,None]+d[:,c_],0,2); F=(lp[rows]+cx.tie[rows]+(bb[:,:,None]*XG[:,rows]).sum(0))/T
    ls=log_softmax_masked(F,m_); pr=np.where(m_,np.exp(np.where(m_,ls,0.)),0.)
    for i in range(len(rows)):
        x=np.where(m_[i][None],XG[:,rows[i]],0.)/T; mu=x@pr[i]; C=(x*pr[i][None])@x.T-np.outer(mu,mu)
        idx=np.r_[np.arange(G),G+np.arange(G)*nl+c_[i]]; H[np.ix_(idx,idx)]+=np.block([[C,C],[C,C]])/len(rows)
    pen=np.r_[np.full(G,lam[0]),np.full(G*nl,lam[1])]
    return float(np.trace(np.linalg.solve(H+np.diag(pen),H)))

def b_exp_level(cx,V,tr_rows,Ein,lpin,dtr,T0,opts):
    """beta_g,level = beta_g + d_g,level. g = member, or a block when opts['part'] is set. opts['base'] multiplies a row form first, with its own T."""
    valid=(V>=0).any(2); ids=level_ids(cx,opts.get('level','cell')); nl=int(ids.max())+1
    base=opts.get('base'); bbf=(base[0](cx,V,tr_rows,Ein,lpin,dtr,T0,base[1])[0] if base else (lambda v: v.astype(float)))
    bb=bbf(valid); X=Ein*bb[:,:,None]
    T1=rf.fit_temperature((lpin+X.sum(0))[dtr],cx.y[dtr],cx.mask[dtr])
    if opts.get('part'):
        gr=np.unique([group_of(n,opts['part']) for n in opts['names']],return_inverse=True)[1]
    else:
        gr=np.arange(V.shape[0])
    ng=gr.max()+1; XG=np.stack([X[gr==a].sum(0) for a in ng_range(ng)])
    lamg,lcg=opts.get('lam_grid',[.01,.03,.1]),opts.get('lamc_grid',[.001,.003,.01,.03,.1])
    grid=[(a,b) for a in lamg for b in lcg]
    mk=lambda bm,d: (lambda v: np.clip(bm[gr][:,None]+d[gr][:,ids],0,2)*bbf(v))
    make=lambda rows,l: mk(*fit_member_class(lpin,XG,cx.y,cx.mask,cx.tie,T1,l[0],l[1],rows,valid,ids,nl))
    lam,sc=choose_lambda(cx,V,Ein,lpin,dtr,grid,make)
    bm,d=fit_member_class(lpin,XG,cx.y,cx.mask,cx.tie,T1,lam[0],lam[1],dtr,valid,ids,nl)
    return mk(bm,d),dict(lam=lam,cv=sc,beta=bm,d_rms=float(np.sqrt((d**2).mean())),df=eff_df(lpin,XG,bm,d,cx,dtr,T1,lam,ids,nl))

def fit_apply(cx,P,train,fn,opts,one_voter_dis=False,V_ac=None):
    """Tables on FIT disagreement rows of train. Held-out evidence by inner folds by picture, on those rows. Beta from it, then T on it, with beta fixed. Then tables as fit on all of train, applied."""
    V,ac,dis=view(cx,P)
    if one_voter_dis:
        nv=(V>=0).any(2).sum(0); one=(nv==1)&(ac>=0); ac=np.where(one,-1,ac); dis=dis|one
    y=cx.y; mask=cx.mask; other=cx.other; pas=rf.passes(ac,other)
    train=np.asarray(train,int); tr=train[dis[train]]
    tb=tables(V,y,cx.mc,cx.cells,cx.K,cx.nc,tr)
    lp,E=evidence(V,tb,cx)
    pr=train[pas[train]]; acc=rf.agree_accuracy(ac[pr]==y[pr],cx.cells[pr],cx.nc)
    dtr=train[~pas[train]]
    Ein=np.zeros_like(E); lpin=np.zeros_like(lp)
    for g in range(4):
        te=dtr[cx.ifold[dtr]==g]
        if not len(te): continue
        tb_g=tables(V,y,cx.mc,cx.cells,cx.K,cx.nc,tr[cx.ifold[tr]!=g]); lpg,Eg=evidence(V,tb_g,cx)
        Ein[:,te]=Eg[:,te]; lpin[te]=lpg[te]
    valid=(V>=0).any(2)
    T0=rf.fit_temperature(llof(lpin,Ein,np.ones(valid.shape))[dtr],y[dtr],mask[dtr])
    bf,info=fn(cx,V,train,Ein,lpin,dtr,T0,opts)
    b=bf(valid); T=rf.fit_temperature(llof(lpin,Ein,b)[dtr],y[dtr],mask[dtr])
    ll=llof(lp,E,b)
    F,_=rf.posterior(ll,cx.tie,mask,ac,other,cx.cells,acc,T)
    info.update(T=T,T0=T0,beta_mean_row=float(b[:,dtr].sum(0).mean()))
    return F,dict(info,b=b,ac=ac,dis=dis,lp=lp,E=E,tb=tb,acc=acc)

def oof(cx,P,fn,opts,one_voter_dis=False,train_rows=None):
    """Outer folds as res_fuse: crc32(cluster)%5 over FIT. Returns F of the FIT rows, held out by picture, and per-fold info."""
    fit=cx.fit if train_rows is None else train_rows
    Fo=np.full((cx.N,cx.K),-np.inf); infos=[]
    for f in range(5):
        te=fit[cx.fold[fit]==f]
        if not len(te): continue
        F,info=fit_apply(cx,P,fit[cx.fold[fit]!=f],fn,opts,one_voter_dis)
        Fo[te]=F[te]; info={k:v for k,v in info.items() if k not in ('lp','E','tb','acc')}; info['b']=None; infos.append(info)
    return Fo,infos

def rho_bar(cx,V,rows):
    """Within class+true column correlation of member error share, on FIT disagreement rows, pooled over pairs by residual df. Pair values shrunk toward the mean (kappa 20); the mean is what A1e uses."""
    return max(rho_err(cx,V,rows,True,np.inf)[2],0.)

def fuse_scaled(cx,P,use_rho=True,y=None):
    """Driver form. F_ij = s_i (log pi_j + sum_m sum_s log A + tie) / T, s_i = 1/(1+(|B_i|-1) rho). use_rho False: A0 itself. Answers equal A0 by construction. Gold read on FIT rows only."""
    V,ac,dis=view(cx,P); y=cx.y if y is None else y; mask,other,fit=cx.mask,cx.other,cx.fit; pas=rf.passes(ac,other); valid=(V>=0).any(2); nv=valid.sum(0)
    def one(train):
        tr=train[dis[train]]; lp,E=evidence(V,tables(V,y,cx.mc,cx.cells,cx.K,cx.nc,tr),cx); pr=train[pas[train]]
        r=rho_bar(cx,V,tr) if use_rho else 0.
        return lp+E.sum(0),1/(1+(nv-1)*r),rf.agree_accuracy(ac[pr]==y[pr],cx.cells[pr],cx.nc),r
    parts,held=[],[]
    for f in range(5):
        te=fit[cx.fold[fit]==f]; ll,s,acc,r=one(fit[cx.fold[fit]!=f]); parts.append((te,ll,s,acc)); d_=te[~pas[te]]; held.append((s[d_,None]*ll[d_],y[d_],mask[d_]))
    T=rf.fit_temperature(np.concatenate([h[0] for h in held]),np.concatenate([h[1] for h in held]),np.concatenate([h[2] for h in held]))
    Fo=np.full((cx.N,cx.K),-np.inf)
    for te,ll,s,acc in parts:
        Fo[te]=rf.posterior(s[:,None]*ll,s[:,None]*cx.tie,mask,ac,other,cx.cells,acc,T)[0][te]
    ll,s,acc,r=one(fit)
    F,_=rf.posterior(s[:,None]*ll,s[:,None]*cx.tie,mask,ac,other,cx.cells,acc,T)
    return F,Fo,T,r

def answers(cx,F):
    G=np.where(cx.mask,F,-np.inf).copy(); r=np.arange(cx.N); ot=cx.other>=0; G[r[ot],cx.other[ot]]=-np.inf
    return G.argmax(1)

def row_ce(cx,F,rows):
    from vrsi27.boost.edge import row_ce as rc
    return rc(F[rows],cx.y[rows],cx.mask[rows])

def cluster_boot(d,cl,B=4000,seed=0):
    """mean of d, 95% CI, q=.2 lower bound (the driver's BOOT_Q), pictures resampled."""
    u,inv=np.unique(cl,return_inverse=True); s=np.bincount(inv,d,len(u)); n=np.bincount(inv,minlength=len(u)).astype(float)
    rng=np.random.default_rng(seed); W=rng.multinomial(len(u),np.ones(len(u))/len(u),size=B).astype(float)
    m=(W@s)/(W@n)
    return float(d.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975)),float(np.quantile(m,.2))

def compare(cx,Fa,Fb,rows,B=4000):
    """b against a on rows: CE drop, accuracy gain in rows, CIs by picture."""
    ca,cb=row_ce(cx,Fa,rows),row_ce(cx,Fb,rows); aa=answers(cx,Fa)[rows]==cx.y[rows]; ab=answers(cx,Fb)[rows]==cx.y[rows]
    dce=cluster_boot(ca-cb,cx.cluster[rows],B); dac=cluster_boot(ab.astype(float)-aa,cx.cluster[rows],B)
    return dict(n=int(len(rows)),ce_a=float(ca.mean()),ce_b=float(cb.mean()),dce=dce[0],dce_ci=dce[1:3],dce_lb20=dce[3],acc_a=float(aa.mean()),acc_b=float(ab.mean()),dacc=dac[0],dacc_ci=dac[1:3],up=int((ab&~aa).sum()),down=int((aa&~ab).sum()),changed=int((answers(cx,Fa)[rows]!=answers(cx,Fb)[rows]).sum()))

def view(cx,P):
    """Agreement of the member subset P."""
    V=cx.V[list(P)]
    ac,dis,tot=rf.agreement(V,cx.K)
    return V,ac,dis


def run(inp, out):
    t0 = time.time()
    cx = Cx(np.load(inp, allow_pickle=False))
    P = list(range(cx.V.shape[0]))
    fit, sel = cx.fit, cx.select
    R = dict(rows_fit=len(fit), rows_select=len(sel), members=cx.names, reads="report only, nothing of the run reads this")
    F0, Fo0, T0, _ = fuse_scaled(cx, P, use_rho=False)
    F1, Fo1, T1, rho = fuse_scaled(cx, P)
    R["ds_agree"] = dict(T=T0)
    R["ds_agree_scale"] = dict(rho=rho, T=T1, held_fit=compare(cx, Fo0, Fo1, fit), select=compare(cx, F0, F1, sel))
    R["secs_scale"] = time.time() - t0
    t1 = time.time()
    equi = (b_equi, dict(rows="dis", cond=True))
    opts = dict(level="cell", base=equi)
    Fn, _ = oof(cx, P, b_one, {})
    Fb, infos = oof(cx, P, b_exp_level, opts)
    Fa, _ = fit_apply(cx, P, fit, b_one, {})
    Fc, info = fit_apply(cx, P, fit, b_exp_level, opts)
    R["beta_mc"] = dict(beta={n: float(b) for n, b in zip(cx.names, info["beta"])}, lam=list(info["lam"]), d_rms=info["d_rms"], df=info["df"], rho=info.get("rho_bar"), T=info["T"], beta_folds=[[float(b) for b in i["beta"]] for i in infos], held_fit=compare(cx, Fn, Fb, fit), select=compare(cx, Fa, Fc, sel))
    R["secs_beta"] = time.time() - t1
    json.dump(R, open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
