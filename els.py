"""Reproducible exploratory analysis and validation-only model comparisons."""
from pathlib import Path
import json
import math
import time
import warnings
import hashlib
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

FEATURES = ['clicks_w1', 'clicks_w2', 'clicks_w3', 'clicks_w4',
            'active_days', 'days_since_activity', 'last_week_change', 'total_clicks']
FEATURE_SHA256 = 'c1dce1a70b51e80a3b5d443d7a103186f4501c7dd384e258e4170b7706f9bb49'
REPO_RAW = 'https://raw.githubusercontent.com/Junghwan0516/early-learning-support/main/'
COLORS = {0: '#247C96', 1: '#DE873D'}

def load_data():
    path = Path('data/features_day28.csv')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(REPO_RAW + str(path), timeout=60) as response:
            path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != FEATURE_SHA256:
        raise ValueError('Feature checksum differs from the documented release.')
    return pd.read_csv(path)

def style():
    plt.rcParams.update({'figure.dpi': 115, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})

def save_figure(fig, name):
    """Save a text SVG for GitHub and a PNG for local export."""
    folder = Path('figures'); folder.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f'{name}.svg', bbox_inches='tight')
    fig.savefig(folder / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.show()

def wilson(successes, totals, z=1.96):
    k, n = np.asarray(successes, float), np.asarray(totals, float)
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    margin = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return center-margin, center+margin

def extend_features(frame):
    result = frame.copy()
    result['early_clicks'] = result.clicks_w1 + result.clicks_w2
    result['late_clicks'] = result.clicks_w3 + result.clicks_w4
    result['activity_change'] = result.late_clicks-result.early_clicks
    result['clicks_per_active_day'] = result.total_clicks.div(result.active_days.where(result.active_days.gt(0)))
    return result

def explore_distributions(train):
    """Compare full distributions, rather than means alone."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    edges = np.linspace(0, np.log1p(train.total_clicks.max())+0.01, 30)
    for target, label in [(0, 'No recorded withdrawal'), (1, 'Later withdrawal')]:
        values = np.log1p(train.loc[train.target.eq(target), 'total_clicks'].to_numpy())
        axes[0].hist(values, bins=edges, density=True, alpha=.45, color=COLORS[target], label=label)
        ordered = np.sort(values)
        axes[1].step(ordered, np.arange(1, len(ordered)+1)/len(ordered),
                     where='post', color=COLORS[target], label=label)
    axes[0].set(title='Activity distributions overlap', ylabel='Density', xlabel='log(1 + total clicks)')
    axes[1].set(title='Empirical cumulative distributions', ylabel='Fraction of outcome group', xlabel='log(1 + total clicks)')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    save_figure(fig, '06_activity_distributions')

def explore_recency_volume(train):
    """Show coarse conditional rates with sparse-cell masking."""
    x = train.copy()
    x['volume'] = pd.qcut(x.total_clicks, 4, duplicates='drop')
    labels = ['0–1', '2–3', '4–7', '8–14', '15–27', 'No activity']
    x['recency'] = pd.cut(x.days_since_activity, [-1,1,3,7,14,27,28], labels=labels)
    counts = pd.crosstab(x.recency, x.volume, dropna=False).reindex(labels)
    rates = pd.crosstab(x.recency, x.volume, values=x.target, aggfunc='mean', dropna=False).reindex(labels)
    visible = rates.where(counts.ge(20)).to_numpy()
    cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#E9EDF0')
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.ma.masked_invalid(visible), vmin=0, vmax=1, cmap=cmap, aspect='auto')
    for i in range(len(rates)):
        for j in range(len(rates.columns)):
            n = int(counts.iloc[i,j]); r = rates.iloc[i,j]
            text = f'{r:.1%}\nn={n}' if n >= 20 else f'Masked\nn={n}'
            ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if n>=20 and r>.55 else '#222222')
    ranges = x.groupby('volume',observed=True).total_clicks.agg(['min','max'])
    ax.set_xticks(range(len(ranges)), [f'{r["min"]:.0f}–{r["max"]:.0f}' for _,r in ranges.iterrows()])
    ax.set_yticks(range(len(labels)), labels)
    ax.set(title='Recency within activity-volume quartiles', xlabel='Observed total-click range', ylabel='Days since last activity at day 27')
    cb=fig.colorbar(im,ax=ax); cb.ax.yaxis.set_major_formatter(PercentFormatter(1)); cb.set_label('Later withdrawal rate')
    save_figure(fig,'07_recency_volume')
    Path('reports').mkdir(exist_ok=True)
    rates.to_csv('reports/recency_volume_rates.csv'); counts.to_csv('reports/recency_volume_counts.csv')

def explore_early_late(train):
    """Show both activity scale and variation between learners."""
    x = extend_features(train)
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),sharex=True,sharey=True)
    limit = np.log1p(max(x.early_clicks.max(),x.late_clicks.max()))
    for ax,target in zip(axes,[0,1]):
        part=x[x.target.eq(target)]
        ax.scatter(np.log1p(part.early_clicks),np.log1p(part.late_clicks),s=12,alpha=.25,color=COLORS[target],rasterized=False)
        ax.plot([0,limit],[0,limit],color='gray',linestyle='--',label='Equal early and late activity')
        ax.set(title=f'{"No recorded withdrawal" if target==0 else "Later withdrawal"} (n={len(part):,})',xlabel='log(1 + weeks 1–2 clicks)',xlim=(-.1,limit+.1),ylim=(-.1,limit+.1))
    axes[0].set_ylabel('log(1 + weeks 3–4 clicks)'); axes[0].legend(fontsize=8)
    save_figure(fig,'08_early_late_activity')

def explore_correlations(train):
    corr=train[FEATURES].corr(method='spearman')
    fig,ax=plt.subplots(figsize=(9,7))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(FEATURES)),FEATURES,rotation=50,ha='right')
    ax.set_yticks(range(len(FEATURES)),FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(corr.iloc[i,j])>.65 else 'black')
    ax.set_title('Feature redundancy: Spearman correlation')
    fig.colorbar(im,ax=ax,label='Spearman correlation')
    save_figure(fig,'09_feature_correlations')
    corr.to_csv('reports/spearman_correlations.csv')

def explore_zero_activity(train):
    rows=[]
    for target in [0,1]:
        part=train[train.target.eq(target)]
        for i,col in enumerate(FEATURES[:4],1):
            k=int(part[col].eq(0).sum()); lo,hi=wilson([k],[len(part)])
            rows.append({'target':target,'week':i,'zero_activity':k,'learners':len(part),'rate':k/len(part),'lower':lo[0],'upper':hi[0]})
    data=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for target,label in [(0,'No recorded withdrawal'),(1,'Later withdrawal')]:
        p=data[data.target.eq(target)]
        ax.errorbar(p.week+(target-.5)*.08,p.rate,yerr=np.vstack([p.rate-p.lower,p.upper-p.rate]),fmt='o-',capsize=4,color=COLORS[target],label=label)
    ax.set(title='Weeks without recorded activity',xlabel='Week',ylabel='Fraction with zero clicks',xticks=[1,2,3,4])
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=9)
    save_figure(fig,'10_zero_activity_weeks')
    data.to_csv('reports/zero_activity_weeks.csv',index=False)

def explore_individual_profiles(train):
    rng=np.random.default_rng(42)
    fig,axes=plt.subplots(1,2,figsize=(10,6),sharey=True)
    for ax,target in zip(axes,[0,1]):
        part=train[train.target.eq(target)]
        sample=part.iloc[rng.choice(len(part),min(100,len(part)),replace=False)]
        values=sample[FEATURES[:4]].to_numpy()
        shares=np.divide(values,values.sum(axis=1,keepdims=True),out=np.zeros_like(values),where=values.sum(axis=1,keepdims=True)>0)
        order=np.argsort(shares[:,2:].sum(axis=1),kind='stable')
        im=ax.imshow(shares[order],cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        ax.set_xticks(range(4),['W1','W2','W3','W4'])
        ax.set(title='No recorded withdrawal' if target==0 else 'Later withdrawal',xlabel='Week')
    axes[0].set_ylabel('Sampled learners, ordered by late-activity share')
    fig.subplots_adjust(right=.86,wspace=.2)
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.03);cb.set_label('Share of own total clicks')
    # A dedicated layout avoids tight_layout warnings with a shared colorbar.
    fig.savefig('figures/11_individual_profiles.svg',bbox_inches='tight')
    fig.savefig('figures/11_individual_profiles.png',dpi=160,bbox_inches='tight')
    plt.show()

def explore_pattern_sensitivity(train):
    x=extend_features(train); both=x.early_clicks.gt(0)&x.late_clicks.gt(0)
    ratio=x.loc[both,'late_clicks']/x.loc[both,'early_clicks']
    rows=[]
    for threshold in [.25,.5,.75]:
        for label,mask in [('Decreased',ratio<threshold),('Similar',(ratio>=threshold)&(ratio<=1/threshold)),('Increased',ratio>1/threshold)]:
            y=x.loc[ratio.index[mask],'target']; k=int(y.sum()); n=len(y)
            lo,hi=wilson([k],[n])
            rows.append({'decrease_threshold':threshold,'pattern':label,'learners':n,'rate':y.mean(),'lower':lo[0],'upper':hi[0]})
    table=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(9,5))
    for offset,(label,color) in enumerate([('Decreased','#DE873D'),('Similar','#247C96'),('Increased','#6E6A9F')]):
        p=table[table.pattern.eq(label)]
        ax.errorbar(np.arange(3)+(offset-1)*.12,p.rate,yerr=np.vstack([p.rate-p.lower,p.upper-p.rate]),fmt='o',capsize=4,color=color,label=label)
    ax.set_xticks(range(3),['<0.25 / >4','<0.50 / >2','<0.75 / >1.33'])
    ax.set(title='Sensitivity to early–late pattern thresholds',xlabel='Late/early ratio cutoffs for decrease / increase',ylabel='Later withdrawal rate')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend()
    save_figure(fig,'12_pattern_sensitivity')
    table.to_csv('reports/pattern_sensitivity.csv',index=False)

def split_data(frame):
    from prepare_data import temporal_splits
    parts,audit=temporal_splits(frame)
    # Randomize tie order once without using outcomes; retain that order for every method.
    for name in parts:
        parts[name]=extend_features(parts[name].sample(frac=1,random_state=2026).reset_index(drop=True))
    return parts,audit

def ranking_metrics(y,scores,fraction=.1):
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=np.asarray(y);scores=np.asarray(scores,float)
    if len(y)!=len(scores) or len(y)==0 or not np.isfinite(scores).all():
        raise ValueError('Finite scores must match a nonempty outcome array.')
    if set(np.unique(y))!={0,1} or not 0<fraction<=1:
        raise ValueError('Both binary outcomes and a fraction in (0,1] are required.')
    k=math.ceil(len(y)*fraction);top=np.argsort(-scores,kind='stable')[:k];hits=int(y[top].sum())
    return {'review_budget':k,'hits':hits,'recall_at_10pct':hits/int(y.sum()),
            'precision_at_10pct':hits/k,'lift_at_10pct':(hits/k)/y.mean(),
            'average_precision':average_precision_score(y,scores),'roc_auc':roc_auc_score(y,scores)}

def model_candidates():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    result={}
    for i,c in enumerate([.1,1.,10.]):
        result[('Logistic regression',i)]=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42))
    for i,depth in enumerate([3,5,8]):
        result[('Decision tree',i)]=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=20,random_state=42)
    for i,leaf in enumerate([5,15,30]):
        result[('Random forest',i)]=RandomForestClassifier(n_estimators=200,min_samples_leaf=leaf,random_state=42,n_jobs=1)
    for i,depth in enumerate([1,2,3]):
        result[('Gradient boosting',i)]=GradientBoostingClassifier(n_estimators=100,learning_rate=.05,max_depth=depth,min_samples_leaf=15,random_state=42)
    for i,k in enumerate([15,35,75]):
        result[('KNN',i)]=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k))
    for i,c in enumerate([.1,1.,10.]):
        result[('SVM',i)]=make_pipeline(StandardScaler(),SVC(C=c,probability=False,random_state=42))
    for i,width in enumerate([(16,),(32,),(32,16)]):
        result[('MLP',i)]=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=width,max_iter=600,early_stopping=True,random_state=42))
    return result

def scores_for(model,x):
    return model.decision_function(x) if hasattr(model,'decision_function') else model.predict_proba(x)[:,1]

def run_benchmark(parts):
    """Run a predeclared three-configuration-per-family validation comparison."""
    train,val=parts['train'],parts['validation']
    candidates=model_candidates(); rows=[]; fitted={}; predictions={}
    specification={f'{family} / {i}':str(model) for (family,i),model in candidates.items()}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/benchmark_protocol.json').write_text(json.dumps({
        'selection':['validation recall_at_10pct','validation average_precision','family name','config index'],
        'seed':42,'tie_order_seed':2026,'features':FEATURES,'configurations':specification,
        'test_evaluated':False,'budget_note':'Three configurations per family, not equal compute time.'},indent=2))
    for (family,config),model in candidates.items():
        started=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(train[FEATURES],train.target)
        fit_seconds=time.perf_counter()-started
        started=time.perf_counter();scores=scores_for(model,val[FEATURES]);predict_seconds=time.perf_counter()-started
        key=f'{family} / {config}';fitted[key]=model;predictions[key]=scores
        rows.append({'family':family,'config':config,'method':key,'fit_seconds':fit_seconds,
                     'predict_seconds':predict_seconds,'warnings':' | '.join(str(w.message) for w in caught),
                     **ranking_metrics(val.target,scores)})
    # The rule may win. It is part of selection, not an ornamental reference.
    predictions['Low-activity rule']=-val.total_clicks.to_numpy()
    rows.append({'family':'Low-activity rule','config':0,'method':'Low-activity rule','fit_seconds':0.,
                 'predict_seconds':0.,'warnings':'',**ranking_metrics(val.target,predictions['Low-activity rule'])})
    all_results=pd.DataFrame(rows).sort_values(['recall_at_10pct','average_precision','family','config'],ascending=[False,False,True,True],kind='stable')
    best=all_results.drop_duplicates('family').reset_index(drop=True)
    rng=np.random.default_rng(42)
    random_results=[ranking_metrics(val.target,rng.random(len(val))) for _ in range(500)]
    random_summary=pd.DataFrame(random_results).agg(['mean','std','min','max'])
    random_summary.to_csv('reports/random_review_summary.csv')
    all_results.to_csv('reports/model_configurations.csv',index=False)
    best.to_csv('reports/validation_comparison.csv',index=False)
    # No serialized model artifacts are trusted or loaded from outside this run.
    return best,all_results,fitted,predictions,random_results

def plot_benchmark(parts,best,predictions,random_results):
    from sklearn.metrics import precision_recall_curve
    val=parts['validation']; y=val.target.to_numpy()
    fig,ax=plt.subplots(figsize=(10,5))
    order=best.sort_values('recall_at_10pct')
    ax.barh(order.family,order.recall_at_10pct,color=['#DE873D' if x=='Low-activity rule' else '#247C96' for x in order.family])
    random_recall=np.array([r['recall_at_10pct'] for r in random_results])
    lo,hi=np.quantile(random_recall,[.025,.975])
    ax.axvspan(lo,hi,color='gray',alpha=.15,label='Random review: central 95% of repetitions')
    ax.axvline(random_recall.mean(),color='gray',linestyle='--')
    ax.set(title='Validation: recall within a fixed 10% review budget',xlabel='Recall@10%')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8)
    save_figure(fig,'13_model_recall')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for row in best.itertuples():
        s=predictions[row.method]
        precision,recall,_=precision_recall_curve(y,s)
        axes[0].plot(recall,precision,label=row.family,alpha=.8)
        fractions=np.arange(.05,.55,.05)
        result=[ranking_metrics(y,s,f)['recall_at_10pct'] for f in fractions]
        axes[1].plot(fractions,result,marker='.',label=row.family,alpha=.8)
    axes[0].axhline(y.mean(),color='gray',linestyle='--',label='Prevalence')
    axes[0].set(title='Validation precision–recall curves',xlabel='Recall',ylabel='Precision')
    axes[1].plot([0,.5],[0,.5],color='gray',linestyle='--',label='Random expectation')
    axes[1].set(title='Recall at different review capacities',xlabel='Fraction reviewed',ylabel='Recall')
    for ax in axes:
        ax.legend(fontsize=7);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    save_figure(fig,'14_ranking_curves')
    fig,ax=plt.subplots(figsize=(9,5))
    for row in best.itertuples():
        if row.fit_seconds>0:
            ax.scatter(row.fit_seconds,row.recall_at_10pct,s=70,color='#247C96')
            ax.annotate(row.family,(row.fit_seconds,row.recall_at_10pct),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set_xscale('log');ax.set(title='Validation performance and training cost',xlabel='Fit time in seconds (log scale; this environment)',ylabel='Recall@10%')
    ax.yaxis.set_major_formatter(PercentFormatter(1));save_figure(fig,'15_training_cost')

def feature_ablation(parts):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    groups={
        'Volume only':['total_clicks'],
        'Volume + consistency + recency':['total_clicks','active_days','days_since_activity'],
        'Weekly representation':FEATURES,
        'Weekly + alternative sums':FEATURES+['early_clicks','late_clicks','activity_change']}
    rows=[]
    for family in ['Logistic regression','Random forest']:
        for label,columns in groups.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42)) if family=='Logistic regression' else RandomForestClassifier(n_estimators=200,min_samples_leaf=15,random_state=42,n_jobs=1)
            model.fit(parts['train'][columns],parts['train'].target)
            scores=scores_for(model,parts['validation'][columns])
            rows.append({'family':family,'feature_set':label,'n_features':len(columns),**ranking_metrics(parts['validation'].target,scores)})
    table=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(11,5))
    for i,(family,color) in enumerate([('Logistic regression','#247C96'),('Random forest','#DE873D')]):
        p=table[table.family.eq(family)]
        ax.bar(np.arange(4)+(i-.5)*.35,p.recall_at_10pct,width=.35,label=family,color=color)
    ax.set_xticks(range(4),['Volume','+ Consistency / recency','Weekly','Weekly + sums'],rotation=12)
    ax.set(title='Validation feature ablation: fixed configurations',ylabel='Recall@10%')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend()
    save_figure(fig,'16_feature_ablation')
    table.to_csv('reports/feature_ablation.csv',index=False)
    return table

def validation_diagnostics(parts,best,predictions):
    val=parts['validation'];winner=best.iloc[0].method
    y=val.target.to_numpy(); s=predictions[winner];k=math.ceil(.1*len(val))
    selected=np.zeros(len(val),bool);selected[np.argsort(-s,kind='stable')[:k]]=True
    labels=np.select([selected & (y==1),selected & (y==0),~selected & (y==1)],['Captured withdrawal','Reviewed non-withdrawal','Missed withdrawal'],default='Unreviewed non-withdrawal')
    frame=val.assign(review_outcome=labels)
    summary=frame.groupby('review_outcome')[['total_clicks','active_days','days_since_activity']].median()
    summary['learners']=frame.groupby('review_outcome').size()
    summary.to_csv('reports/validation_error_groups.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    groups=['Captured withdrawal','Missed withdrawal','Reviewed non-withdrawal']
    for ax,feature in zip(axes,['total_clicks','active_days','days_since_activity']):
        values=[frame.loc[frame.review_outcome.eq(g),feature].to_numpy() for g in groups]
        if feature=='total_clicks': values=[np.log1p(v) for v in values]
        ax.boxplot(values,tick_labels=['Captured','Missed','Reviewed\nnon-withdrawal'],showfliers=False)
        ax.set(title=feature,ylabel='log(1 + clicks)' if feature=='total_clicks' else feature)
    fig.suptitle(f'Validation errors: {winner}');save_figure(fig,'17_error_profiles')
    # Paired resampling fixes the models but recomputes the review budget and ranking.
    baseline=predictions['Low-activity rule'];rng=np.random.default_rng(2026);differences=[]
    for _ in range(1000):
        index=rng.integers(0,len(y),len(y));yy=y[index]
        if yy.sum()==0:continue
        budget=math.ceil(.1*len(yy))
        def recall(scores):return yy[np.argsort(-scores[index],kind='stable')[:budget]].sum()/yy.sum()
        differences.append(recall(s)-recall(baseline))
    lo,hi=np.quantile(differences,[.025,.975]);estimate=ranking_metrics(y,s)['recall_at_10pct']-ranking_metrics(y,baseline)['recall_at_10pct']
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.hist(differences,bins=30,color='#247C96',alpha=.8)
    ax.axvline(0,color='gray',linestyle='--');ax.axvline(estimate,color='#DE873D',label='Observed validation difference')
    ax.set(title='Paired validation bootstrap: selected method minus low-activity rule',xlabel='Difference in Recall@10%',ylabel='Bootstrap repetitions')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8)
    save_figure(fig,'18_bootstrap_difference')
    result={'selected_method':winner,'difference':float(estimate),'lower':float(lo),'upper':float(hi),
            'note':'Exploratory conditional-on-fitted-model interval; model selection is not accounted for. Not final-test evidence.'}
    Path('reports/validation_bootstrap.json').write_text(json.dumps(result,indent=2))
    return summary,result
