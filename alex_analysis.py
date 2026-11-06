"""MS&E 226 Person B analysis. Run ONLY on Emily's development CSV.
Example: python alex_analysis.py --development development.csv --outcome 'Overall Rating' --task classification
Do not pass the original full CSV or the holdout CSV.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

ID_WORDS = ('id', 'ccn', 'name', 'address', 'phone', 'telephone', 'zip', 'latitude', 'longitude', 'location', 'date', 'footnote', 'processing')
# Do not include direct rating components, scoring inputs, or downstream information when predicting Overall Rating.
RATING_LEAKAGE = ('rating', 'health deficiency', 'health score', 'survey score', 'revisit', 'citations', 'fines', 'penalties', 'payment denials', 'special focus', 'abuse icon', 'high performing icon')

def main():
 p=argparse.ArgumentParser();p.add_argument('--development',required=True);p.add_argument('--outcome',required=True);p.add_argument('--task',choices=['regression','classification'],required=True);p.add_argument('--output',default='alex_results');p.add_argument('--seed',type=int,default=42);args=p.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(args.development,low_memory=False)
 if args.outcome not in df: raise ValueError('Outcome not found in development data')
 y=pd.to_numeric(df[args.outcome],errors='coerce') if args.task=='regression' else df[args.outcome]
 valid=y.notna();df=df.loc[valid].copy();y=y.loc[valid]
 if args.task=='classification' and y.nunique()!=2: raise ValueError('Classification requires a BINARY outcome; define the binary outcome in the development dataset first.')
 excluded={args.outcome};
 for c in df.columns:
  lo=c.lower()
  if any(k in lo for k in ID_WORDS): excluded.add(c)
  if args.outcome.lower()=='overall rating' and any(k in lo for k in RATING_LEAKAGE): excluded.add(c)
 features=[c for c in df.columns if c not in excluded and df[c].notna().mean()>=0.65 and df[c].nunique(dropna=True)>1]
 # Limit extremely high-cardinality categorical columns.
 features=[c for c in features if pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True)<=25]
 numeric=[c for c in features if pd.api.types.is_numeric_dtype(df[c])]
 cats=[c for c in features if c not in numeric]
 missing=df[features].isna().mean().sort_values(ascending=False)
 missing.to_csv(out/'missingness.csv',header=['missing_fraction'])
 report={'development_rows':len(df),'outcome':args.outcome,'task':args.task,'numeric_features':numeric,'categorical_features':cats,'excluded_columns':sorted(excluded),'missingness_top_15':missing.head(15).to_dict()}
 if args.task=='regression':
  corrs=df[numeric].corrwith(pd.to_numeric(y),method='spearman').dropna().sort_values(key=abs,ascending=False)
  corrs.to_csv(out/'outcome_correlations.csv',header=['spearman_rho'])
  top=corrs.head(12).index.tolist()
  if top:
   fig,ax=plt.subplots(figsize=(9,5));corrs.loc[top].sort_values().plot.barh(ax=ax);ax.set_xlabel('Spearman correlation with outcome');fig.tight_layout();fig.savefig(out/'outcome_correlations.png',dpi=160);plt.close(fig)
 else:
  rates=[]
  for c in numeric:
   temp=pd.DataFrame({'x':pd.to_numeric(df[c],errors='coerce'),'y':y})
   means=temp.groupby('y')['x'].mean()
   if len(means)==2: rates.append((c,float(means.iloc[1]-means.iloc[0])))
  pd.DataFrame(rates,columns=['feature','mean_difference_class1_minus_class0']).to_csv(out/'numeric_outcome_associations.csv',index=False)
 if len(numeric)>=2:
  corr=df[numeric].corr(method='spearman');corr.to_csv(out/'predictor_correlations.csv')
  pairs=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)).stack().sort_values(key=abs,ascending=False)
  pairs.head(30).to_csv(out/'strongest_predictor_pairs.csv',header=['spearman_rho'])
 for c in cats[:12]:
  tab=df.groupby(c,dropna=False)[args.outcome].agg(['count','mean']) if args.task=='regression' else pd.crosstab(df[c],y,normalize='index')
  tab.to_csv(out/('subgroup_'+re.sub('[^a-zA-Z0-9]+','_',c)[:55]+'.csv'))
 # Two intentionally different feature sets: operational staffing/size vs broader structural profile.
 operations=[c for c in features if any(s in c.lower() for s in ('staffing hours','turnover','certified beds','residents per day','case-mix'))]
 broad=features
 if len(operations)<2: operations=numeric[:min(8,len(numeric))]
 if len(operations)<2: raise ValueError('Too few eligible predictors; inspect outcome and columns manually')
 sets={'A_operational':operations,'B_broad':broad}
 report['feature_sets']=sets
 cv=(StratifiedKFold(n_splits=5,shuffle=True,random_state=args.seed) if args.task=='classification' else KFold(n_splits=5,shuffle=True,random_state=args.seed))
 rows=[]
 for name,cols in sets.items():
  nums=[c for c in cols if c in numeric];categoricals=[c for c in cols if c in cats]
  transformers=[]
  if nums:transformers.append(('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
  if categoricals:transformers.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categoricals))
  prep=ColumnTransformer(transformers)
  model=LogisticRegression(max_iter=3000) if args.task=='classification' else LinearRegression()
  pipe=Pipeline([('prep',prep),('model',model)])
  scores=cross_validate(pipe,df[cols],y,cv=cv,scoring=('accuracy','roc_auc') if args.task=='classification' else ('neg_root_mean_squared_error','r2'),error_score='raise')
  row={'feature_set':name,'n_features':len(cols)}
  for k,v in scores.items():
   if k.startswith('test_'):row[k+'_mean']=float(np.mean(v));row[k+'_sd']=float(np.std(v))
  rows.append(row)
 pd.DataFrame(rows).to_csv(out/'linear_model_cv_results.csv',index=False)
 (out/'analysis_metadata.json').write_text(json.dumps(report,indent=2,default=str))
 print('Saved results to',out)
if __name__=='__main__':main()
