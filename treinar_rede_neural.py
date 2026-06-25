#!/usr/bin/env python3
# Treino da REDE NEURAL (Deep Learning) de toxicidade: TF-IDF -> SVD -> MLP.
# Gera metricas, curva de loss e matriz de confusao. Requer ToLD-Br/ToLD-BR.csv.
import numpy as np, pandas as pd, json, time, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    f1_score, roc_auc_score, confusion_matrix)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time(); SEED=42
CATS=["homophobia","obscene","insult","racism","misogyny","xenophobia"]
FIG="/sessions/beautiful-vigilant-ritchie/mnt/outputs/figuras"
df=pd.read_csv("/tmp/ToLD-Br/ToLD-BR.csv").drop_duplicates(subset=["text"]).reset_index(drop=True)
df["label"]=(df[CATS].max(axis=1)>=2).astype(int)
X=df["text"].astype(str).values; y=df["label"].values
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,stratify=y,random_state=SEED)
Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=0.50,stratify=ytmp,random_state=SEED)
vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=40000,sublinear_tf=True)
Atr=vec.fit_transform(Xtr); Ava=vec.transform(Xva); Ate=vec.transform(Xte)
svd=TruncatedSVD(n_components=300,random_state=SEED)
Str=svd.fit_transform(Atr); Sva=svd.transform(Ava); Ste=svd.transform(Ate)
sc=StandardScaler(); Str=sc.fit_transform(Str); Sva=sc.transform(Sva); Ste=sc.transform(Ste)
print("SVD var explicada:",round(svd.explained_variance_ratio_.sum(),3),"prep:",round(time.time()-t0,1),"s")
clf=MLPClassifier(hidden_layer_sizes=(256,64),activation="relu",solver="adam",alpha=3e-4,
    batch_size=128,learning_rate_init=1e-3,max_iter=120,early_stopping=True,
    n_iter_no_change=10,validation_fraction=0.12,random_state=SEED)
clf.fit(Str,ytr)
pv=clf.predict_proba(Sva)[:,1]; bt,bm=0.5,-1
for t in np.arange(0.10,0.90,0.02):
    m=f1_score(yva,(pv>=t).astype(int),average="macro")
    if m>bm: bm,bt=m,float(t)
prob=clf.predict_proba(Ste)[:,1]; pred=(prob>=bt).astype(int)
acc=accuracy_score(yte,pred); p,r,f,_=precision_recall_fscore_support(yte,pred,labels=[0,1],zero_division=0)
macro=f1_score(yte,pred,average="macro"); auc=roc_auc_score(yte,prob); cm=confusion_matrix(yte,pred)
res={"modelo":"TF-IDF + SVD(300) + MLP 256-64 ReLU Adam","epocas":int(clf.n_iter_),"limiar":round(bt,3),
 "n_total":int(len(y)),"treino":int(len(ytr)),"val":int(len(yva)),"teste":int(len(yte)),
 "accuracy":round(acc,4),"macro_f1":round(macro,4),"roc_auc":round(auc,4),
 "prec_nao":round(p[0],4),"rec_nao":round(r[0],4),"f1_nao":round(f[0],4),
 "prec_tox":round(p[1],4),"rec_tox":round(r[1],4),"f1_tox":round(f[1],4),"cm":cm.tolist(),
 "losses":[round(x,4) for x in clf.loss_curve_]}
print("RESULT macroF1=%.4f f1tox=%.4f auc=%.4f acc=%.4f ep=%d limiar=%.2f t=%.1fs"%(macro,f[1],auc,acc,clf.n_iter_,bt,time.time()-t0))
json.dump(res,open(FIG+"/resultados_mlp.json","w"),ensure_ascii=False,indent=2)
joblib.dump({"vectorizer":vec,"svd":svd,"scaler":sc,"model":clf,"limiar":bt},"/sessions/beautiful-vigilant-ritchie/mnt/outputs/modelo_rede_neural.joblib")
plt.figure(figsize=(5.2,3.8)); plt.plot(clf.loss_curve_,color="#5b4bdb",marker="o",ms=3)
plt.xlabel("Epoca"); plt.ylabel("Loss (entropia cruzada)"); plt.title("Curva de loss - rede neural (MLP)")
plt.tight_layout(); plt.savefig(FIG+"/fig_loss_rede_neural.png",dpi=150); plt.close()
plt.figure(figsize=(4.2,3.6)); plt.imshow(cm,cmap="Greens")
for i in range(2):
    for j in range(2): plt.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=13)
plt.xticks([0,1],["nao-toxico","toxico"]); plt.yticks([0,1],["nao-toxico","toxico"])
plt.xlabel("Predito"); plt.ylabel("Real"); plt.title("Matriz de confusao - rede neural (MLP)")
plt.colorbar(); plt.tight_layout(); plt.savefig(FIG+"/fig_matriz_confusao_mlp.png",dpi=150); plt.close()
