"""Draw workflow, result plots and SVD-input illustrations."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import argparse
parser=argparse.ArgumentParser(description='Draw study figures from reproduced statistics.')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'output')
P=parser.parse_args().output
P.mkdir(parents=True,exist_ok=True)
R=P
d=json.loads((R/'results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#203647','text.color':'#203647','xtick.color':'#203647','ytick.color':'#203647','savefig.facecolor':'white'})
C=['#008E91','#D98732','#7065B2'];M=['CDLD','responseSVD','exposureSVD'];L=['CDLD','Response SVD','Exposure SVD']
def save(fig,name):
 fig.savefig(P/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(P/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
# 1 flow
f,ax=plt.subplots(figsize=(15,8));ax.set(xlim=(0,15),ylim=(0,8));ax.axis('off')
f.suptitle('One set of learned vectors, two evaluation tasks',fontsize=23,fontweight='bold',y=.96)
def box(x,y,w,h,title,body,color='#EAF0F4',edge='#C4D1DA',fs=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',fc=color,ec=edge,lw=1.3))
 ax.text(x+w/2,y+h-.12,title,ha='center',va='top',weight='bold',fontsize=fs+1)
 ax.text(x+w/2,y+h-.43,body,ha='center',va='top',fontsize=fs,linespacing=1.15)
def arrow(x,y,u,v):ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=14,color='#607785',lw=1.5))
box(.2,6.65,6.9,.72,'EdNet · English learning','Last response per student–item pair  |  79.9 million responses')
box(7.9,6.65,6.9,.72,'Eedi · Mathematics','Tasks 1–2 training data  |  15.9 million responses')
arrow(3.65,6.58,5,6.25);arrow(11.35,6.58,10,6.25)
box(.9,5.32,13.2,.85,'Five shared runs · seeds 42–46','Training / validation responses → learn representations and select predictors\nHeld-out correctness: 500 eligible students × 40 responses per run',fs=12)
for x,col,title,body in [(0.3,C[0],'CDLD','Student latent Finder ↔ Item latent Finder\nLearn from response correctness'),(5.3,C[1],'Response SVD','Observed correctness − item mean\nMissing entries → 0'),(10.3,C[2],'Exposure SVD','Observed = 1; unobserved = 0\nNo correctness labels')]:
 arrow(x+2.2,5.24,x+2.2,4.90)
 box(x,3.92,4.4,.9,title,body,color=col+'18',edge=col,fs=10.5)
 arrow(x+2.2,3.84,x+2.2,3.48)
box(.9,2.96,13.2,.44,'Freeze 64-dimensional student and item vectors','',color='#EAF4F2',edge='#8BB5AF')
arrow(4,2.88,4,2.48);arrow(11,2.88,11,2.48)
box(.3,.85,6.8,1.55,'Item-category recovery','Item vectors alone or added to IRT + summaries (B)\nNested logistic regression · item folds\nEdNet: 7 parts + bundle-grouped evaluation\nEedi: 4 mathematics domains',color='#EAF4F8',fs=11)
box(7.9,.85,6.8,1.55,'Response correctness','Student + item vectors + training item proportion correct\nCommon MLP and bilinear predictors\nSame held-out student–item responses\nExisting students and existing items',color='#F4F0F8',fs=11)
f.text(.5,.015,'Item-category labels are used only for classification. Held-out evaluation responses are excluded from representation learning.',ha='center',fontsize=11)
save(f,'figure1')

#2 classification
f,axs=plt.subplots(1,2,figsize=(13,5.5));f.suptitle('Which representations recover item categories?',fontsize=21,weight='bold',y=1.02)
for ax,ds in zip(axs,['EdNet','Eedi']):
 v=d[ds+'_item']['methods'];y=np.arange(3);w=.32
 for j,(m,col) in enumerate(zip(M,C)):
  for off,key,alpha in [(-w/2,m,.4),(w/2,'B_'+m,1)]:
   val=v[key]['logloss']['mean'];ax.barh(j+off,val,height=w*.88,color=col,alpha=alpha)
   ax.text(val+.025,j+off,f'{val:.4f}',va='center',fontsize=10)
 b=v['B']['logloss']['mean'];ax.axvline(b,color='#687984',ls='--',lw=1.4)
 ax.text(b-.025,-.6,f'B alone {b:.4f}',ha='right',fontsize=10,color='#52616D')
 ax.set(yticks=y,yticklabels=L,xlim=(0,1.85 if ds=='EdNet' else 1.55),ylim=(2.65,-.85),xlabel='Classification log loss  ↓ lower is better',title=ds)
 ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
from matplotlib.patches import Patch
f.legend(handles=[Patch(fc='#526B78',alpha=.4,label='Representation alone'),Patch(fc='#526B78',label='B + representation')],loc='lower center',ncol=2,bbox_to_anchor=(.5,-.03),frameon=False)
f.tight_layout(rect=[0,.07,1,.97]);save(f,'figure2')
#3 correctness
f,axs=plt.subplots(2,2,figsize=(12,8));f.suptitle('Correctness performance depends on the predictor',fontsize=21,weight='bold',y=1.0)
for col,ds in enumerate(['EdNet','Eedi']):
 with (R/f'{ds}_correctness_runs.csv').open() as inp:rows=list(csv.DictReader(inp))
 for row,kind in enumerate(['MLP','bilinear']):
  ax=axs[row,col]
  for j,(m,color) in enumerate(zip(M,C)):
   vals=[float(r['logloss']) for r in rows if r['method']==f'{kind}_{m}_rate']
   ax.scatter(np.array(vals),j+np.linspace(-.1,.1,5),s=30,color=color,alpha=.45,zorder=3)
   mean=np.mean(vals);ax.plot([mean,mean],[j-.19,j+.19],color=color,lw=3,zorder=4)
   ax.text(.605,j,f'{mean:.4f}',va='center',ha='left',fontsize=11,color=color,weight='bold')
  ax.set(yticks=range(3),yticklabels=L,xlim=(.50,.625),ylim=(2.55,-.5),title=f'{ds} · {kind}',xlabel='Correctness log loss  ↓ lower is better')
  ax.set_xticks([.50,.52,.54,.56,.58,.60]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
f.text(.5,.015,'All predictors receive the training item proportion correct. Small dots: five runs. Vertical marks and labels: means (not confidence intervals).',ha='center',fontsize=10)
f.tight_layout(rect=[0,.055,1,.97]);save(f,'figure3')
#4 recall
f,axs=plt.subplots(1,2,figsize=(13,5));f.suptitle('Eedi: overall loss and recognition of each domain are different questions',fontsize=19,weight='bold',y=1.03)
for ax,keys,title in zip(axs,[M,['B_'+m for m in M]],['Representation alone','B + representation']):
 vals=np.array([d['domains'][m]['recall'] for m in keys])*100
 im=ax.imshow(vals,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
 for i in range(3):
  for j in range(4):ax.text(j,i,f'{vals[i,j]:.1f}%',ha='center',va='center',fontsize=12,color='white' if vals[i,j]>60 else '#203647',weight='bold')
 ax.set(xticks=range(4),xticklabels=['Number','Algebra','Geometry\n& Measure','Data &\nStatistics'],yticks=range(3),yticklabels=L,title=title)
 ax.tick_params(length=0);ax.spines[:].set_visible(False)
f.subplots_adjust(left=.10,right=.91,wspace=.42,bottom=.22,top=.84)
cax=f.add_axes([.935,.24,.015,.57]);f.colorbar(im,cax=cax,label='Recall (%)')
f.text(.5,.045,'Mean domain recall over five runs. Each domain receives equal visual weight; values are not weighted by domain prevalence.',ha='center',fontsize=10)
save(f,'figureS2')

# 5 explanatory matrix schematic, explicitly not empirical results
Y=np.array([[1,1,np.nan,0,np.nan,1],[0,np.nan,1,0,1,np.nan],[1,0,np.nan,1,0,1],[np.nan,1,0,np.nan,1,0],[0,np.nan,1,1,np.nan,1]],float)
obs=np.isfinite(Y);means=np.nanmean(Y,axis=0);X=np.where(obs,Y-means,0);O=obs.astype(int)
f,axs=plt.subplots(1,3,figsize=(13,5));f.suptitle('The same observed pairs yield two different SVD inputs',fontsize=20,weight='bold',y=1.03)
from matplotlib.colors import ListedColormap
raw=np.where(obs,Y,-1)
for ax,A,title in zip(axs,[raw,X,O],['Training responses','Response SVD input','Exposure SVD input']):
 if title=='Training responses':im=ax.imshow(A,cmap=ListedColormap(['#EDF0F2','#EAB7A0','#5AB4B5']),vmin=-1,vmax=1)
 elif title=='Response SVD input':im=ax.imshow(A,cmap='BrBG',vmin=-1,vmax=1)
 else:im=ax.imshow(A,cmap=ListedColormap(['#EDF0F2','#A69CCC']),vmin=0,vmax=1)
 for i in range(5):
  for j in range(6):
   s=('—' if not obs[i,j] else str(int(Y[i,j]))) if title=='Training responses' else (f'{A[i,j]:.2f}' if title=='Response SVD input' else str(int(A[i,j])))
   ax.text(j,i,s,ha='center',va='center',fontsize=11,color='#183445')
 ax.set(xticks=range(6),xticklabels=[f'I{j+1}' for j in range(6)],yticks=range(5),yticklabels=[f'S{i+1}' for i in range(5)],xlabel='',ylabel='Students',title=title)
 ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='both',length=0);ax.spines[:].set_visible(False)
axs[0].text(.5,-.25,'1 = correct · 0 = incorrect\n— = unobserved',transform=axs[0].transAxes,ha='center',fontsize=11)
axs[1].text(.5,-.25,'Observed: correctness − item mean\nUnobserved: 0',transform=axs[1].transAxes,ha='center',fontsize=11)
axs[2].text(.5,-.25,'Observed: 1, regardless of correctness\nUnobserved: 0',transform=axs[2].transAxes,ha='center',fontsize=11)
f.subplots_adjust(left=.06,right=.98,top=.81,bottom=.28,wspace=.35)
f.text(.5,.015,'Illustrative 5-student × 6-item matrix, not study data. Item means are calculated from observed training responses.',ha='center',fontsize=10)
save(f,'figureS1')
