import cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=""; O="/report_build/figs/"
NAVY="#1F3A5F"; TEAL="#2A9D8F"; AMBER="#E9A23B"; CORAL="#E76F51"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False})

# 1. training curves
h=pd.read_csv(R+"final/unet_128_multiscale_v1/history.csv")
fig,ax=plt.subplots(1,2,figsize=(3.4,1.9),dpi=300)
ax[0].plot(h.epoch,h.train_loss,color=NAVY,lw=1.2,label="train")
ax[0].plot(h.epoch,h.val_loss,color=CORAL,lw=1.2,label="val")
ax[0].set_xlabel("epoch"); ax[0].set_title("Loss",fontsize=8); ax[0].legend(frameon=False,fontsize=6.5)
ax[1].plot(h.epoch,h.val_miou,color=NAVY,lw=1.2,label="mIoU")
ax[1].plot(h.epoch,h.val_boundary_f1,color=TEAL,lw=1.2,label="BF1")
ax[1].plot(h.epoch,h.val_quality,color=AMBER,lw=1.2,label="Q")
ax[1].axvline(56,color="#999",lw=.6,ls="--"); ax[1].set_xlabel("epoch"); ax[1].set_title("Validation",fontsize=8)
ax[1].legend(frameon=False,fontsize=6.5,loc="lower right")
for a in ax: a.grid(alpha=.25,lw=.4)
fig.tight_layout(pad=.4); fig.savefig(O+"curves.pdf"); plt.close(fig)

# 2. captured overlays: input / GT overlay / prediction overlay
miou=[0.915,0.579,0.809,0.938,0.944,0.950,0.648]
names=["Chair","Suitcase","Bicycles","Backpack","Clogs","Controller","Mouse"]
def overlay(img,m,col):
    out=img.astype(float).copy(); c=np.array(col,float)
    out[m>0]=0.55*out[m>0]+0.45*c
    cs,_=cv2.findContours((m>0).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    out=out.astype(np.uint8); cv2.drawContours(out,cs,-1,col,1); return out
fig,ax=plt.subplots(3,7,figsize=(7.2,3.55),dpi=300)
for i in range(7):
    img=cv2.cvtColor(cv2.imread(R+f"captured/images_256/{i+1}.png"),cv2.COLOR_BGR2RGB)
    gt=cv2.imread(R+f"captured/masks_256_binary/{i+1}.png",0)
    pr=cv2.imread(R+f"captured/evaluation/predictions/{i+1}.png",0)
    for r,im in enumerate([img,overlay(img,gt,(42,157,143)),overlay(img,pr,(231,111,81))]):
        ax[r,i].imshow(im); ax[r,i].set_xticks([]); ax[r,i].set_yticks([])
        for s in ax[r,i].spines.values(): s.set_visible(False)
    ax[0,i].set_title(f"{i+1}. {names[i]}\nmIoU {miou[i]:.3f}",fontsize=6.5,pad=2)
for r,t in enumerate(["Photo","Ground truth","Prediction"]):
    ax[r,0].set_ylabel(t,fontsize=7)
fig.subplots_adjust(wspace=.03,hspace=.03,left=.04,right=.995,top=.9,bottom=.005)
fig.savefig(O+"captured.pdf"); plt.close(fig)

# 3. DUTS-TE qualitative: rows (0-based) of the 8-row example sheet
sheet=cv2.cvtColor(cv2.imread(R+"final/unet_128_multiscale_v1/segmentation_examples_256.png"),cv2.COLOR_BGR2RGB)
H=sheet.shape[0]//8
rows=[2,7,3,5]
strip=np.vstack([sheet[r*H:(r+1)*H,:1350] for r in rows])
cv2.imwrite(O+"duts.png",cv2.cvtColor(strip,cv2.COLOR_RGB2BGR))
print(strip.shape)
