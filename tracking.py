import cv2, numpy as np, os, csv, shutil, tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
tmp=Path(tempfile.gettempdir())/'lab2_tracking_result.mp4'
cap=cv2.VideoCapture(str(root/'video_source.mp4'))
start,end=85,184
cap.set(cv2.CAP_PROP_POS_FRAMES,start); ok,frame=cap.read(); assert ok
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
mask=np.zeros_like(gray); mask[865:955,425:585]=255
pts=cv2.goodFeaturesToTrack(gray, maxCorners=90, qualityLevel=.008, minDistance=4, mask=mask, blockSize=5)
print('initial points',len(pts))
model=cv2.imdecode(np.fromfile(root/'model.png',dtype=np.uint8),cv2.IMREAD_UNCHANGED); assert model is not None and model.shape[2]==4
writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),29.97,(720,1280)); assert writer.isOpened()
M=np.eye(3); records=[]
for f in range(start,end+1):
 if f>start:
  ok,frame=cap.read(); assert ok
  nxt=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  dst,status,err=cv2.calcOpticalFlowPyrLK(gray,nxt,pts,None,winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
  a=pts[status.ravel()==1].reshape(-1,2); b=dst[status.ravel()==1].reshape(-1,2)
  A,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=1000)
  if A is None or inliers is None or inliers.sum()<5: raise RuntimeError(f'track lost at frame {f}: {len(a)}')
  M=np.vstack([A,[0,0,1]])@M
  pts=b[inliers.ravel().astype(bool)].reshape(-1,1,2).astype(np.float32); gray=nxt
 anchor=M@np.array([512,865,1]); scale=(M[0,0]**2+M[1,0]**2)**.5; angle=np.degrees(np.arctan2(M[1,0],M[0,0]))
 size=120*scale
 # Position the bottom of the 3D-rendered sign on the tracked camera body.
 T=cv2.getRotationMatrix2D((128,128),-angle,size/256); T[:,2]+=np.array([anchor[0]-128,anchor[1]-120])-T[:,:2]@np.array([128,128]) + np.array([128,128])
 # Define the sprite centre directly in image coordinates.
 T=cv2.getRotationMatrix2D((128,128),-angle,size/256); T[:,2]+=np.array([anchor[0]-128,anchor[1]-size*.57-128])
 over=cv2.warpAffine(model,T,(720,1280),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
 alpha=over[:,:,3:4].astype(np.float32)/255.; frame=(frame*(1-alpha)+over[:,:,:3]*alpha).astype(np.uint8)
 writer.write(frame); records.append([f-start+1,f,round(float(anchor[0]),2),round(float(anchor[1]),2),round(float(scale),4),round(float(angle),2),len(pts)])
 if f in (85,120,160,184): cv2.imencode('.png',frame)[1].tofile(root/f'кадр_{f}.png')
writer.release(); cap.release()
shutil.copy(str(tmp),root/'result_rebuilt.mp4')
with open(root/'tracking.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f); w.writerow(['output_frame','source_frame','x_px','y_px','scale','angle_deg','points']); w.writerows(records)
print('first',records[0],'last',records[-1],'min_points',min(r[-1] for r in records))
