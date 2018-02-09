import numpy as np
import pylab as plt
import healpy as hp
from scipy import optimize
import capo as C

nblock=np.load('nblock.npy')
blm=np.load('blm_freq150.npy')
file_len=np.load('multifile_length.npy')
beam_dot_beam_conj=np.load('beam_time_beam_conj.npy')
fit_model_alltime=np.load('fit_model_alltime.npy')
old_edges=np.load('old_edges.npy')
edge=np.load('edge.npy')
u_old=np.load('u_old.npy')/3.0;v_old=np.load('v_old.npy')/3.0;w_old=np.load('w_old.npy')/3.0
bline_old=np.load('bline.npy')
bline_avg=np.load('bline_avg.npy')
xdata=np.load('xdata.npy')
ydata_alltime=np.load('ydata_alltime.npy')
ydata_alltime_allfreq=np.load('ydata_alltime_allfreq.npy')
beam=np.load('XX_beam_maps.npy')
cll_singlefile_alltime=np.load('cll_alltime.npy')
cll_multifile_alltime=np.load('cll_multifile_alltime.npy')
mean_cll_singlefile_alltime=np.load('mean_cll_singlefile_alltime.npy')
mean_cll_allfile_alltime=np.load('mean_cov_vis_multifile_alltime.npy')
#####################
nside=64
npix=hp.nside2npix(nside)
lmax=3*nside - 1;l,m = hp.Alm.getlm(lmax)
####LSTS
'''
info=C.arp.get_dict_of_uv_data(['zen.2456242.17382.uvcRREcACO'],'1_4','yy')[0]
LST=info['lsts']
times=info['times']
int_time_rad=LST[1]-LST[0]#integration time in radians
int_time_sec=(int_time_rad*24*60*60)/(2*np.pi)#integration time in seconds
snapshot_time_min=(14*int_time_sec)/60.0#total integration time of a snapshot file in minutes

nfreq=10
c=3e8
BEAM=beam[0:10]
s  = np.array(hp.pix2vec(nside,np.arange(npix)))
n_blm=hp.map2alm(beam[0],lmax=lmax,iter=3).shape[0]
nu = np.outer(np.linspace(117e6,182e6,num=nfreq),np.ones(npix))#unit hz
N=len(LST)
########simulating for one baseline
'''

'''
u=np.load('u_old.npy')[0];v=np.load('v_old.npy')[0];w=np.load('w_old.npy')[0]
b = np.resize(np.repeat(np.array([u,v,w]),npix),[3,npix])#*u.meter
b_dot_s = np.sum(b*s,axis=0)
factor = np.exp(1.j*np.outer(np.ones(nfreq),b_dot_s)*nu/c)
BLM = np.zeros((nfreq,n_blm),dtype='complex128')
for i in range(nfreq):
	BLM[i,:] = hp.map2alm((BEAM*factor)[i,:],lmax=lmax,iter=3)
rot_ang = np.linspace(-np.pi,np.pi,num=360*4)#LSTs
n = len(rot_ang)

vis = np.zeros([n,nfreq],dtype='complex128')
rot=[]
for i in range(n):
	rotation = np.outer(np.ones(nfreq),np.exp(-1.j*m*rot_ang[i]))
    vis[i,:] = np.sum(BLM*rotation,axis=1)#vis from beam and fringe part onl
'''
##################beam_alm_time dependence
'''
time_dependent_blm=np.zeros((LST.size,l.size),dtype='complex128')
bblm=[]
for i in range(old_edges[10]):#only some of the first redundent blocks
	for j in range(LST.size):
		bblm.append(blm[i]*np.exp(-1j*m*LST[j]))
bblm=np.sum(np.asarray(bblm).reshape(old_edges[10],LST.size,l.size),axis=2)		
'''		

group_blm=[]#grouping blm according to redundent blocks at single frequency and time
for i in range(old_edges.shape[0]-1):
	group_blm.append(blm[old_edges[i]:old_edges[i+1]])
group_blm=np.asarray(group_blm)

sum_blm_over_l=[]
for i in range(old_edges.shape[0]-1):
	sum_blm_over_l.append(np.sum(group_blm[i],axis=1))
sum_blm_over_l=np.asarray(sum_blm_over_l)
blm_times_blm=sum_blm_over_l*sum_blm_over_l
np.save('blm_times_blm.npy',blm_times_blm)

std_blm=np.zeros(old_edges.shape[0]-1)
for i in range(old_edges.shape[0]-1):
	std_blm[i]=np.std(abs(sum_blm_over_l[i]))
old_bline_std=np.zeros(old_edges.shape[0]-1)
for i in range(old_edges.shape[0]-1):
	old_bline_std[i]=np.std(bline_old[old_edges[i]:old_edges[i+1]])
	
index_old_bline_std=np.where(old_bline_std == old_bline_std.max())
index_std_blm=np.where(std_blm == std_blm.max())			
##mean_cll_singlefile_divided by mean beam_times_beam
mean_cll_per_blm_square=[]#for single file only
for i in range(nblock):
	mean_cll_per_blm_square.append(mean_cll_allfile_alltime[0][i]/blm_times_blm[i])
mean_cll_per_blm_square=np.asarray(mean_cll_per_blm_square)	
		
blm_times_cll=np.zeros(nblock,dtype='complex128')
for i in range(nblock):
	blm_times_cll[i]=np.dot(blm_times_blm[i],mean_cll_per_blm_square[i])


blm_outer_cll=[]
for i in range(nblock):
	blm_outer_cll.append(np.outer(blm_times_blm[i],mean_cll_per_blm_square[i]))









