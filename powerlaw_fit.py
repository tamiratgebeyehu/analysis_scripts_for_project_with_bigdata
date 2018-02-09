import numpy as np
import pylab as plt
import healpy as hp
from scipy import optimize
import pandas as pd

old_num_unique_bl=np.load('old_num_unique_bl.npy')
nblock=np.load('nblock.npy')
old_edges=np.load('old_edges.npy')
u_old=np.load('u_old.npy')/3.0;v_old=np.load('v_old.npy')/3.0;w_old=np.load('w_old.npy')/3.0
bline_avg=np.load('bline_avg.npy')
xdata=np.load('xdata.npy')
mean_vis_cov_each_file_alltime=np.load('mean_cov_vis_multifile_alltime.npy')
mean_vis_cov_allfile=np.mean(mean_vis_cov_each_file_alltime,axis=0)
beam=np.load('XX_beam_maps.npy')
blm_times_blm=np.load('blm_times_blm.npy')
lenfile=np.load('multifile_length.npy')

nside=64
npix = hp.nside2npix(nside)
ipix = np.arange(npix)
theta,phi = hp.pix2ang(nside,ipix)
s=np.array(hp.pix2vec(nside,ipix))
tx=s[0];ty=s[1];tz=s[2];lmax=3*nside - 1;l,m = hp.Alm.getlm(lmax)
freqs=np.linspace(0.117,0.182,num=203)
######Removing extreme outlier from mean_cll_allfile_alltime data
sorted_cll=[]
for i in range(lenfile):
	sorted_cll.append(np.sort(mean_vis_cov_each_file_alltime[i]))
mean_cll_threshold=np.zeros(lenfile,dtype='complex128')
for i in range(lenfile):
	mean_cll_threshold[i]=sorted_cll[i][nblock-2]
df=np.array([np.where(pd.DataFrame(mean_vis_cov_each_file_alltime[i]) >mean_cll_threshold[i], 
np.mean(mean_vis_cov_each_file_alltime[i]),pd.DataFrame(mean_vis_cov_each_file_alltime[i]))for i in range(lenfile)])
df_flatten=[]
for i in range(lenfile):
	df_flatten.append(df[i].flatten())
	
	
max_mean_vis_cov_allfile=np.sort(mean_vis_cov_allfile)[nblock-2]
df_allfile=np.where(pd.DataFrame(mean_vis_cov_allfile) >max_mean_vis_cov_allfile, 
np.mean(mean_vis_cov_allfile),pd.DataFrame(mean_vis_cov_allfile))
df_allfile=df_allfile.flatten()

###############################

powerlaw_alltime = lambda x, amp,coeff, index: amp * (x**index)+coeff
fitfunc_alltime = lambda p, x: p[0]*x**p[1]+p[2]

errfunc_alltime = lambda p, x, y: (y - fitfunc_alltime(p, x))

inds = np.argsort(xdata)
xdata = xdata[inds]
#ydata_alltime=mean_vis_cov_each_file_alltime[0][inds]
ydata_alltime=df_flatten[1][inds]
ydata_alltime_allfile=df_allfile[inds]

logx = np.log10(xdata)
logy = np.log10(np.abs(ydata_alltime))


pinit = [1.0, 1.0,0.]
out_alltime = optimize.leastsq(errfunc_alltime, pinit,args=(logx, logy), full_output=1)

pfinal_alltime= out_alltime[0]
covar_alltime = out_alltime[1]

amp_alltime =-2e6*pfinal_alltime[0]
index_alltime =-2*pfinal_alltime[1]
coeff_alltime=1*pfinal_alltime[2]
model_fit_alltime=powerlaw_alltime(xdata, amp_alltime, coeff_alltime,index_alltime)
fit_model_alltime=np.save('fit_model_alltime.npy',model_fit_alltime)

model_per_blms_sq=[]
for i in range(nblock):
	model_per_blms_sq.append(model_fit_alltime[i]/blm_times_blm[inds[i]])
model_per_blms_sq==np.asarray(model_per_blms_sq)

out_model=np.zeros(nblock,dtype='complex128')
for i in range(nblock):
	out_model[i]=np.mean(np.triu(np.outer(blm_times_blm[inds[i]],model_per_blms_sq[i]),k=0))	
	
#indexErr = np.sqrt( covar[1][1] )
#ampErr = np.sqrt( covar[0][0] ) * amp
plt.plot(xdata,np.abs(ydata_alltime),'mo',label='Data points')
plt.legend()
plt.plot(xdata, model_fit_alltime, 'r',linewidth=3,label=r'$a+bx^{\alpha}$,(Best-fit)')

plt.xlabel('Baseline length (m)')
plt.ylabel('$<V_{i}^{*}V_{j}>$')
plt.plot(xdata,out_model,'b',linewidth=3,label='Mean visibility covarience derived from best-fit model and beam')

#plt.plot(xdata,vis_cov_from_model,'b')
plt.legend()
plt.savefig('Best-fit_powelaw_singlefile.png')
plt.show()

###################################################all one hour file
powerlaw_alltime_allfile = lambda x, amp,coeff, index: amp * (x**index)+coeff
fitfunc_alltime_allfile = lambda p, x: p[0]*x**p[1]+p[2]

errfunc_alltime_allfile = lambda p, x, y: (y - fitfunc_alltime(p, x))

ydata_alltime_allfile=df_allfile[inds]

logy_allfile = np.log10(np.abs(ydata_alltime_allfile))


pinit = [1.0, 1.0,0.]
out_alltime_allfile = optimize.leastsq(errfunc_alltime, pinit,args=(logx, logy_allfile), full_output=1)

pfinal_alltime_allfile= out_alltime[0]
covar_alltime_allfile = out_alltime[1]

amp_alltime_allfile =-2.6e6*pfinal_alltime_allfile[0]
index_alltime_allfile =-1.7*pfinal_alltime_allfile[1]
coeff_alltime_allfile=100*pfinal_alltime_allfile[2]
model_fit_alltime_allfile=powerlaw_alltime(xdata, amp_alltime_allfile, coeff_alltime_allfile,index_alltime_allfile)
fit_model_alltime=np.save('fit_model_alltime_allfile.npy',model_fit_alltime_allfile)

get_model_per_blms_sq_allfile=[]
for i in range(nblock):
	get_model_per_blms_sq_allfile.append(model_fit_alltime_allfile[i]/blm_times_blm[inds[i]])
get_model_per_blms_sq_allfile=np.asarray(get_model_per_blms_sq_allfile)

mean_vis_cov_from_model_allfile=np.zeros(nblock,dtype='complex128')
for i in range(nblock):
	mean_vis_cov_from_model_allfile[i]=np.mean(np.triu(np.outer(blm_times_blm[inds[i]],get_model_per_blms_sq_allfile[i]),k=0))
plt.plot(xdata,ydata_alltime_allfile,'mo',label='Data points')
plt.plot(xdata,model_fit_alltime_allfile,'r',linewidth=3,label=r'$a+bx^{\alpha}$,(Best-fit)')
plt.plot(xdata,mean_vis_cov_from_model_allfile,'b',linewidth=3,label='Mean visibility covarience derived from best-fit model and beam')
plt.xlabel('Baseline length (m)')
plt.ylabel('$<V_{i}^{*}V_{j}>$')
plt.legend()
plt.savefig('Best-fit_powelaw_allfile.png')
plt.show()

##################################################
get_model_cov_singlefile=[]
for i in range(nblock):
	get_model_cov_singlefile.append(np.outer(blm_times_blm[inds[i]],model_per_blms_sq[i]))
get_model_cov_singlefile=np.asarray(get_model_cov_singlefile)	

get_model_cov_allfile=[]
for i in range(nblock):
	get_model_cov_allfile.append(np.outer(blm_times_blm[inds[i]],get_model_per_blms_sq_allfile[i]))
get_model_cov_allfile=np.asarray(get_model_cov_allfile)	
#Separate real and imaginary part of blm_times_blm and model_per_blms_sq
real_imag_blm_times_blm=[np.zeros(2*old_num_unique_bl[inds][i]) for i in range(nblock)]
for i in range(nblock):
	real_imag_blm_times_blm[i][0::2]=blm_times_blm[inds][i].real
	real_imag_blm_times_blm[i][1::2]=blm_times_blm[inds][i].imag
real_imag_model_per_blms_sq=[np.zeros(2*old_num_unique_bl[inds][i]) for i in range(nblock)]
for i in range(nblock):
	real_imag_model_per_blms_sq[i][0::2]=model_per_blms_sq[i].real
	real_imag_model_per_blms_sq[i][1::2]=model_per_blms_sq[i].imag

real_imag_model_cov_sfile=[]#single file
for i in range(nblock):
	real_imag_model_cov_sfile.append(np.outer(real_imag_blm_times_blm[i],real_imag_model_per_blms_sq[i]))
		



##################################################################
'''
powerlaw_alltime_allfreq = lambda x, amp,coeff, index: amp * (x**index)+coeff

inds = np.argsort(xdata)
xdata = bline_avg[inds]
ydata_alltime_allfreq = ydata_alltime_allfreq[inds]
logx = np.log10(xdata)
logy = np.log10(abs(ydata_alltime_allfreq))
fitfunc_alltime_allfreq = lambda p, x: p[0]*x**p[1]+p[2]
errfunc_alltime_allfreq = lambda p, x, y: (y - fitfunc_alltime_allfreq(p, x))

pinit = [1.0, 1.0,0.]
out_alltime_allfreq = optimize.leastsq(errfunc_alltime_allfreq, pinit,args=(logx, logy), full_output=1)

pfinal_alltime_allfreq= out_alltime_allfreq[0]
covar_alltime_allfreq = out_alltime_allfreq[1]

amp_alltime_allfreq =11000*pfinal_alltime_allfreq[0]
index_alltime_allfreq =900*pfinal_alltime_allfreq[1]
coeff_alltime_allfreq=5*pfinal_alltime_allfreq[2]

#indexErr = np.sqrt( covar[1][1] )
#ampErr = np.sqrt( covar[0][0] ) * amp
plt.plot(xdata,np.abs(ydata_alltime_allfreq),'bo',label='Data points for all freqs and time')
plt.legend()
#plt.plot(xdata, powerlaw_alltime_allfreq(xdata, amp_alltime_allfreq, coeff_alltime_allfreq,index_alltime_allfreq),
 #'r',linewidth=3,label=r'$a+bx^{\alpha}$,(Best-fit)')
plt.xlabel('Baseline length (m)')
plt.ylabel('$<V_{i}^{*}V_{j}>$')
plt.legend()
plt.savefig('Best-fit_powelaw.png')
plt.show()
'''
####################################################################

