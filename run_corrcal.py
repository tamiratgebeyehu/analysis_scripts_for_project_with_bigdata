import numpy as np,corrcal2,time
from matplotlib import pyplot as plt
from scipy.optimize import fmin_cg

ant1=np.load('ant1.npy');                   ant2=np.load('ant2.npy'); edges=np.load('edge.npy')
nblock=np.load('nblock.npy');               data_single_freq=np.load('bigdata_real_imag_paired_vis_single_freq.npy')
src=np.zeros((data_single_freq.size));      vecs=np.load('bigvec.npy').transpose()
antpos=np.load('antpos.npy');               antlayout=np.load('antlayout.npy')
uu=np.load('uu.npy');                       vv=np.load('vv.npy')
  

gains_sol =[]
mean_gains=[]
std_abs_gains=[]
std_amp_gains=[]
std_phase_gains=[]
std_sky=[]
gains = np.zeros(antlayout.shape[0]**2,dtype='complex')
eta_0, phi_0 = np.zeros(antlayout.shape[0]**2), np.zeros(antlayout.shape[0]**2)
for g in range(antlayout.shape[0]**2):
	eta= np.random.normal(0.0,1.0)
	eta_0[g]=eta
	amp = np.random.uniform(0.7,1.0)
	phase = np.random.uniform(0.0,np.pi)
	phi_0[g] = phase
	gains[g]=amp*(np.cos(phase)+1j*np.sin(phase))
nant=np.max([np.max(ant1),np.max(ant2)])+1



#gvec=np.zeros(2*nant)
#gvec = np.concatenate((np.real(gains),np.imag(gains)))

#sig=np.std(data)
#snr=1000;
#noise_level=sig/snr
#noise=np.ones(data.size)*noise_level*noise_level
dat_use=data_single_freq #+ noise_level*numpy.random.randn(bigdata[nu_i].size)
noise=np.ones(uu.size)
big_noise=np.zeros(2*noise.size)
big_noise[0::2]=noise
big_noise[1::2]=noise
mat=corrcal2.sparse_2level(big_noise,vecs,src*1,2*edges)
	
nant=np.max([np.max(ant1),np.max(ant2)])+1
gvec=np.zeros(2*ant2.max()+2)
gvec[0::2]=1.0
gvec=gvec+0.1*np.random.randn(gvec.size)
gvec[0]=1
gvec[1]=0


for i in range(len(edges)-1):
	mystd=np.std(abs(uu[edges[i]:edges[i+1]]))+np.std(abs(vv[edges[i]:edges[i+1]]))
	print edges[i],edges[i+1],mystd
		

fac=1000.0
normfac=1
t1=time.time()
fdsa=fmin_cg(corrcal2.get_chisq,gvec*fac,corrcal2.get_gradient,(dat_use,mat,ant1,ant2,fac,normfac))
t2=time.time()
#print 'elapsed time to do nonlinear fit for ' + repr(nant) + ' antennas was ' + repr(t2-t1)
gains=fdsa/fac
gg=gains[0::2]+np.complex(0,1)*gains[1::2]
gg= gg/np.absolute(np.mean(gg)) # removing the degeneracies by dividing by an average of gains (absolute) from that frquency
#print 'average gains', np.mean(gg), 'std abs gains', np.std(np.abs(gg)), 'std amp gains', np.std(gains[0::2]), 'std phase', np.std(gains[1::2])
#print 'standard devaiation signal',sig, 'noise_level',noise_level, 'signal_noise_ratio',snr, 'average_offset_gains',np.mean(0.2*np.random.randn(gvec.size))
#true_sky = np.power(numpy.conj(gg[ant1])*gg[ant1],-1)*data_use
#print np.std(true_sky)
gains_sol.append(gg)
mean_gains.append(np.mean(gg))
std_abs_gains.append(np.std(np.abs(gg)))
std_amp_gains.append(np.std(gains[0::2]))
std_phase_gains.append(np.std(gains[1::2]))
        	
	
	
	
	
	
	
	
	
	
