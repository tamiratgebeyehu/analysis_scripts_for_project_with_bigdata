import numpy as np
import pylab as plt


nblock=np.load('nblock.npy')
edge=np.load('edge.npy')
num_unique_bl=np.load('num_unique_bl.npy')
file_len=np.load('multifile_length.npy')
max_edge=edge.max()
red_block_single_file=np.load('red_block_single_file.npy')
red_block_multi_file=np.load('multi_file_red_block.npy')
freqs=np.linspace(0.117,0.182,num=203)
time_size=14


def regroup_single_file(edge):
	R_vis=[]
	for i in range(len(red_block_single_file)):
		for j in range(len(red_block_single_file[i])):
			R_vis.append(np.squeeze(np.asarray(red_block_single_file[i][j].values())))
	R_vis=np.asarray(R_vis)
	grouped_vis_multifreq=np.concatenate(R_vis)
	concatenated_vis=np.concatenate(grouped_vis_multifreq).flatten()
	conc_vis=[]
	for i in range(time_size*freqs.size):
		conc_vis.append(concatenated_vis[i::freqs.size*time_size])
	conc_vis=np.asarray(conc_vis)
	where_nan=np.isnan(conc_vis)
	conc_vis[where_nan]=0
	conc_vis_regrp=[]
	for i in range(edge.shape[0]-1):
		conc_vis_regrp.append(conc_vis[:,edge[i]:edge[i+1]])
	regrp_vis=[]
	for i in range(edge.shape[0]-1):
		regrp_vis.append(conc_vis_regrp[i].reshape(time_size,freqs.size,num_unique_bl[i]))
	return regrp_vis



def regroup_multifile(edge):
	#multifile stuff for all freq and time (shape=(time_size,freqs_size,no_redundnt_bline_in_a_block))
	multi_file_vis=[]
	for i in range(len(red_block_multi_file)):
		for j in range(len(red_block_multi_file[i])):
			multi_file_vis.append(np.squeeze(np.asarray(red_block_multi_file[i][j].values())))
	multi_file_vis=np.asarray(multi_file_vis)
	concatenate_multi_file_vis=np.concatenate(multi_file_vis).flatten()
	
	file_multi=[]
	for i in range(time_size*freqs.size):
		file_multi.append(concatenate_multi_file_vis[i::freqs.size*time_size])
	file_multi=np.asarray(file_multi)
	where_nan=np.isnan(file_multi)
	file_multi[where_nan]=0
	
	edges_files=1951*np.arange(file_len+1)
	grouped_file=[]
	for i in range(edges_files.shape[0]-1):
		grouped_file.append(file_multi[:,edges_files[i]:edges_files[i+1]])
	grouped_file=np.asarray(grouped_file)
	
	file_multi_regrp=[]
	for i in range(len(grouped_file)):
		for j in range(edge.shape[0]-1):
			file_multi_regrp.append(grouped_file[i][:,edge[j]:edge[j+1]])
	num_unique_bl_multifile=np.tile(num_unique_bl,len(file_multi_regrp))
		
	regrp_multi_file_vis=[]
	for i in range(len(file_multi_regrp)):
		regrp_multi_file_vis.append(file_multi_regrp[i].reshape(time_size,freqs.size,num_unique_bl_multifile[i]))
		
	grouping_edges=np.arange(file_len+1)*(edge.shape[0]-1)
	grouping_multi_files=[]
	for i in range(file_len):
		grouping_multi_files.append(regrp_multi_file_vis[grouping_edges[i]:grouping_edges[i+1]])
	return grouping_multi_files



def mean_vis_cov_single_file(edge):
	cll_alltime=[]
	for i in range(edge.shape[0]-1):
		for j in range(time_size):
			cll_alltime.append(np.mean(np.triu((np.outer(regroup_single_file(edge)[i][j][100],regroup_single_file(edge)[i][j][100].conj()))**2,k=1)))
	cll_alltime=np.asarray(cll_alltime)
	cll_alltime=cll_alltime.reshape(edge.shape[0]-1,time_size)
	mean_cll_alltime=np.mean(cll_alltime,axis=1)
	return mean_cll_alltime

def mean_vis_cov_multifile(edge):
	cll_multifile_alltime=[]
	for i in range(file_len):
		for j in range(edge.shape[0]-1):
			for m in range(time_size):
				cll_multifile_alltime.append(np.mean(np.triu((np.outer(regroup_multifile(edge)[i][j][m][100],
				regroup_multifile(edge)[i][j][m][100].conj()))**2,k=1)))
	cll_multifile_alltime=np.asarray(cll_multifile_alltime).reshape(file_len,edge.shape[0]-1,time_size)
	mean_cll_allfile_alltime=[]
	for i in range(file_len):
		mean_cll_allfile_alltime.append(np.mean(cll_multifile_alltime[i],axis=1))	
	return mean_cll_allfile_alltime
#mean_vis_cov_multi=mean_vis_cov_multifile(edge)
#np.save('mean_cov_vis_multifile.npy',mean_vis_cov_multi)
###########################################
R_vis=[]
for i in range(len(red_block_single_file)):
	for j in range(len(red_block_single_file[i])):
		R_vis.append(np.squeeze(np.asarray(red_block_single_file[i][j].values())))
R_vis=np.asarray(R_vis)
grouped_vis_multifreq=np.concatenate(R_vis)
concatenated_vis=np.concatenate(grouped_vis_multifreq).flatten()
conc_vis=[]
for i in range(time_size*freqs.size):
	conc_vis.append(concatenated_vis[i::freqs.size*time_size])
conc_vis=np.asarray(conc_vis)
where_nan=np.isnan(conc_vis)
conc_vis[where_nan]=0
conc_vis_regrp=[]
for i in range(edge.shape[0]-1):
	conc_vis_regrp.append(conc_vis[:,edge[i]:edge[i+1]])
regrp_vis=[]
for i in range(edge.shape[0]-1):
	regrp_vis.append(conc_vis_regrp[i].reshape(time_size,freqs.size,num_unique_bl[i]))
############################################
cll_alltime=[]
for i in range(nblock):
	for j in range(time_size):
		cll_alltime.append(np.mean(np.triu(np.outer(regrp_vis[i][j][100],regrp_vis[i][j][100].conj()),k=1)))
cll_alltime=np.asarray(cll_alltime)
cll_alltime=cll_alltime.reshape(nblock,time_size)
cll_singlefile_alltime=np.save('cll_alltime.npy',cll_alltime)
mean_cll_singlefile=np.mean(cll_alltime,axis=1)
np.save('mean_cll_singlefile_alltime.npy',mean_cll_singlefile)
############################################
multi_file_vis=[]
for i in range(len(red_block_multi_file)):
	for j in range(len(red_block_multi_file[i])):
		multi_file_vis.append(np.squeeze(np.asarray(red_block_multi_file[i][j].values())))
multi_file_vis=np.asarray(multi_file_vis)
concatenate_multi_file_vis=np.concatenate(multi_file_vis).flatten()
	
file_multi=[]
for i in range(time_size*freqs.size):
	file_multi.append(concatenate_multi_file_vis[i::freqs.size*time_size])
file_multi=np.asarray(file_multi)
where_nan=np.isnan(file_multi)
file_multi[where_nan]=0
	
edges_files=1951*np.arange(file_len+1)
grouped_file=[]
for i in range(edges_files.shape[0]-1):
	grouped_file.append(file_multi[:,edges_files[i]:edges_files[i+1]])
grouped_file=np.asarray(grouped_file)
	
file_multi_regrp=[]
for i in range(len(grouped_file)):
	for j in range(edge.shape[0]-1):
		file_multi_regrp.append(grouped_file[i][:,edge[j]:edge[j+1]])
num_unique_bl_multifile=np.tile(num_unique_bl,len(file_multi_regrp))
		
regrp_multi_file_vis=[]
for i in range(len(file_multi_regrp)):
	regrp_multi_file_vis.append(file_multi_regrp[i].reshape(time_size,freqs.size,num_unique_bl_multifile[i]))
		
grouping_edges=np.arange(file_len+1)*(edge.shape[0]-1)
grouping_multi_files=[]
for i in range(file_len):
	grouping_multi_files.append(regrp_multi_file_vis[grouping_edges[i]:grouping_edges[i+1]])
##########################################
cll_multifile_alltime=[]
for i in range(file_len):
	for j in range(edge.shape[0]-1):
		for m in range(time_size):
			cll_multifile_alltime.append(np.mean(np.triu(np.outer(grouping_multi_files[i][j][m][100],
			grouping_multi_files[i][j][m][100].conj()),k=1)))
cll_multifile_alltime=np.asarray(cll_multifile_alltime).reshape(file_len,edge.shape[0]-1,time_size)
mean_cll_allfile_alltime=[]
for i in range(file_len):
	mean_cll_allfile_alltime.append(np.mean(cll_multifile_alltime[i],axis=1))
mean_cll_allfile=np.asarray(mean_cll_allfile_alltime)
np.save('mean_cov_vis_multifile_alltime.npy',mean_cll_allfile)	
np.save('cll_multifile_alltime.npy',cll_multifile_alltime)
###########################################	
def cov_mat(M):
	#M must be transposed in order to proceed
	M=np.asarray(M)
	I=np.ones(M.shape[0]).reshape(M.shape[0],1)
	d=I.dot(I.T)
	a=M-d.dot(M)/(M.shape[0])
	return np.dot(a.T,a.conj())/(M.shape[0]-1)

#Visibility covarience

def cov(m):
    #return n.cov(m)
    X = np.array(m, ndmin=2, dtype=np.complex)
    X -= X.mean(axis=1)[(slice(None),np.newaxis)]
    N = X.shape[1]
    fact = float(N - 1)
    return (np.dot(X, X.T.conj()) / fact).squeeze()	
###################	
	
	
	
	
	





'''
def get_grouped_vis_for_all_channel(red_block):
	#Returns grouped visibility based on redundency. The grouped vis
	#has been obtained by combining all integration time for a baseline
	R_vis=[]
	for i in range(len(red_block)):
		for j in range(len(red_block[i])):
			R_vis.append(np.squeeze(np.asarray(red_block[i][j].values())))
	R_vis=np.asarray(R_vis)
	
	grouped_block=[]
	for i in range(edge.size-1):
		grouped_block.append(R_vis[edge[i]:edge[i+1]])
	grouped_block=np.asarray(grouped_block)
	
	sum_int_time=[]
	for i in range(edge.size-1):
		for j in range(len(grouped_block[i])):
			sum_int_time.append(np.mean(grouped_block[i][j],axis=0))
	sum_int_time=np.asarray(sum_int_time)
	
	int_time_sum_grouped=[]
	for i in range(edge.size-1):
		int_time_sum_grouped.append(sum_int_time[edge[i]:edge[i+1]])
	return int_time_sum_grouped,sum_int_time

data_all_chan_grouped,data_all_chan=get_grouped_vis_for_all_channel(red_block)[0],get_grouped_vis_for_all_channel(red_block)[1]
np.save('bigdata_grouped_for_all_freqs.npy',data_all_chan_grouped)
np.save('bigdata_not_grouped_all_frqs.npy',data_all_chan)	

def order_real_imag_vis_multifreq(edge):
	real_imag_ordered_vis=np.zeros((np.sum(num_unique_bl),2*freqs.size))
	for i in range(np.sum(num_unique_bl)):
		real_imag_ordered_vis[i][0::2]=np.real(data_all_chan[i])
		real_imag_ordered_vis[i][1::2]=np.imag(data_all_chan[i])
	group_real_imag_ordered_vis=[]
	for i in range(len(edge)-1):
		group_real_imag_ordered_vis.append(real_imag_ordered_vis[edge[i]:edge[i+1]])
	group_real_imag_ordered_vis=np.asarray(group_real_imag_ordered_vis)
	return group_real_imag_ordered_vis
np.save('bigdata_real_imag_paired_vis_multifreq.npy',order_real_imag_vis_multifreq(edge))
	
def group_vis_by_redundency(red_block):
	#Grouping visibility together from unique baselines for a single freq channel and int time
	grouped_vis=[]
	for i in range(len(red_block)):
		for j in range(len(red_block[i])):
			grouped_vis.append(np.squeeze(np.asarray(red_block[i][j].values()))[time][f])#f=50:100
	grouped_vis=np.asarray(grouped_vis)
	red_vec=np.c_[grouped_vis.real,grouped_vis.imag]#real and imag col
	vis_real_imag=red_vec.flatten()#respective real/img ordered vis
	return vis_real_imag,grouped_vis
np.save('bigdata_grouped_vis_single_freq.npy',group_vis_by_redundency(red_block)[1])
np.save('bigdata_real_imag_paired_vis_single_freq.npy',group_vis_by_redundency(red_block)[0])

def group_redundent_bline_xyz(ant_index,edge):
	bline=[]
	for i in range(len(ant_index)):
		for k in range(len(ant_index[i])/2):
			bline.append(aa.get_baseline(ant_index[i][2*k],ant_index[i][2*k+1]))
	bline=np.asarray(bline)
	red_bline_xyz=[]
	for i in range(len(edge)-1):
		red_bline_xyz.append(bline[edge[i]:edge[i+1]])
	red_bline_xyz=np.asarray(red_bline_xyz)
	return red_bline_xyz
	
def group_redundent_bline(red_bline_xyz,edge):
	bline_len=[]
	for i in range(len(red_bline_xyz)):
		for j in range (len(red_bline_xyz[i])):
			bline_len.append(np.sqrt(red_bline_xyz[i][j][0]**2+red_bline_xyz[i][j][1]**2+red_bline_xyz[i][j][2]**2))
	bline_len=np.asarray(bline_len)
	grouped_list_bl=[bline_len[edge[i]:edge[i+1]] for i in range(len(ant_index))] 
	return grouped_list_bl 

###################################
#freqs=np.linspace(0.117,0.182,num=131)[0:5]#aipy likes GHz units.





#calculate relevant map parameters
c = 3e8 #m/s
ipix = np.arange(npix)
theta,phi = hp.pix2ang(nside,ipix)
s=np.array(hp.pix2vec(nside,ipix))
tx=s[0];ty=s[1];tz=s[2]

#we care about scales ~21 degrees
lmax=3*nside - 1
l,m = hp.Alm.getlm(lmax)

#frequencies in Hz
nfreq=freqs.shape[0]
nu = np.outer(np.linspace(117e6,182e6,num=nfreq),np.ones(npix))#*u.Hz

#define sky -- completely arbitrary choice of temp
uniform_sky = np.ones(npix)*100.#*u.K

#completely arbitrary choice of noise level XXX UN-USED RIGHT NOW
noise = np.zeros(npix)
for i in range(npix): noise[i] = np.random.uniform(-100,100)#* u.K

def cov_mat(M):
	#M must be transposed in order to proceed
	M=np.asarray(M)
	I=np.ones(M.shape[0]).reshape(M.shape[0],1)
	d=I.dot(I.T)
	a=M-d.dot(M)/(M.shape[0])
	return np.dot(a.T,a.conj())/(M.shape[0]-1)

#Visibility covarience

def cov(m):
    #return n.cov(m)
    X = np.array(m, ndmin=2, dtype=np.complex)
    X -= X.mean(axis=1)[(slice(None),np.newaxis)]
    N = X.shape[1]
    fact = float(N - 1)
    return (np.dot(X, X.T.conj()) / fact).squeeze()	

#defining sky_power spectrum

def vis_cov_mat(edge,ant_index):
	#returns sky cov for a single freq channel
	n_blm=hp.map2alm(beam[0],lmax=2*nside,mmax=2*nside).shape[0]
	bline_comp=group_redundent_bline_xyz(ant_index,edge)
	b=np.concatenate(bline_comp)
	blx=b[:,0]/3.0;bly=b[:,1]/3.0;blz=b[:,2]/3.0
	Blm=np.zeros((blx.size,n_blm),dtype=np.complex)
	for i in range(Blm.shape[0]):
		Blm[i]=hp.map2alm(beam[0]*np.exp(-2j*freqs[0]*(blx[i]*tx+bly[i]*ty+blz[i]*tz)),lmax=2*nside,mmax=2*nside,iter=3)
	beam_cl=[]
	for i in range(edge.size-1):
		beam_cl.append(np.dot(Blm[edge[i]:edge[i+1]],np.conjugate(Blm[edge[i]:edge[i+1]]).T))
	beam_cl=np.asarray(beam_cl)
	
	vis_group=group_vis_by_redundency(red_block)[0]#real/imag separated
	vis=[]
	for i in range(edge.size-1):
		vis.append(vis_group[2*edge[i]:2*edge[i+1]])
	vis=np.asarray(vis)
	data_cov=[]
	for j in range(edge.size-1):
		data_cov.append(cov(vis[j]))
	data_cov=np.asarray(data_cov)
	sky_cl=[]
	for i in range(len(data_cov)):
		sky_cl.append(data_cov[i]/beam_cl[i])
	sky_cl=np.asarray(sky_cl)
	vis_cov=[]
	for i in range(edge.size-1):
		vis_cov.append(np.dot(sky_cl[i],beam_cl[i]))
	return vis_cov

viscov=vis_cov_mat(edge,ant_index)
np.save('vis_cov_mat_single_freq.npy',viscov)	
	
			
def vis_cov_mat_multi_freq(edge,ant_index):
	#returns sky cov for multi-frequency channel
	LMAX=8;MMAX=8
	n_blm=hp.map2alm(beam[0],lmax=LMAX,mmax=MMAX).shape[0]
	bline_comp=group_redundent_bline_xyz(ant_index,edge)
	b=np.concatenate(bline_comp)
	blx=b[:,0]/3.0;bly=b[:,1]/3.0;blz=b[:,2]/3.0
	BLM=[]
	for i in range(blx.size):
		for j in range(freqs.size):
			BLM.append(hp.map2alm(beam[j]*np.exp(-2j*np.pi*freqs[j]*np.pi*(blx[i]*tx+bly[i]*ty+blz[i]*tz)),lmax=LMAX))
	blm=np.asarray(BLM)
	lims=freqs.size*edge
	blm_grouped=[]
	for i in range(edge.shape[0]-1):
		blm_grouped.append(blm[lims[i]:lims[i+1]])
	blm_grouped=np.asarray(blm_grouped)
	reshaped_blm=[blm_grouped[i].reshape(freqs.size,num_unique_bl[i],n_blm) for i in range(edge.size-1)]
	#zz=np.asarray([reshaped_blm[i].shape for i in range(len(edge)-1)])
	#kk=[np.zeros((zz[i])) for i in range(len(edge)-1)]#grouped dot(blm,blm.T.conj())
	
	blm_times_blm_conj=[]
	for i in range(len(edge)-1):
		for j in range(freqs.size):
			blm_times_blm_conj.append(np.dot(reshaped_blm[i][j],reshaped_blm[i][j].T.conj()))
	blm_times_blm_conj=np.asarray(blm_times_blm_conj)
	freq_edges=freqs.size*np.arange(freqs.size*(edge.size-1))
	#group_by_freqs=[]
	#for i in range(freqs.size*(edge.size-1)):
		#group_by_freqs.append(blm_times_blm_conj[freq_edges[i]:freq_edges[i+1]])
	
		
		
	

	
	
	
	
	blm_transposed=[]
	for i in range(edge.shape[0]-1):
		blm_transposed.append(np.sum(reshaped_blm[i].T,axis=0))
	blm_transposed=np.asarray(blm_transposed)
	dot_blm=[]
	for i in range(edge.shape[0]-1):
		dot_blm.append(np.dot(blm_transposed[i],blm_transposed[i].T.conj()))
	dot_blm=np.asarray(dot_blm)
	
	vgroup=np.asarray(get_grouped_vis_for_all_channel(red_block)[0])
	vis_cov=[]
	for j in range(edge.shape[0]-1):
		vis_cov.append(cov(vgroup[j]))
	vis_cov=np.asarray(vis_cov)
	sky_pspec=[]
	for i in range(len(vis_cov)):
		sky_pspec.append(vis_cov[i]/dot_blm[i])
	sky_pspec=np.asarray(sky_pspec)
	vis_cov_multi_freq=[]
	for i in range(len(edge)-1):
		vis_cov_multi_freq.append(np.dot(sky_pspec[i],dot_blm[i]))
	return vis_cov_multi_freq
vis_cov_multi_freq=np.asarray(vis_cov_mat_multi_freq(edge,ant_index))
np.save('vis_cov_mat_multi_freq.npy',vis_cov_multi_freq)	

def eignvals_single_freq_vis_cov(edge):
	eig_val_sing=[]
	for i in range(edge.size-1):
		eig_val_sing.append(np.linalg.eig(viscov[i]))
	return np.asarray(eig_val_sing)
eigval_single=eignvals_single_freq_vis_cov(edge)
np.save('eigvals_eigvecs_single_freq.npy',eigval_single)

def eigvals_multifreq_vis_cov(edge):
	eig_val_mult=[]
	for i in range(edge.size-1):
		eig_val_mult.append(np.linalg.eig(vis_cov_multi_freq[i]))
	return np.asarray(eig_val_mult)
eigval_multi=eigvals_multifreq_vis_cov(edge)
np.save('eigvals_eigvecs_multi_freq.npy',eigval_multi)
'''






	








	
	
	
	
	
	
	
	
	 


