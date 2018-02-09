
import aipy; import capo as C;import numpy as np
import re;import glob

pol = 'yy'

CALFILE0='psa6240_v000'
CALFILE1='psa6240_v001'
CALFILE2='psa6240_v002'
CALFILE3='psa6240_v003'
CALFILE4='psa6240_v004'
CALFILE_psa128='psa6622_v003'#psa128 calfile

multi_filename=glob.glob("*.uvcRREcACO")[0:6]#Calibrated paper64
len_file=len(multi_filename)
np.save('multifile_length.npy',len_file)
filename=multi_filename[0]
#filename='zen.2456242.63314.uvcRREcACO'
####################################################


uv=aipy.miriad.UV('zen.2456242.17382.uvcRREcACO')
aa = aipy.cal.get_aa(CALFILE4,uv['sdf'],uv['sfreq'],uv['nchan'])
freqs = aa.get_afreqs()#aipy likes GHz units. avoiding band edges
nant=len(aa)
nbl=int(nant*(nant-1)/2.)
antlayout=aa.ant_layout
antpos = [a.pos for a in aa] #nanoseconds
antpos = np.array(antpos) #antpos needs to be (#ant, 3) array with positions
antpos = antpos*aipy.const.len_ns/100. #meters
np.save('antpos.npy',antpos)
np.save('antlayout.npy',antlayout)

sep2ij_zsa,blconj_zsa,bl2sep_zsa=C.zsa.grid2ij(aa.ant_layout)


R=[]#redundunt unique baselines index
for i in range(len(sep2ij_zsa.keys())):
	R.append(sep2ij_zsa.values()[i])
np.save('R.npy',R)

ant_red=[re.sub('_',(','),R[i]) for i in range(len(R))]

ant_index=[]
for j in range(len(R)):
	ant=[i.strip() for i in ant_red[j].split(',')] 
	ant_index.append(np.asarray(ant,dtype=int))
ant=np.concatenate(np.asarray(ant_index))
ant1,ant2=ant[::2],ant[1::2]
np.save('ant1.npy',ant1);np.save('ant2.npy',ant2);np.save('ant_index.npy',ant_index)

########################
red_block_single_file=[]##Visibility grouped by redundency for all freqs and time
for j in np.arange(len(sep2ij_zsa)):
	red_block_single_file.append(np.asarray(C.arp.get_dict_of_uv_data([filename],R[j],'yy')[1].values()))
red_block_single_file=np.asarray(red_block_single_file)
np.save('red_block_single_file.npy',red_block_single_file)

multi_file_red_block=[]
for i in range(len(multi_filename)):
	for j in range(110):
		multi_file_red_block.append(np.asarray(C.arp.get_dict_of_uv_data([multi_filename[i]],R[j],'yy')[1].values()))
multi_file_red_block=np.asarray(multi_file_red_block)
np.save('multi_file_red_block.npy',multi_file_red_block)

##############################


num_unique_bl=np.zeros(len(red_block_single_file),dtype=np.int)
for i in range(len(red_block_single_file)):
	num_unique_bl[i]=len(red_block_single_file[i])
edge=[]
total=0	
for x in num_unique_bl:
	total +=x
	edge.append(total)
edge=np.asarray(edge)
edge=np.concatenate((np.array([0]),edge))
nblock=len(edge)-1
np.save('nblock.npy',nblock);np.save('edge.npy',edge);np.save('num_unique_bl.npy',num_unique_bl)

########################################
#The following piece of script don't containing info of baselines
#derived from bad antennas, therefore,  not updated based on len of bad baselines
old_num_unique_bl=np.zeros(len(red_block_single_file),dtype=np.int)
for i in range(len(red_block_single_file)):
	old_num_unique_bl[i]=int(len(ant_index[i])/2.0)
old_edges=[]
total=0	
for x in old_num_unique_bl:
	total +=x
	old_edges.append(total)
old_edges=np.asarray(old_edges)
old_edges=np.concatenate((np.array([0]),old_edges))
np.save('old_edges.npy',old_edges);np.save('old_num_unique_bl.npy',old_num_unique_bl)
#########################################
def old_group_redundent_bline_xyz(ant_index,old_edges):
	bline=[]
	for i in range(len(ant_index)):
		for k in range(len(ant_index[i])/2):
			bline.append(aa.get_baseline(ant_index[i][2*k],ant_index[i][2*k+1]))
	bline=np.asarray(bline)
	red_bline_xyz=[]
	for i in range(len(old_edges)-1):
		red_bline_xyz.append(bline[old_edges[i]:old_edges[i+1]])
	red_bline_xyz=np.asarray(red_bline_xyz)
	return red_bline_xyz
bline_xyz_old=np.concatenate(old_group_redundent_bline_xyz(ant_index,old_edges))
blx_old=bline_xyz_old[:,0];bly_old=bline_xyz_old[:,1];blz_old=bline_xyz_old[:,2]
u_old=bline_xyz_old[:,0];v_old=bline_xyz_old[:,1];w_old=bline_xyz_old[:,2]
u_old=np.save('u_old.npy',u_old);v_old=np.save('v_old.npy',v_old);w_old=np.save('w_old.npy',w_old)
bline=np.sqrt(blx_old**2+bly_old**2+blz_old**2)
np.save('bline.npy',bline)
########################################



