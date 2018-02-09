import aipy 
import capo as C
import sys
import numpy as np
import bm_prms 
import healpy as hp

u_old=np.load('u_old.npy')/3.0;v_old=np.load('v_old.npy')/3.0;w_old=np.load('w_old.npy')/3.0
nside=256
npix = hp.nside2npix(nside)
ipix=np.arange(npix)
lmax=3*nside - 1;l,m = hp.Alm.getlm(lmax)
theta,phi = hp.pix2ang(nside,ipix)
s=np.array(hp.pix2vec(nside,ipix))
tx=s[0];ty=s[1];tz=s[2]

freqs=np.linspace(0.117,0.182,num=203)#aipy likes GHz units.[0:4]
nfreq=freqs.size
nu = np.outer(np.linspace(117e6,182e6,num=nfreq),np.ones(npix))
c=3e8
def rotate_hmap(map,rot):
	npix = map.shape[0]
	nside = hp.npix2nside(npix)

	rotmap = np.zeros(npix)
	ipix = np.arange(npix)
	t,p = hp.pix2ang(nside,ipix)

	r = hp.Rotator(rot=rot)

	# For each pixel in the new map, find where it would have come 
	# from in the old    
	trot,prot = r(t,p)
	ipix_rot = hp.ang2pix(nside,trot,prot)

	rotmap = map[ipix_rot]

	return rotmap


beams = np.zeros((freqs.shape[0],npix))#selected only few frequency channels
for i, freq in enumerate(freqs):
	bm = bm_prms.prms['beam'](np.array([freq]),nside=nside,lmax=20,mmax=20,deg=7)
	bm.set_params(bm_prms.prms['bm_prms'])
	px = range(hp.nside2npix(nside))
	xyz = hp.pix2vec(nside,px)
	poly = np.array([h.map[px] for h in bm.hmap])
	Axx = np.polyval(poly,freq)
	Axx = np.where(xyz[-1] >= 0, Axx, 0)
	Axx /= Axx.max()
	Axx = Axx*Axx
	beams[i,:] = rotate_hmap(Axx,[21,120]) #[0,0]=north pole, [0,90]=equator, [21,120]=about right for PAPER
	
np.save('XX_beam_maps.npy',beams)
beam = beams
###calculating blm's at single frequency

blm=np.zeros((u_old.size,l.size),dtype='complex128')
for i in range(u_old.size):
	blm[i]=hp.map2alm(beam[100]*np.exp(-2j*np.pi*nu[100][0]/c*(u_old[i]*tx+v_old[i]*ty+w_old[i]*tz)),lmax=lmax)
np.save('blm_freq150.npy',blm)	
	








	
