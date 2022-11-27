#!/usr/bin/env python
# coding: utf-8

# In[1]:

import mcfost
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterExponent
from numpy import unravel_index
import astropy.units as unit
import pysynphot
import matplotlib.ticker as mticker
import os
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import time
import random
import emcee
from multiprocessing import Pool
import signal
import shutil
from pathlib import Path
Path("/tmp/mcmc").mkdir(parents=True, exist_ok=True)


# In[2]:


#obs_data
Wise_wav=np.array([3.4, 4.6, 12, 22])
Wise_flux=np.array([56.7436,47.7518,107.821,83.6299])
w_err=np.array([1.18940,0.914726,1.57726,2.12915])
Alma_flux=np.array([41.6*10**-3])
Alma_wav=1260
a_err=np.array([8*10**-3])


# In[3]:


#record best
best_sol_dir='best_sol.npy'
best_sol = np.array([-3.28664347e+02,4.44964423e+01,1.82190546e+00,-1.45841990e+00,6.46660290e-02,3.50066859e+00,-1.04048118e+01,2.26889763e-01])
np.save(best_sol_dir, best_sol)


# In[4]:


def lnlike(theta, original=False):
    inc, flar_exp, surfd_exp, r_in, dr, log_dust_mass, scale_height = theta
    #read paramfile
    file_dir='RZ_psc/'
    par = mcfost.Paramfile(file_dir+'RZ_psc.para')
    #change parameters
    if not original:
        par.RT_imin = inc #40~90
        par.RT_imax = inc #40~90
        par.density_zones[0]['flaring_exp'] = flar_exp #1.2~2.0
        par.density_zones[0]['surface_density_exp'] = surfd_exp #-2.5~0.0
        par.density_zones[0]['r_in'] = r_in #0~2
        par.density_zones[0]['r_out'] = r_in+dr #r_in+0.1~5.0
        par.density_zones[0]['dust_mass'] = 10**log_dust_mass #-10.5 -7.5
        par.density_zones[0]['scale_height'] = scale_height #0.01~1.5
    
    #make_dir to write new paramfile
    file_dir = '/tmp/'
    rint = str(random.randint(0,999999999))
    os.mkdir(file_dir+'mcmc/'+rint)
    file_dir = file_dir+'mcmc/'+rint+'/'
    par.writeto(file_dir+'RZ_psc.para', log_show=False)
    par = mcfost.Paramfile(file_dir+'RZ_psc.para')
    
    #run mcfost
    mcfost.run_one_file(file_dir+'RZ_psc.para', wavelengths=[0.575], move_to_subdir=False, log_show=False, timeout = 600)

    #IR Excess
    filename=file_dir+'data_th/sed_rt.fits.gz'
    os.system("gunzip {}".format(filename))
    

    try:sed_model=fits.open(file_dir+'data_th/sed_rt.fits')
    except:return -np.inf
    wav=sed_model[1].data

    #total result
    modelt_flux=sed_model[0].data[0]
    modeltotal = pysynphot.ArraySpectrum(wav, modelt_flux.flatten(), name='total-model',fluxunits='flam',waveunits='microns')
    c=2.99792*10**14
    mt_jy=10**26*modeltotal.flux*modeltotal.wave*1000/c

    #likelihood
    mt_jy_cs = CubicSpline(wav, mt_jy)
    try:output = sum(norm.logsf((np.abs(Wise_flux-mt_jy_cs(Wise_wav))/w_err)))+sum(norm.logsf((np.abs(Alma_flux-mt_jy_cs(Alma_wav))/a_err)))
    except:return -np.inf
    shutil.rmtree(file_dir, ignore_errors=True)
    global best_sol_dir
    best_sol = np.load(best_sol_dir,)
    if output>best_sol[0]:
        best_sol[0]=output
        best_sol[1:8]=theta
        print(best_sol)
        np.save(best_sol_dir, best_sol)
    return(output)


# In[5]:


def lnprior(theta):
    inc, flar_exp, surfd_exp, r_in, dr, log_dust_mass, scale_height = theta
    if 40.0 < inc < 90.0 and 1.2 < flar_exp < 2.0 and -2.5 < surfd_exp < 0.0 and 0.0 < r_in < 2.0 and 0.1 < dr < 5.0 and -10.5 < log_dust_mass < -7.5 and 0.01 < scale_height < 1.5:
        return 0
    return -np.inf


# In[6]:


def lnprob(theta, original=False):
    if original:
        return lnlike(theta, original=True)       
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)


# In[10]:


#random position
pos = np.random.rand(256, 7)*[50,0.8,2.5,2.0,4.9,3,1.49]+np.ones((256, 7))*[40,1.2,-2.5,0,0.1,-10.5,0.01]
nwalkers, ndim = pos.shape
# Don't forget to clear it in case the file already exists
filename = "Mcfost_emcee_py.h5"
mcfost_backend = emcee.backends.HDFBackend(filename, name="Mcfost_emcee")
#mcfost_backend.reset(nwalkers, ndim)


# In[ ]:


#run mcmc
import os
os.environ["OMP_NUM_THREADS"] = "32"
steps = 666
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=mcfost_backend)
    sampler.run_mcmc(pos, steps, progress=True,**{'skip_initial_state_check':True})


# In[9]:


fig, axes = plt.subplots(7, figsize=(10, 20), sharex=True)
samples = sampler.get_chain()
labels = ['inc','flar_exp','surf_den_exp','r_in','dr','log_dustm','scale_height']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
# Save the full figure...
fig.savefig('full_figure.png', bbox_inches='tight')


# In[10]:


from IPython.display import display, Math
flat_samples = sampler.get_chain(discard=50, thin=1, flat=True)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))


# In[11]:


#print best
print(best_sol)

