#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
from aml_storage import Labels, Block, Descriptor
from utils.builder import DescriptorBuilder
import ase.io
from itertools import product
from utils.clebsh_gordan import ClebschGordanReal
from utils.hamiltonians import fix_pyscf_l1, dense_to_blocks, blocks_to_dense, couple_blocks, decouple_blocks
import matplotlib.pyplot as plt
from utils.librascal import  RascalSphericalExpansion, RascalPairExpansion

import numpy as np
# In[2]:


frames = ase.io.read("data/water-hamiltonian/water_coords_1000.xyz",":10")
for f in frames:
    f.cell = [100,100,100]
    f.positions += 50


# In[3]:


jorbs = json.load(open('data/water-hamiltonian/orbs_def2_water.json', "r"))
orbs = {}
zdic = {"O" : 8, "H":1}
for k in jorbs:
    orbs[zdic[k]] = jorbs[k]


# In[4]:


hams = np.load("data/water-hamiltonian/water_fock.npy", allow_pickle=True)
for i, f in enumerate(frames):
    hams[i] = fix_pyscf_l1(hams[i], f, orbs)


# In[5]:


cg = ClebschGordanReal(4)


# In[6]:


rascal_hypers = {
    "interaction_cutoff": 3.5,
    "cutoff_smooth_width": 0.5,
    "max_radial": 3,
    "max_angular": 2,
    "gaussian_sigma_type": "Constant",
    "compute_gradients":  False,
}


# In[7]:


frames[0].numbers[0]


# In[8]:


spex = RascalSphericalExpansion(rascal_hypers)
rhoi = spex.compute(frames)


# In[9]:


from rascal.representations import SphericalExpansion


# In[10]:

pairs = RascalPairExpansion(rascal_hypers)
gij = pairs.compute(frames)


# In[12]:


gij.sparse


# In[13]:


new_sparse = Labels(
    names=["sigma", "L", "nu"],
    values=np.array(
        [
            [1, l, 0]
            for l in range(rascal_hypers["max_angular"] + 1)                    
        ],
        dtype=np.int32,
            ),
        )


# In[14]:


new_sparse


# In[15]:


blocks=[]
for i , block in gij:
    print(i, block)
    blocks.append(block)


# In[ ]:


aa=Descriptor(new_sparse, blocks)

input()
