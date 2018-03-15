### MAIN CODE - to be used with neural_model.py
import numpy as np
import matplotlib.pyplot as plt
import neural_model as nm
from importlib import reload
from scipy.integrate import RK45

NECells = 1
NICells = 1

np.random.seed(1)
rDC = 5e-5*np.random.rand(NECells+NICells, 1).flatten()
DC0 = 7e-5
ENs = [nm.Exc_Cell(V0=-70, DC=DC0+rDC[i]) for i in range(NECells)]
INs = [nm.Exc_Cell(V0=-70, DC=DC0+rDC[NECells+i]) for i in range(NICells)]
#INs = [nm.Inh_Cell(V0=-61) for _ in range(NICells)]

#Global variables 
# y		current state of all dynamic variables (v1,m1,h1,n1,v2,...) size NEQN,1
# y_sv 		voltages only over time
# t_sv		t over time
# cell_idxs	This is a list of length NCELL where each entry contains the indicies of
#			dynamic variables associated with each cell. This is critical to feed
#			during Cell.calc(y[cell_idxs[i]],...)
# synapses	list of synapse variables	
y = []; y_sv = []; t_sv = []; cell_idxs = []; synapses = []

g_AMPA = 1e-1; g_GABA = 100; g_Inp = 1e-1


def rhs(y,t):
	dy = []
	for syn in synapses:
		i_pre = syn.pre_idx; i_post = syn.post_idx
		if ((i_pre is not -1) and (i_post is not -1)):
			v_pre = y[i_pre]; v_post = y[i_post]
		else:
			v_pre = -1; v_post = -1
		syn.calc(t,v_pre,v_post)

	for i in range(NECells):
		I_syn = sum([synapses[j].I for j in ENs[i].syn_idxs])
		dy +=ENs[i].calc(y[cell_idxs[i]],t,I_syn)
	for i in range(NICells):
		I_syn = sum([synapses[j].I for j in INs[i].syn_idxs])
		dy += INs[i].calc(y[cell_idxs[NECells+i]],t,I_syn)
		
		
	return np.array(dy)


def main():
	
	h = 0.05		#MS
	TF = 2000
	NT = int(np.ceil(TF/h))
	
	
	
	# INITIALIZE VARS
	# Y contains all of the current states of all the
	# dynamic variables (voltages, m,h,n).
	idx = 0
	global y
	global cell_idxs
	for i in range(len(ENs)):
		y0 = ENs[i].get_initial()
		y+=y0
		cell_idxs.append(idx + np.array(range(len(y0))))
		idx = idx + len(y0)
	for i in range(len(INs)):
		y0 = INs[i].get_initial()
		y+=y0
		cell_idxs.append(idx + np.array(range(len(y0))))
		idx = idx + len(y0)
	y = np.array(y)
	cell_idxs = np.array(cell_idxs)



	#####################################
	#CREATE SYNAPSES
	#####################################
	p_ENIN = 1.0; p_INEN = 0.0; p_ININ = 0.0; p_EN = 1.0; p_IN = 0.0

	#AMPA EN to IN
	C_ENIN = np.random.rand(NECells,NICells)<p_ENIN
	if (len(C_ENIN)>0):
		n_AMPA = sum(sum(C_ENIN))
		pre_AMPA, post_AMPA = np.where(C_ENIN)
		for i in range(n_AMPA):
			synapses.append(nm.AMPA(cell_idxs[pre_AMPA[i]][0], 
				cell_idxs[post_AMPA[i]][0], g_AMPA))
			INs[post_AMPA[i]].add_syn_idx(i)

	#GABA IN to EN
	C_INEN = np.random.rand(NICells,NECells)<p_INEN
	if (len(C_INEN)>0):
		n_GABA1 = sum(sum(C_INEN))
		pre_GABA, post_GABA = np.where(C_INEN)
		for i in range(n_GABA1):
			synapses.append(nm.GABA(cell_idxs[pre_GABA[i]][0], 
				cell_idxs[post_GABA[i]][0], g_GABA))
			ENs[post_GABA[i]].add_syn_idx(len(synapses)-1)

	#GABA IN to IN
	C_ININ = np.random.rand(NICells,NICells)<p_ININ
	if (len(C_ININ)>0):
		n_GABA2 = sum(sum(C_ININ))
		pre_GABA, post_GABA = np.where(C_ININ)
		for i in range(n_GABA2):
			synapses.append(nm.GABA(cell_idxs[pre_GABA[i]][0], 
				cell_idxs[post_GABA[i]][0], g_GABA))
			INs[post_GABA[i]].add_syn_idx(len(synapses)-1)

	#Input to IN - note there is only one input "synapse" that all cells listen to
	t_on = 500; t_off = 1500
	C_IN = np.random.rand(1,NICells)<p_IN
	if (len(C_IN)>0):
		n_InpIN = sum(sum(C_IN))
		jnk, post_Inp = np.where(C_IN)
		synapses.append(nm.InPulse(g_Inp, t_on, t_off))
		for i in range(n_InpIN):
			INs[post_Inp[i]].add_syn_idx(len(synapses)-1)
			

	#Input to EN
	C_EN = np.random.rand(1,NECells)<p_EN
	if (len(C_EN)>0):
		n_InpEN = sum(sum(C_EN))
		jnk, post_Inp = np.where(C_EN)
		synapses.append(nm.InPulse(g_Inp, t_on, t_off))
		for i in range(n_InpEN):
			ENs[post_Inp[i]].add_syn_idx(len(synapses)-1)


	if NECells>0:
		print('Synapses on EN[0]', ENs[0].syn_idxs)
	if NICells>0:
		print('Synapses on IN[0]', INs[0].syn_idxs)






	#####################################
	#MAIN CALCULATION 
	#####################################
	t = 0
	for i in range(0,NT):
		if (i%2000==0):
			print('t = ',t)
		#SAVE y
		#y_sv.append(y)
		y_sv.append(y[cell_idxs.T[0]])
		t_sv.append(t)
		#UPDATE y
		y,t = nm.rk4(y,t,h,rhs)

if __name__ is '__main__':
	main()


