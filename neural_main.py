### MAIN CODE - to be used with neural_model.py
import numpy as np
import matplotlib.pyplot as plt
from neural_model import *

NECells = 1
NICells = 1

np.random.seed(0)
rDC = 1.5e-4*np.random.rand(NECells+NICells, 1).flatten()
ENs = [Exc_Cell(V0=-70, DC0=rDC[i]) for i in range(NECells)]
INs = [Exc_Cell(V0=-70, DC0=rDC[i]) for i in range(NICells)]
#INs = [Inh_Cell(V0=-61) for_ in range(NICells)]
y_sv = []
t_sv = []
synapses = []

g_AMPA = 0; g_GABA = 0; g_Inp = 1e-1

#TODO -variability; 


def rhs(y,t):
	dy = []
	for syn in synapses:
		#TODO - Need to do deal with the input case
		i_pre = syn.pre_idx; i_post = syn.post_idx
		if ((i_pre is not -1) and (i_post is not -1)):
			v_pre = y[i_pre][0]; v_post = y[i_post][0]
		else:
			v_pre = -1; v_post = -1
		syn.calc(t,v_pre,v_post)

	for i in range(NECells):
		I_syn = sum([synapses[j].I for j in ENs[i].syn_idxs])
		dy.append(ENs[i].calc(y[i],t,I_syn))
	for i in range(NICells):
		I_syn = sum([synapses[j].I for j in INs[i].syn_idxs])
		dy.append(INs[i].calc(y[i],t,I_syn))
		
	return np.array(dy)


def main():
	
	h = 0.05		#MS
	NT = 40000		#2s
	
	
	
	# INITIALIZE VARS
	# Y contains all of the current states of all the
	# dynamic variables (voltages, m,h,n).
	y = [EN.get_initial() for EN in ENs]
	[y.append(IN.get_initial()) for IN in INs]
	y = np.array(y)



	#CREATE SYNAPSES
	p_ENIN = 0.0; p_INEN = 0.0; p_ININ = 0.0; p_EN = 1.0; p_IN = 0.0

	C_ENIN = np.random.rand(NECells,NICells)<p_ENIN
	n_AMPA = sum(sum(C_ENIN))
	pre_AMPA, post_AMPA = np.where(C_ENIN)
	for i in range(n_AMPA):
		synapses.append(AMPA(pre_AMPA[i], post_AMPA[i], g_AMPA))
		INs[post_AMPA[i]].add_syn_idx(i)

	C_INEN = np.random.rand(NICells,NECells)<p_INEN
	n_GABA1 = sum(sum(C_INEN))
	pre_GABA, post_GABA = np.where(C_INEN)
	for i in range(n_GABA1):
		synapses.append(GABA(pre_GABA[i], post_GABA[i], g_GABA))
		ENs[post_GABA[i]].add_syn_idx(len(synapses)-1)

	C_ININ = np.random.rand(NICells,NICells)<p_ININ
	n_GABA2 = sum(sum(C_ININ))
	pre_GABA, post_GABA = np.where(C_ININ)
	for i in range(n_GABA2):
		synapses.append(GABA(pre_GABA[i], post_GABA[i], g_GABA))
		INs[post_GABA[i]].add_syn_idx(len(synapses)-1)

	C_IN = np.random.rand(1,NICells)<p_IN
	n_InpIN = sum(sum(C_IN))
	jnk, post_Inp = np.where(C_IN)
	synapses.append(InPulse(g_Inp))
	for i in range(n_InpIN):
		INs[post_Inp[i]].add_syn_idx(len(synapses)-1)

	C_EN = np.random.rand(1,NECells)<p_EN
	n_InpEN = sum(sum(C_EN))
	jnk, post_Inp = np.where(C_EN)
	for i in range(n_InpEN):
		ENs[post_Inp[i]].add_syn_idx(len(synapses)-1)


	print('Synapses on EN[0]', ENs[0].syn_idxs)
	print('Synapses on IN[0]', INs[0].syn_idxs)
	#MAIN CALCULATION 
	t = 0
	for i in range(0,NT):
		if (i%2000==0):
			print('t = ',t)
		#SAVE y
		y_sv.append(y.T[0])
		t_sv.append(t)
		#UPDATE y
		y,t = rk4(y,t,h,rhs)

if __name__ is '__main__':
	main()


