#####################################
# DEFINITION OF CLASSES TO DESCRIBE 
#   -HODGKIN-HUXLEY CURRENTS
#   -Neurons 
#   -Synapses
####################################

##########
#CURRENTS
##########

import numpy as np

#Sodium Potassium Current
class INaK: 
	def __init__(self, V0):
		self.iNa = 0
		self.iK = 0

		#Voltage relative v set points
		self.V0 = V0
		self.Vtr = -50

		#Conductance
		self.G_K = 10
		self.G_Na = 100

		#Reversal
		self.E_K = -95
		self.E_Na = 50

		#Temp. dependence
		self.Phi = pow(3,((22-36)/10))

	def set_initials(self):
		v2 = self.V0 + 50
		Alpha1 = 0.32*(13-v2)/(np.exp((13-v2)/4)-1)
		Beta1 = 0.28*(v2-40)/(np.exp((v2-40)/5)-1)
		m0 = Alpha1/(Alpha1+Beta1)

		Alpha2 = 0.128*np.exp((17-v2)/18)
		Beta2 = 4/(np.exp((40 - v2)/5) + 1)
		h0 = Alpha2/(Alpha2+Beta2)

		Alpha3 = 0.02*(15-v2)/(np.exp((15-v2)/5)-1)
		Beta3 = 0.5*np.exp((10-v2)/40)
		n0 = Alpha3/(Alpha3+Beta3)

		return m0, h0, n0

	def calc(self,v,m,h,n,t):
		v2 = v-self.Vtr

		#Sodium
		self.iNa = self.G_Na*m*m*m*h*(v-self.E_Na)

		Alpha1 = 0.32*(13-v2)/(np.exp((13-v2)/4)-1)
		Beta1 = 0.28*(v2-40)/(np.exp((v2-40)/5)-1)
		tau_m = 1/(Alpha1+Beta1) / self.Phi
		m_inf = Alpha1/(Alpha1 + Beta1)

		Alpha2 = 0.128*np.exp((17-v2)/18)
		Beta2 = 4/(np.exp((40 - v2)/5) + 1)
		tau_h = 1/(Alpha2+Beta2) / self.Phi
		h_inf = Alpha2/(Alpha2 + Beta2)

		dm = -(m-m_inf)/tau_m
		dh = -(h-h_inf)/tau_h

		#Potassium
		self.iK = self.G_K*n*n*n*n*(v-self.E_K)

		Alpha3 = 0.02*(15-v2)/(np.exp((15-v2)/5)-1)
		Beta3 = 0.5*np.exp((10-v2)/40)
		tau_n = 1/(Alpha3+Beta3) / self.Phi
		n_inf = Alpha3/(Alpha3 + Beta3)

		dn = -(n-n_inf)/tau_n
		return dm, dh, dn

# Potassium A current
class I_Pot_A:
	def __init__(self, V0):
		self.iA = 0
		self.V0 = V0
		self.G_A = 10
		self.E_K = -95
		self.Phi = pow(3,((36-23.5)/10))

	def set_initials(self):
		m0 = 1/(1+np.exp(-(self.V0+60)/8.5))
		h0 = 1/(1+np.exp(-(self.V0+78)/6))

		return m0, h0

	def calc(self,v,m,h,n,t):
		self.iA = self.G_A*m*m*m*m*h*(v-self.E_K)

		tau_m = (1/(np.exp((v+35.82)/19.69)+np.exp(-(v+79.69)/12.7)) + 0.37)/self.Phi
		m_inf = 1/(1+np.exp(-(v+60)/8.5))

		tau_h = (1/((np.exp(((v+46.05)/5)+np.exp(-(v+238.4)/37.45)))))/self.Phi
		if (v>=-63):
			tau_h = 19/self.Phi

		h_inf = 1/(1+exp((v+78)/6))

		dm = -(m-m_inf)/tau_m
		dh = -(h-h_inf)/tau_h

		return dm, dh


#########
#Cells
#########
class Cell:
	def __init__(self, V0=0,DC=0):
		self.E_l = -50
		self.G_l = 0.15
		self.G_kl = 0.02
		self.DC = DC
		self.S = 1.43e-4
		self.syn_idxs = []

		self.I_Na_K = INaK(V0=V0) 
		m0, h0, n0 = self.I_Na_K.set_initials()

		#self.I_KA = I_Pot_A(V0=V0)
		self.y0 = [V0, m0, h0, n0]
	
	def get_initial(self):
		return self.y0
	
	def add_syn_idx(self,idx):
		self.syn_idxs.append(idx)

	def calc(self,y,t,I_syn):
		dm, dh, dn = self.I_Na_K.calc(y[0], y[1], y[2], y[3], t)
		iNa = self.I_Na_K.iNa
		iK = self.I_Na_K.iK
		E_K = self.I_Na_K.E_K

		dV = -self.G_l*(y[0] - self.E_l) - iNa - iK - self.G_kl*(y[0] - E_K) + self.DC/self.S +I_syn 
		return dV, dm, dh, dn


# Inhibitory Cell - Same as Cell
class Inh_Cell(Cell):
	def __init__(self, V0=0, DC=0):
		self.E_l = -50
		self.G_l = 0.15
		self.G_kl = 0.02
		self.DC = DC
		self.S = 1.43e-4
		self.syn_idxs = []

		self.I_Na_K = INaK(V0=V0) 
		m0, h0, n0 = self.I_Na_K.set_initials()
		self.y0 = [V0, m0, h0, n0]

# Excitatory Cell - Similar to Cell
class Exc_Cell(Cell):
	def __init__(self, V0=0, DC=0):
		self.E_l = -70
		self.G_l = 0.01
		self.G_kl = 0.012
		self.DC = DC #0.001
		self.S = 1.43e-4
		self.syn_idxs = []

		self.I_Na_K = INaK(V0) 
		self.I_Na_K.G_Na = 90
		self.I_Na_K.G_K = 10

		m0, h0, n0 = self.I_Na_K.set_initials()
		self.y0 = [V0, m0, h0, n0]

################
# Synapses
###############

# Synapse - main class - covers first order kinetic model.
# NOT used in ode solver

class synapse:
	def __init__(self, pre_idx, post_idx, g_MAX):
		self.g_MAX = g_MAX
		self.E_rev = -70
		self.Alpha = 10.5
		self.Beta = 0.166
		self.R=0; self.R0=0; self.R1=0
		self.C=0; self.Cmax = 0.5; self.Cdur = 0.3
		self.lastrelease = -100
		self.Deadtime = 1
		self.Prethresh = -20
		self.R_inf = self.Cmax*self.Alpha / (self.Cmax*self.Alpha + self.Beta)
		self.R_tau = 1/ (self.Alpha*self.Cmax + self.Beta)

		self.pre_idx = pre_idx; self.post_idx = post_idx
		self.I = 0

	def calc(self, t, v_pre, v_post):
		q = t-(self.lastrelease+self.Cdur)
		if (q > self.Deadtime):
			if (v_post > self.Prethresh):
				self.C = self.Cmax
				self.R0 = self.R
				self.lastrelease = t
		elif (q < 0):
			pass
		elif (self.C==self.Cmax):
			self.R1=self.R
			self.C=0
		if (self.C>0):
			self.R = self.R_inf + (self.R0 - self.R_inf)*np.exp(-(t-self.lastrelease)/self.R_tau)
		else:
			self.R = self.R1*np.exp(-self.Beta * (t-(self.lastrelease+self.Cdur)))

		self.I = self.g_MAX * self.R * (v_pre - self.E_rev)

#AMPA Synapse
class AMPA(synapse):
	def __init__(self, pre_idx, post_idx,g_MAX):
		self.g_MAX = g_MAX
		self.E_rev = 0
		self.Alpha = 1
		self.Beta = 0.2
		self.R=0; self.R0=0; self.R1=0
		self.C=0; self.Cmax = 0.5; self.Cdur = 0.3
		self.lastrelease = -100
		self.Deadtime = 1
		self.Prethresh = 0
		self.R_inf = self.Cmax*self.Alpha / (self.Cmax*self.Alpha + self.Beta)
		self.R_tau = 1/ (self.Alpha*self.Cmax + self.Beta)

		self.pre_idx = pre_idx; self.post_idx = post_idx
		self.I = 0

#GABA Synapse
class GABA(synapse):
	def __init__(self, pre_idx, post_idx,g_MAX):
		self.g_MAX = g_MAX
		self.E_rev = -70
		self.Alpha = 10.5
		self.Beta = 0.166
		self.R=0; self.R0=0; self.R1=0
		self.C=0; self.Cmax = 0.5; self.Cdur = 0.3
		self.lastrelease = -100
		self.Deadtime = 1
		self.Prethresh = -20
		self.R_inf = self.Cmax*self.Alpha / (self.Cmax*self.Alpha + self.Beta)
		self.R_tau = 1/ (self.Alpha*self.Cmax + self.Beta)

		self.pre_idx = pre_idx; self.post_idx = post_idx
		self.I = 0


#Input "Synapse"

class InPulse:
	def __init__(self, g_MAX, t_on, t_off):
		self.g_MAX = g_MAX
		self.t_on = t_on
		self.t_off = t_off
		self.tau_on = 150
		self.tau_off = 150
		self.R = 0
		self.pre_idx = -1
		self.post_idx = -1
		self.I = 0
		
	def calc(self,t,v_pre,v_post):
		if (t>self.t_on and t<self.t_off):
			self.I = self.g_MAX*(1 - np.exp(-(t-self.t_on)/self.tau_on))
			R = self.I
		elif (t>=self.t_off):
			self.I = self.R*np.exp(-(t-self.t_off)/self.tau_off)
		else:
			self.I = 0
		

###############
# RK4
###############
def rk4(y, t, h, fun):
	k1 = fun(y,t)
	k2 = fun(y+(h/2)*k1,t+(h/2)) 	
	k3 = fun(y+(h/2)*k2,t+(h/2)) 	
	k4 = fun(y+h*k3,t+h) 	

	y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
	t = t+h
	return y,t




