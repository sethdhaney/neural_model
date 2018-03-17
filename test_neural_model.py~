from neural_model import *
import matplotlib.pyplot as plt

def test_rk4():
	h = 0.005
	TF = 1
	NT = int(np.ceil(TF/h))

	t = 0
	y = np.ones((3,1))
	for i in range(NT):
		y,t = rk4(y,t,h,rhs_lin)

	y_ex_2 = ((5/4)*np.exp(5))/(1+(1/4)*np.exp(5))
	y_exact = np.reshape(np.array([np.exp(1), y_ex_2, np.exp(3)]), (3,1))
	error = np.sqrt(np.sum((y-y_exact)*(y-y_exact)))
	TOL = h*h*h*h
	if (error>TOL):
		print('RK4 test FAILED')
		print('Error = ', error)
		print('Expected error below ', TOL) 
		print('Expected y_exact', y_exact)
		print('Got y', y)


def rhs_lin(y,t):
	ret_y = []
	ret_y.append(y[0])
	ret_y.append(y[1]*(5-y[1]))
	ret_y.append(3*y[2])
	return np.array(ret_y)

def test_InPulse():
	inpul = InPulse(100)
	h = 0.05
	NT = 40000
	t = 0
	i_sv = []; t_sv = []
	for i in range(NT):
		#i_sv.append(inpul.calc(t,0,0))
		inpul.calc(t,0,0)
		i_sv.append(inpul.I)
		t_sv.append(t)
		t = t+h
	

	plt.figure()
	plt.plot(i_sv)
	
def test_synapse():
	g_MAX = 1e-1
	a = AMPA(-1,-1,g_MAX)
	h = 0.05
	TF = 100
	NT = int(np.ceil(TF/h))

	t = 0; t_on = 50; t_off = 75
	I_sv = []; t_sv = []
	for i in range(NT):
		if (t>t_on and t<t_off):
			v_pre = 50; v_post = 50
		else:
			v_pre = a.E_rev; v_post = -100

		a.calc(t,v_pre,v_post)
		I_sv.append(a.I)
		t_sv.append(t)
		t = t+h

	return t_sv, I_sv


