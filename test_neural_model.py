from neural_model import *
import matplotlib.pyplot as plt

def test_rk4():
	h = 0.005
	NT = 200

	t = 0
	y = np.ones((3,1))
	for i in range(NT):
		y,t = rk4(y,t,h,rhs_lin)

	y_exact = np.reshape(np.array([np.exp(1), np.exp(2), np.exp(3)]), (3,1))
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
	ret_y.append(2*y[1])
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
	
