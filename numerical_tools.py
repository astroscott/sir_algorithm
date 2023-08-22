import numpy as np

# Discrete integration methods:
def fe(u, du, t, dt, *args): # Forward Euler
    # where 
    # u  = state of the system
    # du = state derivative
    # dt = time increment
    return u + du(t, u, *args)*dt

# def be(u, du, dt, J, *args, tol=0.001, iter=1000): # backwards euler (WIP)
    # # where u is the state of the system
    # # du is the state derivative
    # # dt is the time step
    # # J is the jacobian of du
    
    # # take a forward euler step for the initial guess:
    # x = np.array([-1,5,7]).reshape(3,1)

    # for i in range(iter):
    #     print(i)
    #     print(x - np.matmul(np.linalg.inv(J(x, dt, *args)),du(x, *args)))
    #     if (x-u-dt*du(x, *args)).all() < tol:
    #         return x
    #     x = x - np.matmul(np.linalg.inv(J(x, dt, *args)),du(x, *args))
        

def rk2(u, du, t, dt, *args): # Runge-Kutta 2
    k = u + du(t, u, *args)*(dt/2)
    return u + du(t, k, *args)*dt

def rk4(u, du, t, dt, *args): # Runge-Kutta 4
    k1 = du(t, u, *args)*dt
    k2 = du(t, u + k1/2, *args)*dt
    k3 = du(t, u + k2/2, *args)*dt
    k4 = du(t, u + k3, *args)*dt
    return u + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    