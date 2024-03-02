import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class World():
    def __init__(self, wind_mean, wind_var) -> None:
        self.wind_mean = wind_mean 
        self.wind_var = wind_var 

    def acceleration(self):
        return random.gauss(self.wind_mean, self.wind_var)


class Drone():
    def __init__(self):
        self.pos = 0.0
        self.vel = 0.0

    def resetWalk(self):
        # Probability to move backwards or forwards
        prob = [0.05, 0.95]  
        
        start = 10.0
        positions = [start]
        
        # creating the random points
        rr = np.random.random(1000)
        downp = rr < prob[0]
        upp = rr > prob[1]
        
        
        for idownp, iupp in zip(downp, upp):
            down = idownp and positions[-1] > 1
            up = iupp and positions[-1] < 4

            positions.append(positions[-1] - down + up)
        
        self.walk = positions
    
    def measure(self, t):
        return random.gauss(self.walk[t], 0.8)
    

class KalmanFilter():
    def __init__(self, B, R, C, I, Z, Q, A:None) -> None:
        #Matrices
        self.A = A
        self.B = B
        self.R = R
        self.C = C
        self.I = I
        self.Q = Q

    #Update Filter
    def update(self, cov, state, measurement):
        # print("cov", cov)
        # print("state", state)
        # print("measurement", measurement)



        y = np.array(measurement) - self.C.dot(state)
        S = self.C.dot(cov).dot(self.C.T) + self.Q
        K = cov.dot(self.C.T).dot(inv(S))
        state = state + K*y
        cov = (self.I - K.dot(self.C)).dot(cov)

        return state, cov

    #Predict Step
    def predict(self, obs, cov, control):
        # print("observe state", obs)
        # print("observe cov", cov)
        # print("control", control)

        obs = np.dot(self.A,obs) + self.B*control
        cov = cov*self.A*self.A.T + self.R*self.B.dot(self.B.T)

        return obs, cov
    

    #Draw the confidence Ellipse
    def confidence_ellipse(self, x, y, cov, ax, n_std=1.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
    
        cov = cov
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        edgecolor = "red",facecolor="none", **kwargs)

     
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)


def question1():
    #Question 1
    A = np.array([[1.0,1.0], [0.0,1.0]], dtype=np.float32)
    B = np.array([[0.5],[1.0]], dtype=np.float32)
    ut = np.array([[0.0],[0.0]], dtype=np.float32)
    R = 1.0

    #World and Filter Classes
    world = World(wind_mean=0.0, wind_var=1.0)
    filter = KalmanFilter(A=A, B=B, C=None, R=R, I=None, Q=None, Z=None)

    #Start Subplots
    fig, axs = plt.subplots(nrows=1, ncols=5)
    sigt = 1.0

    #For each subplot, plot the uncertainty matrix
    for i, ax in enumerate(axs.reshape(-1)):
        #Predict State
        ut, sigt = filter.predict(ut, sigt,world.acceleration())
        print("ut", ut)
        print("cov", sigt)


        #Gaussian Data
        x=[random.gauss(ut[0], 1.0) for i in range(500)]
        y=[random.gauss(ut[1], 1.0) for i in range(500)]

        #Scatter Data
        ax.scatter(x, y, s=0.5)

        #Set Limits
        ax.set_xlim(min(x)-1,max(x)+1)
        ax.set_ylim(min(y)-1,max(y)+1)

        #Draw Ellipse

        filter.confidence_ellipse(x,y,sigt, ax, 1.0, facecolor="red")
        
    
    plt.show()

def question2p2():
    #Question 2
    A = np.array([[1.0,1.0], [0.0,1.0]], dtype=np.float32)
    B = np.array([[0.5],[1.0]], dtype=np.float32)
    C = np.array([[1,0]], dtype=np.float32)
    Q = np.array([[8.0]], dtype=np.float32)
    R = 1.0
    I = np.array([[1.0,0.0], [0.0,1.0]], dtype=np.float32)

    #Filter and World Class
    world = World(wind_mean=0.0, wind_var=1.0)
    filter = KalmanFilter(A=A, B=B, C=C, R=R, I=I, Q=Q, Z=None)
    drone = Drone()


    #Starting Position and Velocity
    ut = np.array([[0.0], [0.0]], dtype=np.float32)

    #Variance
    sigt = 1.0


    data=[]
    for j in range(20):
        drone.resetWalk()
        error_vals = []
        for i in range(20):
            ut, sigt = filter.predict(ut, sigt, world.acceleration())
            measurement = drone.measure(i)
            if i >= 5:
                ut, sigt = filter.update(cov=sigt, state=ut, measurement=measurement)
                error = abs(measurement - ut[0][0])
                error_vals.append(error)

        data.append(error_vals)

    x =np.linspace(5, 20, num=15)
    data = np.mean(data, axis=0)
    plt.plot(x,data)

    plt.title("Error between Measurement and True Position")
    plt.ylabel("Error")
    plt.xlabel("Time")
    plt.show()

def question2p3():
    #Question 2
    A = np.array([[1.0,1.0], [0.0,1.0]], dtype=np.float32)
    B = np.array([[0.5],[1.0]], dtype=np.float32)
    C = np.array([[1,0]], dtype=np.float32)
    Q = np.array([[0.8]], dtype=np.float32)
    R = 1.0
    I = np.array([[1.0,0.0], [0.0,1.0]], dtype=np.float32)

    #Filter and World Class
    world = World(wind_mean=0.0, wind_var=1.0)
    filter = KalmanFilter(A=A, B=B, C=C, R=R, I=I, Q=Q, Z=None)
    drone = Drone()


    #Starting Position and Velocity
    ut = np.array([[0.0], [0.0]], dtype=np.float32)

    #Variance
    sigt = 1.0

    #Start Subplots
    failprobs=[0.1, 0.6, 0.9]
    # failprobs=np.ones(10)
    data=[]
    for j in range(20):
        error_vals_all = []
        for failure in failprobs:
            error_vals = []
            drone.resetWalk()
            for i in range(20):
                ut, sigt = filter.predict(ut, sigt, world.acceleration())
                measurement = drone.measure(i)
                if np.random.rand() > failure and i >= 5:
                    
                    ut, sigt = filter.update(cov=sigt, state=ut, measurement=measurement)
                    error = abs(measurement - ut[0][0])
                    error_vals.append(error)
                else:
                    error = abs(measurement - ut[0][0])
                    error_vals.append(error)
            error_vals_all.append(error_vals)
        data.append(error_vals_all)

    x =np.linspace(0, 20, num=20)
    data = np.mean(data, axis=0)
    plt.plot(x,data[0])
    plt.plot(x,data[1])
    plt.plot(x,data[2])

    plt.title("Error With Sensor Failures")
    plt.ylabel("Error")
    plt.xlabel("Time")
    plt.legend([0.1,0.6,0.9])
    plt.show()

def main():
    question1()
    question2p2()
    question2p3()
    
if __name__ == "__main__":
    main()
