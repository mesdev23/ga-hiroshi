import numpy as np
import matplotlib.pyplot as plt
import copy
from Bezier import Bezier
from typing import List
from dataclasses import dataclass
import pickle
import asyncio
import concurrent.futures
import timeit
from db.models import Seed, SeedResults
import logging


# configure logging
logger = logging.getLogger(__name__)

# configure logging parameters
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# custom colours for logging


class Colours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# configure logging colours
logging.addLevelName(
    logging.INFO, f'{Colours.OKGREEN}{logging.getLevelName(logging.INFO)}{Colours.ENDC}')
logging.addLevelName(
    logging.WARNING, f'{Colours.WARNING}{logging.getLevelName(logging.WARNING)}{Colours.ENDC}')
logging.addLevelName(
    logging.ERROR, f'{Colours.FAIL}{logging.getLevelName(logging.ERROR)}{Colours.ENDC}')
logging.addLevelName(
    logging.CRITICAL, f'{Colours.FAIL}{logging.getLevelName(logging.CRITICAL)}{Colours.ENDC}')
logging.addLevelName(
    logging.DEBUG, f'{Colours.OKBLUE}{logging.getLevelName(logging.DEBUG)}{Colours.ENDC}')


def process_time(func):  # decorator to calculate the time elapsed for a function
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        func(*args, **kwargs)
        end = timeit.default_timer()

        print(f"[*] Time elapsed: {end-start:.2f} seconds")
    return wrapper


@dataclass
class Particle:
    control_points: np.ndarray
    fitness: float = float('inf')
    velocity: np.ndarray = None
    position: np.ndarray = None
    airfoil: np.ndarray = None

    def __post_init__(self):
        self.airfoil = np.loadtxt("SG6043.txt")
        self.position = self.control_points
        self.velocity = np.random.uniform(
            0.01, 0.05, self.control_points.shape)

    def update_fitness(self):
        points = np.linspace(0, 1, 162)
        ctrl = self.position.reshape(-1, 2)
        curve = Bezier.Curve(points, ctrl)

        # calculate the mean square error between the airfoil coordinates and the curve coordinates
        Xdiff = np.absolute(self.airfoil[:, 0] - curve[:, 0])
        Ydiff = np.absolute(self.airfoil[:, 1] - curve[:, 1])
        return (np.sum(np.sum(Xdiff) + np.sum(Ydiff)))*10

    @property
    def centroid(self):
        ctrlpoints = self.position.reshape(-1, 2)
        x = np.mean(ctrlpoints[:, 0])
        y = np.mean(ctrlpoints[:, 1])
        return x, y

    @property
    def shape(self):
        return self.control_points.shape

    def get_airfoil(self):
        x = self.airfoil[:, 0]
        y = self.airfoil[:, 1]
        return x, y

    def get_curve(self):
        points = np.linspace(0, 1, 162)
        ctrl = self.position.reshape(-1, 2)
        curve = Bezier.Curve(points, ctrl)
        x = curve[:, 0]
        y = curve[:, 1]
        return x, y

    def save_profile(self):
        # save the airfoil coordinates and the curve coordinates to a text file
        curve = Bezier.Curve(np.linspace(0, 1, 162),
                             self.position.reshape(-1, 2))
        np.savetxt('seed-finder/curve.txt', curve, fmt='%1.4f')

    def save_position(self) -> None:
        # save the control points to a text file
        # Seed.objects.create(position=self.position.reshape(-1,2).tolist())
        np.savetxt('seed-finder/control_points.txt',
                   self.position.reshape(-1, 2), fmt='%1.4f')


@dataclass
class Swarm:

    '''
        This class is used to find the controls points that could be used to approximate
        an airfoi shape. The class is initialized with the airfoil coordinates and
        the number of control points that should be found. The class will use 
        Particle Swarm Optimization to find the control points.
    '''

    seed_particle: Particle
    total_particles: int = 5
    swarm: List[Particle] = None
    swarm_best: List[Particle] = None
    w = 0.75
    c1 = 2.0
    c2 = 0.7
    converge_count = 0
    max_converge_count = 100
    original_seed: Particle = None

    def __post_init__(self):

        particles = self.seed_particle.control_points + np.random.uniform(
            0.0, 1.0, (self.total_particles, self.seed_particle.shape[0]))*np.random.uniform(0.01, 0.05, self.seed_particle.shape)  # shape = (5,22)
        self.original_seed = copy.deepcopy(self.seed_particle)
        # ENFORCE CONSTRAINTS: the first, second, second last and last control points should be the same

        particles[:, 0:20] = copy.deepcopy(
            self.original_seed.control_points[0:20])
        particles[:, -
                  20:] = copy.deepcopy(self.original_seed.control_points[-20:])

        self.swarm = [Particle(control_points=particle)
                      for particle in particles]
        self.swarm_best = copy.deepcopy(self.swarm)

    # inject noise to the swarm to avoid local minima and to explore the search space
    def inject_noise(self):
        for particle in self.swarm:
            particle.position = particle.position + \
                np.random.uniform(0.0, 1.0, particle.shape)
        # increase the threshold for convergence count so that the noise is not injected too often and allows the swarm to converge
        self.max_converge_count += int(self.max_converge_count*2.87)
        logger.info(">> Injecting noise to the swarm")

    def adapt_inertia_weight(self, iteration, max_iterations):
        self.w = self.w - (self.w/2)*iteration/max_iterations

    def update_converge_count(self):
        if self.converge_count < self.max_converge_count:
            self.converge_count += 1
        else:
            self.converge_count = 0

    def update_velocity(self):
        '''
            The function updates the velocity of each particle in the swarm using the following formula:
            v = w*v + c1*r1*(pbest-p) + c2*r2*(gbest-p)

            where;
                v: velocity of the particle
                w: inertia weight
                c1: cognitive parameter
                c2: social parameter
                r1: random number between 0 and 1
                r2: random number between 0 and 1
                pbest: best position of the particle taken from swarm copy
                gbest: best position of the swarm taken from swarm copy
                p: current position of the particle taken from swarm

        '''

        # get the particle with best position
        gbest = self.swarm_best[np.argmin(
            [particle.fitness for particle in self.swarm_best])].position

        async def uv(particle, i):
            r1 = np.random.uniform(0, 1, particle.shape)
            r2 = np.random.uniform(0, 1, particle.shape)
            particle.velocity = self.w*particle.velocity + self.c1*r1 * \
                (self.swarm_best[i].position-particle.position) + \
                self.c2*r2*(gbest-particle.position)

        async def run():
            tasks = []
            for i, particle in enumerate(self.swarm):
                tasks.append(uv(particle, i))
            await asyncio.gather(*tasks)

        asyncio.run(run())

        # def uv(particle, i):
        #     r1 = np.random.uniform(0,1,particle.shape)
        #     r2 = np.random.uniform(0,1,particle.shape)
        #     particle.velocity = self.w*particle.velocity + self.c1*r1*(self.swarm_best[i].position-particle.position) + self.c2*r2*(gbest-particle.position)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     tasks = []

        #     for i, particle in enumerate(self.swarm):
        #         tasks.append(executor.submit(uv, particle, i))

        #     concurrent.futures.wait(tasks)

        # for i, particle in enumerate(self.swarm):
        #     r1 = np.random.uniform(0.1,0.5,particle.shape)
        #     r2 = np.random.uniform(0.1,0.5,particle.shape)
        #     particle.velocity = self.w*particle.velocity + self.c1*r1*(self.swarm_best[i].position-particle.position) + self.c2*r2*(gbest-particle.position)

    def update_position(self):

        async def up(particle):
            particle.position = particle.position + particle.velocity

        # async def set_constrains(particle):
        #     particle.position[:2] = particle.position[-2:]
        #     # ensure smoothness of the curve especially at the trailing edge
        #     particle.position[2:4] = particle.position[0:2] + (particle.position[4:6] - particle.position[0:2])*0.5 # this line means that the second control point is the average of the first and third control points
        #     particle.position[-4:-2] = particle.position[-2:] + (particle.position[-6:-4] - particle.position[-2:])*0.5 # this line means that the second last control point is the average of the last and second last control points

        async def run():
            tasks = [up(particle) for particle in self.swarm]
            # tasks2 = [set_constrains(particle) for particle in self.swarm]
            # for particle in self.swarm:
            #     tasks.append(up(particle))
            #     tasks2.append(set_constrains(particle))

            await asyncio.gather(*tasks)
            # await asyncio.gather(*tasks2)

        asyncio.run(run())

        # def up(particle):
        #     particle.position = particle.position + particle.velocity

        # def set_constrains(particle):
        #     particle.position[:2] = particle.position[-2:]
        #     # ensure smoothness of the curve especially at the trailing edge
        #     particle.position[2:4] = particle.position[0:2] + (particle.position[4:6] - particle.position[0:2])*0.5

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     tasks = []
        #     tasks2 = []

        #     for particle in self.swarm:
        #         tasks.append(executor.submit(up, particle))
        #         tasks2.append(executor.submit(set_constrains, particle))

        #     concurrent.futures.wait(tasks)
        #     concurrent.futures.wait(tasks2)

        # for particle in self.swarm:
        #     particle.position = particle.position + particle.velocity

        # # make sure the constraints are enforced
        # for particle in self.swarm:
        #     particle.position[0:20] = copy.deepcopy(self.original_seed.control_points[0:20])
        #     particle.position[-20:] = copy.deepcopy(self.original_seed.control_points[-20:])

    def update_fitness(self):

        # async def uf(particle):
        #     particle.fitness = particle.update_fitness()
        #     particle.control_points = particle.position

        # async def run():
        #     tasks = [uf(particle) for particle in self.swarm]
        #     await asyncio.gather(*tasks)

        # asyncio.run(run())

        def uf(particle):
            particle.fitness = particle.update_fitness()
            particle.control_points = particle.position

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = []

            for particle in self.swarm:
                tasks.append(executor.submit(uf, particle))

            concurrent.futures.wait(tasks)

        # for particle in self.swarm:
        #     particle.fitness = particle.update_fitness()
        #     particle.control_points = particle.position

    def update_swarm(self):
        '''
            This function updates the swarm by updating the velocity, position and fitness of each particle
        '''
        self.update_velocity()
        self.update_position()
        self.update_fitness()

        async def uf(particle, i):
            p_best = self.swarm_best[i]
            if particle.fitness < p_best.fitness:
                self.swarm_best[i] = copy.deepcopy(particle)

        async def run():
            tasks = []
            for i, particle in enumerate(self.swarm):
                tasks.append(uf(particle, i))
            await asyncio.gather(*tasks)

        asyncio.run(run())

        # def uf(particle, i):
        #     p_best = self.swarm_best[i]
        #     if particle.fitness < p_best.fitness:
        #         self.swarm_best[i] = copy.deepcopy(particle)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     tasks = []

        #     for i, particle in enumerate(self.swarm):
        #         tasks.append(executor.submit(uf, particle, i))

        #     concurrent.futures.wait(tasks)
        # for i, particle in enumerate(self.swarm):
        #     p_best = self.swarm_best[i]
        #     if particle.fitness < p_best.fitness:
        #         self.swarm_best[i] = copy.deepcopy(particle)

    def save_swarm(self, path=None):
        if not path:
            path = 'seed-finder/swarm.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Swarm saved to {path}")


class Plotter:

    def __init__(self):
        plt.ion()

        self.figure = plt.figure(figsize=(20, 10))
        self.ax = self.figure.add_subplot(1, 2, 1)
        self.ax2 = self.figure.add_subplot(1, 2, 2)

        self.ax.set_title('Airfoil')
        self.ax2.set_title('Swarm')

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.grid()

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.grid()

        self.sg_airfoil, = self.ax.plot(
            [], [], 'k-', label='SG6043', alpha=0.5, linestyle='--')
        self.pso_airfoil, = self.ax.plot([], [], 'r-', label='PSO')
        self.ax.legend()

        self.swarm, = self.ax2.plot([], [], 'bo', label='Swarm')
        self.best_swarm, = self.ax2.plot([], [], 'ro', label='Best Swarm')
        self.ax2.legend()

    def update_airfoil(self, sg_xdata, sg_ydata, pso_xdata, pso_ydata):
        self.sg_airfoil.set_xdata(sg_xdata)
        self.sg_airfoil.set_ydata(sg_ydata)

        self.pso_airfoil.set_xdata(pso_xdata)
        self.pso_airfoil.set_ydata(pso_ydata)

        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-0.5, 0.5)

    def update_swarm(self, swarm_xdata, swarm_ydata, best_swarm_xdata, best_swarm_ydata):
        self.ax2.clear()
        self.ax2.grid()
        self.ax2.scatter(swarm_xdata, swarm_ydata, c='b',
                         marker='o', label='Swarm')
        self.ax2.scatter(best_swarm_xdata, best_swarm_ydata,
                         c='r', marker='o', label='Best Swarm', alpha=0.8)

        x_minima = min(np.min(swarm_xdata), np.min(best_swarm_xdata))
        x_maxima = max(np.max(swarm_xdata), np.max(best_swarm_xdata))
        y_minima = min(np.min(swarm_ydata), np.min(best_swarm_ydata))
        y_maxima = max(np.max(swarm_ydata), np.max(best_swarm_ydata))

        self.ax2.set_xlim(x_minima - x_minima*0.12, x_maxima + x_maxima*0.12)
        self.ax2.set_ylim(y_minima - y_minima*0.12, y_maxima + y_maxima*0.12)

    def refresh(self):
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save_plot(self, path=None):
        if not path:
            path = 'seed-finder/airfoil.png'
        self.figure.savefig(path)
        print(f"Plot saved to {path}")


if __name__ == "__main__":

    # initialize the control points
    # control_points = [
    #     [1, 0.0],  # Leading Edge Point
    #     [0.7, 109],  # Point near Leading Edge
    #     [0.5, 110],  # Maximum Thickness Point
    #     [0.357, 111],  # Upper Surface Point (at maximum camber)
    #     [0.174, 118],  # Lower Surface Point (at maximum camber)
    #     [0.025, 162], # Point near Maximum Thickness Point
    #     [0.00011, 25],  # Point near Trailing Edge
    #     [0.0, 0.0],  # Upper Surface Point (at trailing edge)
    #     [0.0, -25],  # Lower Surface Point (at trailing edge)
    #     [0.1, -42],  # Trailing Edge Point
    #     [0.2, -10],  # Point after the Trailing Edge
    #     [0.36, 0],  # Point after the Trailing Edge
    #     [0.5, -6.3],  # Point after the Trailing Edge
    #     [0.7, -5],  # Point after the Trailing Edge
    #     [1, 0.0],  # Point after the Trailing Edge
    # ]

    # # control_points = np.loadtxt('SG6043.txt').tolist()

    # for i in range(len(control_points)):
    #     control_points[i][1] = control_points[i][1] / 1000
    # control_points = np.loadtxt('best-solution/control_points.txt')

    Value = [0.62,
            0.56,
            0.19,
            0.24,
            0.02,
            0.0,
            0.0,
            0.08,
            0.54,
            0.54,
            0.01,
            1.02,
            97,
            188,
            117,
            127,
            80,
            17,
            25,
            10,
            0,
            2,
            0,
            2
        ]

    Ux1 = 1.0
    Ux2 = Value[0]
    Ux3 = Value[1]
    Ux4 = Value[2]
    Ux5 = Value[3]
    Ux6 = Value[4]
    Ux7 = Value[5]
    Ux8 = 0.0

    # Lower Surface x Control Points
    Lx1 = 1.0
    Lx2 = Value[11]
    Lx3 = Value[10]
    Lx4 = Value[9]
    Lx5 = Value[8]
    Lx6 = Value[7]
    Lx7 = Value[6]
    Lx8 = 0.0
    
    # Upper Surface y Control Points
    Uy1 = 0.0
    Uy2 = Value[12]
    Uy3 = Value[13]
    Uy4 = Value[14]
    Uy5 = Value[15]
    Uy6 = Value[16]
    Uy7 = Value[17]
    Uy8 = 0.0


    # Lower Surface y Control Points
    Ly1 = 0.0
    Ly2 = -Value[23]
    Ly3 = -Value[22]
    Ly4 = -Value[21]
    Ly5 = -Value[20]
    Ly6 = -Value[19]
    Ly7 = -Value[18]
    Ly8 = 0.0

    control_points = [
        [Ux1, Uy1],  # Leading Edge Point
        [Ux2, Uy2],  # Point near Leading Edge
        [Ux3, Uy3],  # Maximum Thickness Point
        [Ux4, Uy4],  # Upper Surface Point (at maximum camber)
        [Ux5, Uy5],  # Lower Surface Point (at maximum camber)
        [Ux6, Uy6], # Point near Maximum Thickness Point
        [Ux7, Uy7],  # Point near Trailing Edge
        [Ux8, Uy8],  # Upper Surface Point (at trailing edge)
        [Lx7, Ly7],  # Lower Surface Point (at trailing edge)
        [Lx6, Ly6],  # Trailing Edge Point
        [Lx5, Ly5],  # Point after the Trailing Edge
        [Lx4, Ly4],  # Point after the Trailing Edge
        [Lx3, Ly3],  # Point after the Trailing Edge
        [Lx2, Ly2],  # Point after the Trailing Edge
        [Lx1, Ly1],  # Point after the Trailing Edge
    ]

    for i in range(len(control_points)):
        control_points[i][1] = control_points[i][1] / 1000

    # flatten the array
    seed_control_points = np.array(control_points).flatten()
    particle = Particle(seed_control_points)

    # initialize the swarm
    swarm = Swarm(particle, total_particles=50)

    iterations = []
    max_itr = 500
    tolerance = 0.01

    g_best = copy.deepcopy(swarm.swarm_best[np.argmin(
        [p.fitness for p in swarm.swarm_best])])

    start_time = timeit.default_timer()  # start time

    # create seed
    seed = Seed.objects.create()

    for itr in range(max_itr):
        best_particle = swarm.swarm_best[np.argmin(
            [p.fitness for p in swarm.swarm_best])]
        if best_particle.fitness < tolerance:
            logger.critical(f"[+] Converged at iteration {itr+1}..")
            break

        if best_particle.fitness < g_best.fitness:
            g_best = copy.deepcopy(best_particle)
        else:
            if swarm.converge_count == swarm.max_converge_count:
                swarm.inject_noise()

            # update the swarm converge_count
            swarm.update_converge_count()

        logger.info(f"[+] Iteration: {itr+1}, MSE: {best_particle.fitness}..")
        swarm.update_swarm()

        iterations.append(itr+1)

        if itr % 50 == 0:  # save the results every 50 iterations
            SeedResults.objects.create(
                seed=seed,
                iteration=itr+1,
                solution={'solution': best_particle.position.tolist()},
                mse=best_particle.fitness
            )
            logger.info(f"[+] Saved results for iteration {itr+1}..")

    elapsed = timeit.default_timer() - start_time  # end time
    logger.info(f"[+] Time taken: {elapsed} seconds..")

    # get the best particle
    best_particle = swarm.swarm_best[np.argmin(
        [p.fitness for p in swarm.swarm_best])]
    # print(f"\n[+] Best Particle: {best_particle.position}, MSE: {best_particle.fitness}")
    # save the best particle
    best_particle.save_position()
    # save the plot
