from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from Bezier import Bezier
from subprocess import Popen, PIPE, TimeoutExpired
import os
from os import path
import sys
import uuid
import asyncio
import copy
import timeit
import pickle
from matplotlib import pyplot as plt
import concurrent.futures
import logging
from db.models import Swarm as SwarmModel
from db.models import Solutions as SolutionsModel
import psutil
import subprocess
import multiprocessing

logger = logging.getLogger(__name__)

# configure logging parameters
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
logging.addLevelName(logging.INFO, f'{Colours.OKGREEN}{logging.getLevelName(logging.INFO)}{Colours.ENDC}')
logging.addLevelName(logging.WARNING, f'{Colours.WARNING}{logging.getLevelName(logging.WARNING)}{Colours.ENDC}')
logging.addLevelName(logging.ERROR, f'{Colours.FAIL}{logging.getLevelName(logging.ERROR)}{Colours.ENDC}')
logging.addLevelName(logging.CRITICAL, f'{Colours.FAIL}{logging.getLevelName(logging.CRITICAL)}{Colours.ENDC}')
logging.addLevelName(logging.DEBUG, f'{Colours.OKBLUE}{logging.getLevelName(logging.DEBUG)}{Colours.ENDC}')


os.environ['DISPLAY'] = ':1' # set environment variable for xvfb


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        logger.info(f'{func.__name__} took {end - start} seconds')
        return result
    return wrapper

@dataclass
class Particle:
    
    '''
        A particle represents a set of control points that are used to 
        generate the airfoil shape. The profile is generated using a Bezier curve.
        
    '''
    control_points: np.ndarray # 1D array of control points
    fitness: float = 0.0 # The fitness of the particle
    particle_id: str = None # The unique id of the particle
    additional_constraints: List = None # Additional constraints that can be added to the particle
    velocity: np.ndarray = None # The velocity of the particle
    position:np.ndarray = None
    results: dict = None
    gradient: np.ndarray = None
    # position: np.ndarray = None # The best position of the particle
    # best_fitness: float = 0.0 # The best fitness of the particle

    def __post_init__(self):
        if not isinstance(self.control_points, np.ndarray):
            self.control_points = np.array(self.control_points)

        self.particle_id = str(uuid.uuid4()) # Generate a unique id for the particle
        self.velocity = np.random.uniform(0.001, 0.005, self.control_points.shape)
        self.position = self.control_points.copy()
        self.gradient = np.zeros(self.control_points.shape) # The gradient of the particle


    @property
    def shape(self):
        return self.control_points.shape

    def evaluate_constraints(self, additional_constraints) -> None:
        if additional_constraints is None:
            additional_constraints = []

        # Check if the control points are within the domain
        if np.any(self.control_points < -1) or np.any(self.control_points > 1):
            raise ValueError(f'Control points must be within the domain [-1, 1] but got {self.control_points}')

        # the x,y coordinates of the first and last control points must be 1,0
        # if tuple(self.control_points[:2]) != (1, 0) or tuple(self.control_points[-2:]) != (1, 0):
        #     raise ValueError(f'The first and last control points must be (1, 0) but got {self.control_points[:2]} and {self.control_points[-2:]}')

        # evaluate additional constraints
        for constraint in additional_constraints:
            constraint(self) # the constraint function will evalute the control points and raise an error if the constraint is not satisfied

        # check that the thickness of the airfoil is within the domain

    @property
    def centroid(self): # Returns the centroid of the particle
        # reshape the control points
        ctrlpoints = self.control_points.reshape(-1, 2)
        x = np.mean(ctrlpoints[:, 0])
        y = np.mean(ctrlpoints[:, 1])
        return x, y

    @property
    def bezierProfile(self) -> List[Tuple[float, float]]:
        # generate the airfoil profile using a Bezier curve
        points = np.linspace(0, 1, 32)
        # reshape the control points
        ctrlpoints = self.position.reshape(-1,2)
        curve = Bezier.Curve(points, ctrlpoints)
        return curve.tolist()

    @property
    def max_thickness(self) -> float:
        # Returns the maximum thickness of the airfoil
        return max(point[1] for point in self.bezierProfile)
    
    def plot(self):
        # plot the airfoil profile
        x, y = zip(*self.bezierProfile)
        plt.plot(x, y)
        plt.show()

    def save_profile(self) -> None:
        # save the airfoil profile to a txt file
        with open('best-solution/airfoil.txt', 'w') as f:
            array = np.array(self.bezierProfile)
            np.savetxt(f, array, fmt='%.3f')

    def save_control_points(self):
        with open('best-solution/control_points.txt', 'w') as f:
            np.savetxt(f, self.position.reshape(-1, 2))

    def save_airfoil(self) -> None:
        # save the airfoil profile to a txt file
        with open(f'airfoils/{self.particle_id}.txt', 'w') as f:
            array = np.array(self.bezierProfile)
            np.savetxt(f, array, fmt='%.3f')
    
    def load_results_from_file(self):
        results = {
            'alpha': [],
            'cl': [],
            'cd': [],
            'cm': [],
            'cdp': [],
            'xtr_top': [],
            'xtr_bot': [],
            'itr_top': [],
            'itr_bot': [],
        }

        # read the xfoil output file
        if not path.exists(f'outputs/{self.particle_id}.txt'):
            return results
        with open(f'outputs/{self.particle_id}.txt', 'r') as f:
            lines = f.readlines()
            # remove empty lines
            lines = [line for line in lines if line.strip() != '']

        results_list = [[float(x) for x in line.split()[:7]] for line in lines[7:]] # skip the first 7 lines of the file and get the first 6 values
        for result in results_list:
            results['alpha'].append(result[0])
            results['cl'].append(result[1])
            results['cd'].append(result[2])
            results['cdp'].append(result[3])
            results['cm'].append(result[4])
            # results['xtr_top'].append(result[5])
            # results['xtr_bot'].append(result[6])
            # results['itr_top'].append(result[7])
            # results['itr_bot'].append(result[8])

        # for line in lines[7:]: # skip the first 7 lines of the file
        #     # split the line into a list
        #     line = line.split()
        #     # get the values
        #     try:
        #         results['alpha'].append(float(line[0]))
        #         results['cl'].append(float(line[1]))
        #         results['cd'].append(float(line[2]))
        #         results['cdp'].append(float(line[3]))
        #         results['cm'].append(float(line[4]))
        #         results['xtr_top'].append(float(line[5]))
        #         results['xtr_bot'].append(float(line[6]))
        #         results['itr_top'].append(float(line[7]))
        #         results['itr_bot'].append(float(line[8]))
        #     except ValueError:
        #         continue

        return results

    def calculate_fitness(self, res:dict) -> float:
        fitness = 0.0
        if res['cd'] == []:
            return fitness
        if len(res['cd']) < 7:
            return fitness

        # avg_cl = sum(res['cl'])/len(res['cl']) # average lift coefficient
        # avg_cd = sum(res['cd'])/len(res['cd'])
        # fitness += avg_cl/avg_cd
        coeff = [cl/cd for cl, cd in zip(res['cl'], res['cd'])]
        fitness = sum(coeff)
        return res

    def solve(self):
        # run xfoil simulation on the generated airfoil profile
        # self.save_airfoil_coordinates() # save the airfoil profile to a txt file to be used by xfoil
        self.save_airfoil()
        try:
            # direct xfoil to xvfb
            # xvfb = Popen(["Xvfb", ":1", "-screen", "0", "1024x768x24"], close_fds=True)
            # run xfoil using subprocess
            p = Popen(["xfoil", "-s" "-n", "-p"], shell=True, stdin=PIPE,
                    stdout=PIPE, stderr=PIPE, close_fds=True)  # run xfoil

            # send commands to xfoil
            p.stdin.write(b'load\n')
            p.stdin.write(f'airfoils/{self.particle_id}.txt\n'.encode())
            p.stdin.write(b'ppar\n')
            p.stdin.write(b'n\n')
            p.stdin.write(b'260\n')
            p.stdin.write(b'\n')
            p.stdin.write(b'\n')
            p.stdin.write(b'oper\n')
            p.stdin.write(b'visc\n')
            p.stdin.write(b'600000\n')
            p.stdin.write(b'iter\n')
            p.stdin.write(b'3000\n')
            p.stdin.write(b'pacc\n')
            p.stdin.write(f'outputs/{self.particle_id}.txt\n'.encode())
            p.stdin.write(b'\n')
            p.stdin.write(b'\n')
            p.stdin.write(b'oper\n')
            p.stdin.write(b'aseq\n')
            p.stdin.write(b'0\n')
            p.stdin.write(b'12\n')
            p.stdin.write(b'2\n')
            p.stdin.write(b'\n')
            p.stdin.write(b'!\n')
            p.stdin.write(b'\n')
            p.stdin.write(b'\n')
            p.stdin.write(b'quit\n')
            # implement a try catch block to handle the timeout error
            
            # run the process and wait only 5 seconds for it to finish, then kill it
            output, error = p.communicate(timeout=5) #
            if p.poll() is None:
                p.kill()

            # xvfb.kill()
        except TimeoutExpired:
            logger.warning(f'   >> Xfoil process timed out for particle {self.particle_id}')
            p.kill()
            # xvfb.kill()
            # clean up the files
            if path.exists(f'airfoils/{self.particle_id}.txt'):
                os.remove(f'airfoils/{self.particle_id}.txt')
            if path.exists(f'outputs/{self.particle_id}.txt'):
                os.remove(f'outputs/{self.particle_id}.txt')

            return 0.0, {
                'alpha': [],
                'cl': [],
                'cd': [],
                'cm': [],
                'cdp': [],
                'xtr_top': [],
                'xtr_bot': [],
                'itr_top': [],
                'itr_bot': [],
            }

        res = self.load_results_from_file()
        # clean up the files

        if path.exists(f'airfoils/{self.particle_id}.txt'):
            os.remove(f'airfoils/{self.particle_id}.txt')
        # os.remove(f'airfoils/{self.particle_id}.txt')
        if path.exists(f'outputs/{self.particle_id}.txt'):
            os.remove(f'outputs/{self.particle_id}.txt')
        # os.remove(f'outputs/{self.particle_id}.txt')
        
        fitness = self.calculate_fitness(res)

        return fitness,res

    def __repr__(self):
        return f'Particle: {self.particle_id}'

@dataclass
class Swarm:
    
    seed_particle: Particle # the seed particle used as reference to initialize the swarm
    total_particles: int = 5 # the number of particles in the swarm
    ASYNC:bool = False # run the simulation asynchronously
    c1:float = 2.0 # cognitive constant
    c2:float = 0.7 # social constant
    w:float = 0.75 # inertia constant
    swarm: List[Particle] = None # the swarm is a collection of particles
    swarm_best: List[Particle] = None # copy of the swarm to hold best
    swarm_id: str = None # the id of the swarm
    g_best = None
    original_seed:Particle = None
    converge_count = 0 
    max_converge_count = 100
    swarm_db = None
 
    def __post_init__(self):
        '''
            Post initialization, the swarm must be initiliazed. The swarm is
            initiliazed using the control points of the seed particle which is added to the product of a randomly generated array and the array of initial velocity. The initial velocity can also be a randomly generated array.

            swarm = seed + array1*array2

            where;
                seed: the control points of the seed particle
                array1: randomly generated array of shape [n,k]
                        n is the number of particles in the swarm
                        k is the number of control points
                array2: initial velocity array of shape [j,l]
                        j = 1 (only 1 row)
                        l = 22 (the no of control points x 2)

        '''

        self.original_seed = copy.deepcopy(self.seed_particle)
        particles = self.seed_particle.control_points + np.random.uniform(0.0,0.0,(self.total_particles, self.seed_particle.shape[0]))*np.random.uniform(0.0,0.0, self.seed_particle.shape) # shape = (5,22)

        # ENFORCE CONSTRAINTS, the trailing section must be remain the same
        particles[:,0:2] = copy.deepcopy(self.original_seed.control_points[0:2]) # enforce constraints
        particles[:,-2:] = copy.deepcopy(self.original_seed.control_points[-2:]) 

        self.swarm = [Particle(control_points=particle) for particle in particles]
        self.swarm_best =  copy.deepcopy(self.swarm) # stores personal best for each particle
        self.swarm_id = str(uuid.uuid4()) 

        self.swarm_db = self.__initiliase_db__()
       

        logger.info(f'>> Swarm initialized with {self.total_particles} particles')

        
    def inject_noise(self):
        '''
            The function injects noise to the swarm when expected improvement is not observed for a certain number of iterations. The noise is injected by adding a random number to the position of each particle in the swarm. 
        '''

        for particle in self.swarm:
            particle.position[2:-2] = particle.position[2:-2] + np.random.uniform(0.0, 0.001, particle.position[2:-2].shape)

        # increase the converge count so that the noise is not injected too often and allows the swarm to converge
        self.max_converge_count += int(self.max_converge_count*1.5)
        logger.info(f'>> Noise injected to the swarm due to no improvement..')

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
        gbest = self.swarm_best[np.argmax([particle.fitness for particle in self.swarm_best])].position

        if self.ASYNC == True:
            async def uv(particle, i):
                particle.velocity = self.w*particle.velocity + self.c1*np.random.uniform(0.0,0.002,particle.shape)*(self.swarm_best[i].position-particle.position) + self.c2*np.random.uniform(0.0,0.002,particle.shape)*(gbest-particle.position)
                    

            # async def ug(particle, i):
            #     '''
            #         The function calculates the gradient of the swarm. The gradient is calculated using the formula:

            #         gradient = c1*r1*(pbest - current_position) + c2*r2*(gbest - current_position)

            #         where;
            #             c1: cognitive constant
            #             c2: social constant
            #             r1: random number between 0 and 1
            #             r2: random number between 0 and 1
            #             pbest: personal best
            #             gbest: global best
            #             current_position: the current position of the particle

            #         The gradient is then added to the current velocity of the particle and the new velocity is returned.
            #     '''

            #     p_best = self.swarm_best[i]
            #     r1 = np.random.uniform(0.0, 1.0, particle.shape)
            #     r2 = np.random.uniform(0.0, 1.0, particle.shape)
            #     gradient = self.c1*r1*(p_best.position - particle.position) + self.c2*r2*(gbest.position - particle.position)
            #     particle.velocity += gradient

            async def run():
                # tasks = []
                # for i, particle in enumerate(self.swarm):
                #     tasks.append(uv(particle, i))
                tasks = [uv(particle, i) for i, particle in enumerate(self.swarm)]
                # tasks2 = [ug(particle, i) for i, particle in enumerate(self.swarm)]
                await asyncio.gather(*tasks)
                # await asyncio.gather(*tasks2)
            
            asyncio.run(run())
            
        else:
            # update the velocity of each particle in the swarm
            for i, particle in enumerate(self.swarm):
                r1 = np.random.uniform(0.0, 1.0,particle.shape)
                r2 = np.random.uniform(0.0, 1.0,particle.shape)
                particle.velocity = self.w*particle.velocity + self.c1*r1*(self.swarm_best[i].position-particle.position) + self.c2*r2*(gbest-particle.position)
        
    def update_position(self, ignore=False):
        '''
            The function updates the position of each particle in the swarm using the following formula:
            p = p + v

            where;
                p: position of the particle
                v: velocity of the particle
        '''

        if self.ASYNC == True:

            async def up(particle):
                if ignore == False:
                    particle.position = particle.position + particle.velocity
                    particle.position[0:2] = copy.deepcopy(self.original_seed.control_points[0:2])
                    particle.position[-2:] = copy.deepcopy(self.original_seed.control_points[-2:])
                else:
                    particle.position = copy.deepcopy(self.original_seed.control_points)
                    

            # async def set_constraints(particle):
            #    # ENFORCE CONSTRAINTS, the trailing section must be remain the same
            #     particle.position[0:2] = copy.deepcopy(self.original_seed.control_points[0:2])
            #     particle.position[-2:] = copy.deepcopy(self.original_seed.control_points[-2:])
                
            async def run():
                # tasks = []
                # tasks2 = []
                # for particle in self.swarm:
                #     tasks.append(up(particle))
                #     tasks2.append(set_constraints(particle))
                tasks = [up(particle) for particle in self.swarm]
                # tasks2 = [set_constraints(particle) for particle in self.swarm]

                await asyncio.gather(*tasks)
                # await asyncio.gather(*tasks2)

            asyncio.run(run())
            
            # def up(particle):
            #     particle.position = particle.position + particle.velocity
            #     particle.position[0:2] = copy.deepcopy(self.original_seed.control_points[0:2])
            #     particle.position[-2:] = copy.deepcopy(self.original_seed.control_points[-2:])

            # # def set_constraints(particle):
            #    # ENFORCE CONSTRAINTS, the trailing section must be remain the same
            #     particle.position[0:2] = copy.deepcopy(self.original_seed.control_points[0:2])
            #     particle.position[-2:] = copy.deepcopy(self.original_seed.control_points[-2:])

            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(up, particle) for particle in self.swarm]
            #     concurrent.futures.wait(futures)

            
        else:
            for particle in self.swarm:
                particle.position = particle.position + particle.velocity

            # make sure the constraints are enforced
            for particle in self.swarm:
                particle.position[0:2] = copy.deepcopy(self.original_seed.control_points[0:2])
                particle.position[-2:] = copy.deepcopy(self.original_seed.control_points[-2:])

    def update_fitness(self):
        '''
            The function updates the fitness of each particle in the swarm
        '''
        # if self.ASYNC == True:
        #     async def uf(particle):
        #         particle.fitness, particle.results = particle.solve()
        #         # update control point
        #         particle.control_points = particle.position

        #     async def run():
        #         # tasks = []
        #         # for particle in self.swarm:
        #         #     tasks.append(uf(particle))
        #         tasks = [uf(particle) for particle in self.swarm]
        #         await asyncio.gather(*tasks)
                
        #     asyncio.run(run())

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(uf, particle) for particle in self.swarm]
        #     concurrent.futures.wait(futures)
        # run the uf function concurrently
        # with concurrent.futures.ThreadPoolExecutor() as executor:
            # futures = [executor.submit(uf, particle) for particle in self.swarm]
            # concurrent.futures.wait(futures)
            # results = executor.map(uf, self.swarm)
            # for result in results:
            #     logger.info(result)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
            
        #     futures = [executor.submit(uf, particle) for particle in self.swarm]
        #     concurrent.futures.wait(futures, timeout=5)
        # 
        def uf(particle):
            particle.fitness, particle.results = particle.solve()
            # update control point
            particle.control_points = particle.position

        # use the multiprocessing module
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(uf, particle) for particle in self.swarm]
            concurrent.futures.wait(futures, timeout=5)
        
        # else:
        # for particle in self.swarm:
        #     particle.fitness, particle.results = particle.solve()
        #     # update control point
        #     particle.control_points = particle.position

    def update_swarm(self, iteration):
        '''
            The function updates the swarm by updating the velocity, position, fitness and best position of each particle in the swarm
        '''
        # in the first iteration, set all the particles to the same position as the original seed
        self.update_velocity()
        if iteration == 0:
            self.update_position(ignore=True)
        else:
            self.update_position()
        self.update_fitness()

        if self.ASYNC == True:
            async def u_s(particle, i):
                p_best = self.swarm_best[i]
                if particle.fitness > p_best.fitness:
                    self.swarm_best[i] = copy.deepcopy(particle)
                    self.converge_count = 0 # decrease the converge count if the swarm is converging i.e the best position of the swarm is changing
                else:
                    self.converge_count += 1
            
            async def run():
                tasks = [u_s(particle, i) for i, particle in enumerate(self.swarm)]
                await asyncio.gather(*tasks)
            
            asyncio.run(run())
        # else:
        #     for i, particle in enumerate(self.swarm):
        #         p_best = self.swarm_best[i]
        #         if particle.fitness > p_best.fitness:
        #             self.swarm_best[i] = copy.deepcopy(particle)
        #             self.converge_count = 0 # decrease the converge count if the swarm is converging i.e the best position of the swarm is changing

        #         else:
        #             self.converge_count += 1 # increase the converge count if the swarm is not converging i.e the best position of the swarm is not changing

    def save_swarm(self, path=None):
        '''
            The function saves the swarm to a pickle file
        '''
        if not path:
            path = f"swarm_{self.swarm_id}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f">> Swarm saved to: {path}")

    def __initiliase_db__(self):

        swarm = SwarmModel.objects.create(
            seed_particle={
                "control_points": self.original_seed.control_points.tolist(),
            },
            total_particles=self.total_particles,
            w=self.w, c1=self.c1, c2=self.c2,
            
        )
        
        return swarm
            

    def update_solution_model(self):
        logger.info(">> Updating solution model...")
        SolutionsModel.objects.create(
            swarm=self.swarm_db,
            particles={
                'particles':[
                    [particle.position.tolist()] for particle in self.swarm
                ],
            },
            best_particles={
                'particles':[particle.position.tolist() for particle in self.swarm_best]
            },
        )

    @classmethod
    def load_swarm(cls, path, *args, **kwargs):
        '''
            The function loads the swarm from a pickle file
            and recreates the swarm object
        '''

        print("Loading swarm...")
        # check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        with open(path, 'rb') as f:
            # check if the file is empty
            if os.stat(path).st_size == 0:
                raise ValueError(f"File {path} is empty")

            # load the class object
            obj = pickle.load(f)
            # check if the object is a swarm object
            if isinstance(obj, Swarm):
                # update the swarm object with the new parameters
                for par in args:
                    obj.__dict__.update(par)
                for key, value in kwargs.items():
                    obj.__dict__.update({key: value})
                # call the post__init__ function to update the swarm
                obj.__post_init__()
                return obj 
            else:
                raise TypeError(f"File {path} is not a swarm object")

    def __repr__(self):
        return f'Swarm: {self.swarm_id}'

class Plotter:

    def __init__(self):
        plt.ion() # set plot to animated
        self.figure = plt.figure(figsize=(20,10)) # create figure
        self.ax1 = self.figure.add_subplot(2,2,1) # add subplot 1 (1 row, 2 columns, first plot)
        self.ax2 = self.figure.add_subplot(2,2,2) # add subplot 2 (1 row, 2 columns, second plot)
        self.ax3 = self.figure.add_subplot(2,2,3) # add subplot 3 (1 row, 2 columns, third plot)
        self.ax4 = self.figure.add_subplot(2,2,4) # add subplot 4 (1 row, 2 columns, fourth plot)

        self.ax1.set_title('Residuals')
        self.ax2.set_title('Airfoil')
        self.ax3.set_title('Swarm')
        self.ax4.set_title('Performance')

        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Fitness')

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')

        self.ax3.set_xlabel('x')
        self.ax3.set_ylabel('y')

        self.ax4.set_xlabel('Alpha')
        self.ax4.set_ylabel('Cl/Cd')

        self.residual, = self.ax1.plot([], [], 'b-')
        self.ax1.grid()

        self.profile, = self.ax2.plot([], [], 'b-')
        self.ctrl_pts, = self.ax2.plot([], [], 'ro')
        self.ax2.grid()

        self.swarm, = self.ax3.plot([], [], 'bo', label='Current Generation')
        self.swarm_best, = self.ax3.plot([], [], 'ro', label='Best Generation')
        self.ax3.grid()
    

        self.performance, = self.ax4.plot([], [], 'b-')
        self.performance_2, = self.ax4.plot([], [], 'r-')
        self.ax4.grid()
        # self.ax4.legend()

    def update_residual_plot(self, xdata, ydata):
        # self.ax1.plot(xdata, ydata, 'b-')
        self.residual.set_xdata(xdata)
        self.residual.set_ydata(ydata)
        self.ax1.set_xlim(0, max(xdata)*1.5)
        self.ax1.set_ylim(0, max(ydata)*1.5)

    def update_profile_plot(self, xdata, ydata, xctrl, yctrl):

        self.profile.set_xdata(xdata)
        self.profile.set_ydata(ydata)
        self.ctrl_pts.set_xdata(xctrl)
        self.ctrl_pts.set_ydata(yctrl)
        self.ctrl_pts.set_markersize(5)
        self.ctrl_pts.set_markerfacecolor('r')
        self.ctrl_pts.set_alpha(0.5)

        self.ax2.set_xlim(-1.5, 1.5)
        self.ax2.set_ylim(-1.5, 1.5)
        # # set the scale of the plot
        # self.ax2.set_xlim(min(xdata)*1.5, max(xdata)*1.5)
        # self.ax2.set_ylim(min(ydata)*1.5, max(ydata)*1.5)

    def update_swarm_plot(self, swarm_xdata, swarm_ydata, best_swarm_xdata, best_swarm_ydata):
        self.ax3.clear()
        self.ax3.grid()
        self.ax3.scatter(swarm_xdata, swarm_ydata, c='b', marker='o', label='Swarm')
        self.ax3.scatter(best_swarm_xdata, best_swarm_ydata, c='r', marker='o', label='Best Swarm', alpha=0.8)
        
        x_minima = min(np.min(swarm_xdata), np.min(best_swarm_xdata))
        x_maxima = max(np.max(swarm_xdata), np.max(best_swarm_xdata))
        y_minima = min(np.min(swarm_ydata), np.min(best_swarm_ydata))
        y_maxima = max(np.max(swarm_ydata), np.max(best_swarm_ydata))

        self.ax2.set_xlim(x_minima - x_minima*0.12, x_maxima + x_maxima*0.12)
        self.ax2.set_ylim(y_minima - y_minima*0.12, y_maxima +  y_maxima*0.12)

    def update_performance_plot(self, xdata, ydata, xdata_2, ydata_2):
        self.performance.set_xdata(xdata)
        self.performance.set_ydata(ydata)

        self.performance_2.set_xdata(xdata_2)
        self.performance_2.set_ydata(ydata_2)

        self.performance.set_color('b')
        self.performance.set_label('Best Particle')

        self.performance_2.set_color('r')
        self.performance_2.set_label('Seed Particle')
        # set legend
        self.ax4.legend()

        x_minima = min(np.min(xdata), np.min(xdata_2))
        x_maxima = max(np.max(xdata), np.max(xdata_2))
        y_minima = min(np.min(ydata), np.min(ydata_2))
        y_maxima = max(np.max(ydata), np.max(ydata_2))
        self.ax4.set_xlim(x_minima - x_minima*0.12, x_maxima + x_maxima*0.12)
        self.ax4.set_ylim(y_minima - y_minima*0.12, y_maxima +  y_maxima*0.12)


    def refresh(self):
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save_plot(self, path=None):
        if not path:
            path = "plot.png"
        plt.savefig(path)
        print(f"Plot saved to: {path}")

def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

if __name__ == '__main__':
    
    if not checkIfProcessRunning('Xvfb'):
        xvfb = Popen(["Xvfb", ":1", "-screen", "0", "1024x768x24"], close_fds=True) # start virtual display
        logger.info(">> Xvfb started")
    else:
        logger.info(">> Xvfb already running")

    os.environ['DISPLAY'] = ':1' # set display

    # control_points = np.loadtxt('SG6043.txt').flatten()

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

    if path.exists('best-solution/control_points.txt'):
        control_points = np.loadtxt('best-solution/control_points.txt')
        logger.info(">> Control points loaded from previous run")
    else:
        control_points = np.loadtxt('SG6043.txt')
        logger.info(">> Control points loaded from SG6043.txt")
    # for i in range(len(control_points)):
    #     control_points[i][1] = control_points[i][1] / 1000

    control_points = np.array(control_points).flatten()

    seed_particle = Particle(control_points)
    temp = copy.deepcopy(seed_particle)
    seed_fitness, seed_res = temp.solve() # solve the seed particle
    # target_fitness = 84.5
    sg_coeff = [115.922619,
                147.4285714,
                148.7848101,
                121.3309025,
                92.73074475,
                73.5483871,
                56.83816014,
                41.41044061,
                28.41207988,
                18.57112069
                ]


    # target_fitness = sum(sg_coeff[:7])
    target_fitness = 756.5842
    
    # seed particle results
    s_alpha = seed_res['alpha']
    s_cd = seed_res['cd']
    s_cl = seed_res['cl']
    s_coeff = sg_coeff

    # # # create the swarm
    # LOAD_SWARM = False
    # #instantiate swarm object with the seed particle and the total number of particles
    # if LOAD_SWARM:
    #     swarm = Swarm(seed_particle).load_swarm('swarm.pkl', total_particles=30, ASYNC=True)
    # else:
    swarm = Swarm(seed_particle, total_particles=15, ASYNC=True)

    # initialize the plotter
    # plotter = Plotter()
    start_time = timeit.default_timer()

    iterations = []
    residuals = []

    max_iterations = 100

    g_best = copy.deepcopy(seed_particle)
    logger.info(f"** Seed particle fitness: {seed_fitness}")

    for _ in range(max_iterations):
        # update the swarm
        try:
            # check if there are xfoil processes running

            logger.info(f"[+] Iteration: {_+1}...")
       
            swarm.update_swarm(_)

            local_best = swarm.swarm[np.argmax([particle.fitness for particle in swarm.swarm])]
            best_particle = swarm.swarm_best[np.argmax([particle.fitness for particle in swarm.swarm_best])]

            # data for residual plot
            # iterations.append(_+1)
            # residuals.append(swarm.swarm_best[np.argmax([particle.fitness for particle in swarm.swarm_best])].fitness)

            # data for profile plot
            # ctrlpts = best_particle.position.reshape(-1,2)

            # data for performance plot of best particle
            # alpha = best_particle.results['alpha']
            # cl = best_particle.results['cl']
            # cd = best_particle.results['cd']
            
            # coeff = [cl/cd for cl, cd in zip(cl, cd)]

            # xctrl = ctrlpts[:,0]
            # yctrl = ctrlpts[:,1]

            # profile = best_particle.bezierProfile
            # xprofile = [p[0] for p in profile]
            # yprofile = [p[1] for p in profile]

            if best_particle.fitness > g_best.fitness:
                logger.info(f">> New best fitness: {best_particle.fitness} improved from {g_best.fitness}")
                g_best = copy.deepcopy(best_particle)
                # save the best particle control points 
            else:
                logger.info(f">> No improvement in fitness: {local_best.fitness} vs {g_best.fitness}")
                if swarm.converge_count >= swarm.max_converge_count:
                    swarm.inject_noise()

                # update the swarm convergence count
                swarm.update_converge_count()

            # swarm plot data
            # swarm_xdata = [p.centroid[0] for p in swarm.swarm]
            # swarm_ydata = [p.centroid[1] for p in swarm.swarm]
            # best_swarm_xdata = [p.centroid[0] for p in swarm.swarm_best]
            # best_swarm_ydata = [p.centroid[1] for p in swarm.swarm_best]
            
            '''Residual PLOT'''
            # plotter.update_residual_plot(iterations, residuals) # 
            
            '''Profile PLOT'''
            # plotter.update_profile_plot(xprofile, yprofile, xctrl, yctrl)

            '''Swarm PLOT'''
            # plotter.update_swarm_plot(swarm_xdata, swarm_ydata, best_swarm_xdata, best_swarm_ydata)
        
            '''Performance PLOT'''
            # plotter.update_performance_plot(alpha, coeff, s_alpha, s_coeff) 

            # plotter.refresh() # refresh the plot    

            # save the swarm
            # swarm.save_swarm('swarm.pkl')

            if best_particle.fitness < target_fitness:
                logger.info(f">> Best fitness: {best_particle.fitness} ▼ - Target fitness: {target_fitness} ▲")
                # print("\033[96m {}\033[00m".format(f"[*] Best fitness: {best_particle.fitness} ▼ - Target fitness: {target_fitness} ▲"))
            else:
                logger.critical(">> Target fitness has been breached!")
                logger.critical(f">> Best fitness: {best_particle.fitness} ▲ - Target fitness: {target_fitness} ▼")
                # print("\033[92m {}\033[00m".format(f"[*] Best fitness: {best_particle.fitness} ▲ - Target fitness: {target_fitness} ▼"))

            # best_particle.save_profile()
            # best_particle.save_control_points()

            swarm.update_solution_model()
            # plot the residuals
            # plotter.save_plot()
            # kill any remaining xfoil processes
            if checkIfProcessRunning('xfoil'):
                os.system('killall xfoil')
                logger.info(">> Killing xfoil processes...")

            print("\n")
            

        except Exception as e:
            # exc_type, exc_obj, exc_tb = sys.exc_info()
            # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # logger.error(exc_type, fname, exc_tb.tb_lineno)
            logger.error(f'An exception occured...{e}')
            os.system('killall xfoil')
            
            # print("\033[91m {}\033[00m".format(f'Aborting optimization...{e}'))
            # print("\033[96m {}\033[00m".format(f'Optimization took: {timeit.default_timer() - start_time} seconds'))
            # logger.info(f'Optimization took: {timeit.default_timer() - start_time} seconds')
            continue

    # save the plot
    # plotter.save_plot()
    # save the swarm
    # save the best particle
    g_best.save_control_points()
    # swarm.save_swarm('swarm.pkl')
    logger.info(f'[*] Optimization took: {timeit.default_timer() - start_time} seconds...')
    # print(f'[*] Optimization took: {timeit.default_timer() - start_time} seconds...')
    
        
        
        
    









