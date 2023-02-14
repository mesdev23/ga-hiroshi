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
from multiprocessing import Process, Value, Array, Manager
from multiprocessing.managers import BaseManager
import multiprocessing
from time import sleep

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

        # evaluate additional constraints
        for constraint in additional_constraints:
            constraint(self) # the constraint function will evalute the control points and raise an error if the constraint is not satisfied

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

        results_list = [[float(x) for x in line.split()[:9]] for line in lines[7:]]
        for result in results_list:
            results['alpha'].append(result[0])
            results['cl'].append(result[1])
            results['cd'].append(result[2])
            results['cdp'].append(result[3])
            results['cm'].append(result[4])
            results['xtr_top'].append(result[5])
            results['xtr_bot'].append(result[6])
            results['itr_top'].append(result[7])
            results['itr_bot'].append(result[8])

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

    def mse(self):
        # calculates the mean square error
        return

    def calculate_fitness(self, res:dict) -> float:
        fitness = 0.0
        if res['cd'] == []:
            return fitness
        if len(res['cd']) < 9:
            return fitness

        avg_cl = sum(res['cl'])/len(res['cl']) # average lift coefficient
        avg_cd = sum(res['cd'])/len(res['cd'])
        fitness += avg_cl/avg_cd
        return fitness


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
            p.stdin.write(b'18\n')
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
        self.fitness = fitness
        self.results = res
        
        # return fitness,res

    def __repr__(self):
        return f'Particle: {self.particle_id}'

# @dataclass
# class Swarm:
#     seed_particle: Particle
#     inertia: float = 0.0
#     cognitive: float = 1.49618
#     social: float = 1.49618
#     total_particles: int = 10
#     best_solutions: List[Particle] = None
#     current_solutions: List[Particle] = None

#     def __post_init__(self):
        
#         particles = self.seed_particle.control_points + np.random.uniform(0.0,0.0,(self.total_particles, self.seed_particle.shape[0]))*np.random.uniform(0.0,0.0, self.seed_particle.shape) # initialize the particles

#         # ENFORCE CONSTRAINTS, the trailing section must be remain the same
#         particles[:,0:2] = copy.deepcopy(self.seed_particle.control_points[0:2]) # enforce constraints
#         particles[:,-2:] = copy.deepcopy(self.seed_particle.control_points[-2:])

#         self.current_solutions = [Particle(control_points=particle) for particle in particles]
#         self.best_solutions = copy.deepcopy(self.current_solutions)


class Swarm(Process):


    def __init__(self, seed_particle:Particle, total_particles:int, iterations:int):
        Process.__init__(self)

        self.seed_particle = seed_particle
        particles = self.seed_particle.control_points + np.random.uniform(0.0,0.0,(total_particles, self.seed_particle.shape[0]))*np.random.uniform(0.0,0.0, self.seed_particle.shape) # initialize the particles

        # ENFORCE CONSTRAINTS, the trailing section must be remain the same
        particles[:,0:2] = copy.deepcopy(self.seed_particle.control_points[0:2]) # enforce constraints
        particles[:,-2:] = copy.deepcopy(self.seed_particle.control_points[-2:])
 
        self.current_solutions = Array('i', )
        self.best_solutions = copy.deepcopy(self.current_solutions)
        self.iterations = iterations
        self.inertia = Value('i', 0) # initialize the inertia as a shared variable of type int

    def update_particle_velocity(self, particle, local_best, global_best):
        # update the velocity of the particle
        particle.velocity = self.inertia.value*particle.velocity + \
            np.random.uniform(0.0,0.0, particle.shape)*self.cognitive*(local_best.control_points - particle.control_points) + \
            np.random.uniform(0.0,0.0, particle.shape)*self.social*(global_best.control_points - particle.control_points)

    def update_particle_position(self, particle):
        # update the position of the particle
        particle.control_points = particle.control_points + particle.velocity

        # ENFORCE CONSTRAINTS, the trailing section must be remain the same
        particle.control_points[0:2] = copy.deepcopy(self.seed_particle.control_points[0:2])
        particle.control_points[-2:] = copy.deepcopy(self.seed_particle.control_points[-2:])
     
    def update_particle_fitness(self, particle):
        # update the fitness of the particle
        particle.fitness, particle.results = particle.calculate_fitness()

    def update_swarm(self):
        self.inertia.value += 1 # increase the inertia

    def run(self):
        
        processes = [Process(target=self.update_swarm) for particle in self.current_solutions]

        for p in processes:
            p.start()
        
        for p in processes:
            p.join()

        print(self.inertia.value)
        
        

if __name__ == '__main__':
    
    if not checkIfProcessRunning('Xvfb'):
        xvfb = Popen(["Xvfb", ":1", "-screen", "0", "1024x768x24"], close_fds=True) # start virtual display
        os.environ['DISPLAY'] = ':1' # set display
        logger.info(">> Xvfb started")
    else:
        logger.info(">> Xvfb already running")


    if path.exists('best-solution/control_points.txt'):
        control_points = np.loadtxt('best-solution/control_points.txt')
        logger.info(">> Control points loaded from previous run")
    else:
        control_points = np.loadtxt('SG6043.txt')
        logger.info(">> Control points loaded from SG6043.txt")


    control_points = np.array(control_points).flatten()

    seed_particle = Particle(control_points)
    temp = copy.deepcopy(seed_particle)
    # seed_fitness, seed_res = temp.solve() # solve the seed particle
    target_fitness = 84.5
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
    
    # seed particle results
    # s_alpha = seed_res['alpha']
    # s_cd = seed_res['cd']
    # s_cl = seed_res['cl']
    # s_coeff = sg_coeff

    
    swarm = Swarm(seed_particle=seed_particle, total_particles=5, iterations=5)
    swarm.run()


   

    
        
        
        
    









