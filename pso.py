import numpy as np
import multiprocessing as mp
import uuid
from dataclasses import dataclass
from Bezier import Bezier
from os import path
from subprocess import Popen, PIPE, TimeoutExpired
import os
import logging
import copy
from db.models import Simulation, SimulationResults
import psutil

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

@dataclass
class Particle:
    position: list
    velocity: list = None
    id: str = None
    fitness: float = 0.0

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4().hex)
        self.position = np.array(self.position).flatten()
        self.velocity = np.random.uniform(0.0, 0.0, size=self.position.shape)

    def generate_curve(self):
        # generate the curve from the control points
        points = np.linspace(0, 1, 82)
        ctrl = np.array(self.position).reshape(-1, 2)
        curve = Bezier.Curve(points, ctrl)
        return curve

    def save_control_points(self):
        with open(f'best-solution/{self.id}.txt', 'w') as f:
            array = np.array(self.generate_curve())
            np.savetxt(f, array, fmt='%.3f')

    def save_airfoil(self) -> None:
        # save the airfoil profile to a txt file
        with open(f'airfoils/{self.id}.txt', 'w') as f:
            array = np.array(self.generate_curve())
            np.savetxt(f, array, fmt='%.3f')

    @classmethod
    def to_class(cls, *args, **kwargs):
        return cls(*args, **kwargs)

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
        if not path.exists(f'outputs/{self.id}.txt'):
            return results
        with open(f'outputs/{self.id}.txt', 'r') as f:
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

        return results

    def calculate_fitness(self, res:dict) -> float:
        fitness = 0
        if res['cd'] == []:
            return fitness
        if len(res['cd']) < 7:
            return fitness

        # avg_cl = sum(res['cl'])/len(res['cl']) # average lift coefficient
        # avg_cd = sum(res['cd'])/len(res['cd'])
        # fitness += avg_cl/avg_cd
        coeff = [cl/cd for cl, cd in zip(res['cl'], res['cd'])]
        return sum(coeff)
        
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
            p.stdin.write(f'airfoils/{self.id}.txt\n'.encode())
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
            p.stdin.write(f'outputs/{self.id}.txt\n'.encode())
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
            output, error = p.communicate(timeout=10) #
            if p.poll() is None:
                p.kill()

            # xvfb.kill()
        except TimeoutExpired:
            logger.warning(f'   >> Xfoil process timed out for particle {self.id}')
            p.kill()
            # xvfb.kill()
            # clean up the files
            if path.exists(f'airfoils/{self.id}.txt'):
                os.remove(f'airfoils/{self.id}.txt')
                
            if path.exists(f'outputs/{self.id}.txt'):
                os.remove(f'outputs/{self.id}.txt')
            
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
        if path.exists(f'airfoils/{self.id}.txt'):
            os.remove(f'airfoils/{self.id}.txt')
        # os.remove(f'airfoils/{self.id}.txt')
        if path.exists(f'outputs/{self.id}.txt'):
            os.remove(f'outputs/{self.id}.txt')
        # os.remove(f'outputs/{self.id}.txt')
        
        fitness = self.calculate_fitness(res)

        return fitness, res

    def __reduce__(self):
        return (Particle, (self.position, self.velocity, self.id, self.fitness,))

    @classmethod
    def from_dict(cls, data):
        cls.position = data['position']
        cls.velocity = data['velocity']
        cls.id = data['id'] or str(uuid.uuid4().hex)
        cls.fitness = data['fitness']
        return cls(cls.position, cls.velocity, cls.id, cls.fitness)
    

total_particles = 27 # number of particles
initial_control_points = np.loadtxt('best-solution/control_points.txt').tolist() # initial control points
w = 0.5 # inertia weight
c1 = 2.0 # cognitive weight - local best
c2 = 0.7 # social weight - global best
iterations = 5000 # total number of iterations

def initialize_solution(current_solutions, best_solutions, particle):
    
    # initialize the current solutions and best solutions
    current_solutions[particle.id] = {
        'position': particle.position,
        'velocity': particle.velocity,
        'fitness': particle.fitness,
    }

    best_solutions[particle.id] = {
        'position': particle.position,
        'velocity': particle.velocity,
        'fitness': particle.fitness,
    }

def update_solution(lock, current_solutions, best_solutions, id, i):
    '''
        The method updates the current solutions and best solutions
        for each particle. The current solutions are updated by the
        new position and velocity of the particle. The best solutions
        are updated by the current solutions if the current solutions
        are better than the best solutions.

        Steps:
            1. Update the velocity of the particle
            2. Update the position of the particle
            3. Update the fitness of the particle
            4. Update the current solutions
            5. Update the best solutions
    '''

    
    particle = Particle.from_dict(data={
        'position': current_solutions[id]['position'],
        'velocity': current_solutions[id]['velocity'],
        'id': id,
        'fitness': current_solutions[id]['fitness'],
    })
   


    g_best = 0.0
    g_id = None

    
    for key, item in best_solutions.items():
        if key != particle.id and item['fitness'] >= g_best:
            g_best = item['fitness']
            g_id = key

    if g_id is None:
        raise ValueError('No global best found')
    
    global w
    global c1
    global c2

    # generate random array of numbers of the same shape as the position
    r1 = np.random.uniform(0.0, 0.0001, particle.position.shape)
    r2 = np.random.uniform(0.0, 0.0001, particle.position.shape)


    # update the velocity of the particle
    if i > 0:
        cognitive = c1*r1*(best_solutions[particle.id]['position'])

        social = c2*r2*(best_solutions[g_id]['position'] - particle.position)
        inertia = w*particle.velocity

        particle.velocity = inertia + cognitive + social
        # particle.velocity = w*particle.velocity + c1*r1*(best_solutions[particle.id]['position'] - particle.position) + c2*r2*(best_solutions[g_id]['position'] - particle.position)

    # update the position of the particle
    particle.position = particle.position + particle.velocity

    # fix the position if it is out of bounds
    global seed_particle
    particle.position[0:10] = copy.deepcopy(seed_particle.position[0:10])
    particle.position[-10:] = copy.deepcopy(seed_particle.position[-10:])

    # update the fitness of the particle
    particle.fitness, _ = particle.solve()

    # update the current solutions
    # lock the variable so that it is not updated by another process at the same time
    
    current_solutions[particle.id] = {
        'position': particle.position,
        'velocity': particle.velocity,
        'fitness': particle.fitness,
    }
    

    
    # update the best solutions
    if particle.fitness > best_solutions[particle.id]['fitness']  and particle.fitness < 800:
        best_solutions[particle.id] = {
            'position': particle.position,
            'velocity': particle.velocity,
            'fitness': particle.fitness,
        }
    
def solution_serializer(solutions) -> dict:
    '''
        The method serializes the solution to a dictionary
    '''

    return [{
        'position': solution['position'].tolist(),
        'velocity': solution['velocity'].tolist(),
        'fitness': solution['fitness'],
    } for solution in solutions.values()]

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


if __name__ == "__main__":

    if not checkIfProcessRunning('Xvfb'):
        xvfb = Popen(["Xvfb", ":1", "-screen", "0", "1024x768x24"], close_fds=True) # start virtual display
        logger.info(">> Xvfb started")
    else:
        logger.info(">> Xvfb already running")
    
    # os.environ['DISPLAY'] = ':1' # set display

    lock = mp.Lock() # create a lock for the processes to use so that they do not update the same variable at the same time
    
    with mp.Manager() as manager:
        simulation = Simulation.objects.create()
        target_fitness = sum([
            115.922619,
            147.4285714,
            148.7848101,
            121.3309025,
            92.73074475,
            73.5483871,
            56.83816014,
            # 41.41044061,
            # 28.41207988,
            # 18.57112069
            ])

        global seed_particle
        seed_particle = Particle(initial_control_points)
        fitness, _ = seed_particle.solve()
        
        logger.info(f" >> Seed particle fitness: {fitness}..")

        # generate a list of particles
        particles = [Particle(initial_control_points) for _ in range(total_particles)]

        current_solutions = manager.dict()
        bests_solutions = manager.dict()
        g_fitness = manager.Value('d', 0.0) # global best fitness value

        # initilize the current solutions and best solutions
        [initialize_solution(current_solutions, bests_solutions, particle) for particle in particles]

        for i in range(iterations):
            logger.info(f"[+] Iteration {i+1}... of {iterations}")
            # update the current solutions and best solutions
            processes = [mp.Process(target=update_solution, args=(lock, current_solutions, bests_solutions, id, i)) for id in current_solutions.keys()]

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            best_current_fitness = max([item['fitness'] for item in current_solutions.values()])
            
            best_global_fitness = max([item['fitness'] for item in bests_solutions.values()])

            if best_global_fitness > g_fitness.value:
                g_fitness.value = best_global_fitness
                logger.info(f" >> New global best fitness: {g_fitness.value}")
            
            if best_global_fitness >= target_fitness:
                logger.critical(f" >> Target fitness reached: {best_global_fitness}")

            if best_current_fitness > best_global_fitness:
                logger.critical(f" >> Best current fitness is greater than best global fitness: {best_current_fitness} > {best_global_fitness}")
            else:
                logger.info(f" >> Best current fitness: {best_current_fitness} | Best global fitness: {best_global_fitness}")

            logger.info(f" ** Difference: {target_fitness-best_global_fitness} **")
            # clean up the outputs and airfoils folders
            os.system('rm -rf outputs/*')
            os.system('rm -rf airfoils/*')
     
            simulation_result = SimulationResults.objects.create(
                simulation=simulation,
                current_solution={'solutions': solution_serializer(current_solutions), 'fitness': best_current_fitness},
                best_solution={'solutions': solution_serializer(bests_solutions), 'fitness': best_global_fitness},
            )

            # save the best solutions and current solutions to the model


            

        
