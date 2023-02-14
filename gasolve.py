import numpy as np
import pandas as pd
import time
import random
import os.path
from os import path
from subprocess import Popen, PIPE, TimeoutExpired
import random
from random import randrange
import concurrent.futures
import logging
import uuid
import psutil
import timeit
from db.models import Ga, GaResults
import copy
import sys


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


def measure_time(func):
    def wrapper(*args, **kwargs):

        start_time = timeit.default_timer()
        func(*args, **kwargs)
        logger.info(
            f'Took {(timeit.default_timer() - start_time)} seconds to solve..')

    return wrapper


def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def get_control_points(binary_array):
    """
    This function takes a 192 binary array and returns a 1D 24 control points list which house the x,y control point coords 
    The control points array is made up of 12 X-coodinate control points followed by 12 Y-coordinate control points.
    The first six coordinates of each set represent the upper surface control points
    The list order is 0,1,2,3,4,5,11,10,9,8,7,6,12,13,14,15,16,17,23,22,21,20,19,18
    """
    value_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # num = np.dot(value_array, 2**np.arange(len(value_array)-1, -1, -1))
    # decimal = num/100

    # decoded_coords = np.empty([12, 2], dtype = float)
    j = 0
    k = 8
    for i in range(0, 12):
        value_array[i] = (np.dot(binary_array[j:k], 2 **
                          np.arange(len(binary_array[j:k])-1, -1, -1)))/100
    #     print(i,j,k)
        j += 8
        k += 8

    n = 96
    o = 104
    for m in range(12, 24):
        value_array[m] = (np.dot(binary_array[n:o], 2 **
                          np.arange(len(binary_array[n:o])-1, -1, -1)))
#         print(m,n,o)
        n += 8
        o += 8
    return value_array


def generate_Bezier_Curve(val_a, val_b, val_c, val_d, val_e, val_f, val_g, val_h, val_u):
    """
    Function to generate Bezier Curves using the provided constants - 8 constants for 7th Order
    """
    a = float(val_a)
    b = float(val_b)
    c = float(val_c)
    d = float(val_d)
    e = float(val_e)
    f = float(val_f)
    g = float(val_g)
    h = float(val_h)
    u = float(val_u)

    member0 = ((1.0-u)**7.0)*a
    member1 = ((1.0-u)**6.0)*(u**1)*b*7.0
    member2 = ((1.0-u)**5.0)*(u**2)*c*21.0
    member3 = ((1.0-u)**4.0)*(u**3)*d*35.0
    member4 = ((1.0-u)**3.0)*(u**4)*e*35.0
    member5 = ((1.0-u)**2.0)*(u**5)*f*21.0
    member6 = ((1.0-u)**1.0)*(u**6)*g*7.0
    member7 = ((1.0-u)**0.0)*(u**7)*h

    P = member0 + member1 + member2 + member3 + \
        member4 + member5 + member6 + member7

    return P


def solve(uid, val):
    fitness = 0.0
    # run xfoil simulation on the generated airfoil profile
    # self.save_airfoil_coordinates() # save the airfoil profile to a txt file to be used by xfoil

    def load_results_from_file(uid):
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
        if not path.exists(f'outputs/{uid}.txt'):
            return results
        with open(f'outputs/{uid}.txt', 'r') as f:
            lines = f.readlines()
            # remove empty lines
            lines = [line for line in lines if line.strip() != '']

        # skip the first 7 lines of the file and get the first 6 values
        results_list = [[float(x) for x in line.split()[:7]]
                        for line in lines[7:]]
        for result in results_list:
            results['alpha'].append(result[0])
            results['cl'].append(result[1])
            results['cd'].append(result[2])
            results['cdp'].append(result[3])
            results['cm'].append(result[4])

        return results

    def calculate_fitness(res: dict) -> float:
        fitness = 1.0
        if res['cd'] == []:
            return fitness
        if len(res['cd']) < 4:
            return fitness

        if any(res['cd']) < 0.0 or any(res['cl']) < 0.0:
            logger.critical("Negative lift or drag coefficient found!")
            return fitness

        # replace all 0.0 values with 0.0001 in the cd list
        # res['cd'] = [0.0001 if x == 0.0 else x for x in res['cd']]
        # avg_cl = sum(res['cl'])/len(res['cl']) # average lift coefficient
        # avg_cd = sum(res['cd'])/len(res['cd'])
        # fitness += avg_cl/avg_cd
        if any(x == 0.0 for x in res['cd']):
            logger.critical("Zero drag coefficient found!")
            return fitness
        coeff = [cl/cd for cl, cd in zip(res['cl'], res['cd'])]
        return sum(coeff)

    np.savetxt(f'airfoils/{uid}.txt', val)

    try:
        # direct xfoil to xvfb
        # xvfb = Popen(["Xvfb", ":1", "-screen", "0", "1024x768x24"], close_fds=True)
        # run xfoil using subprocess
        p = Popen(["xfoil", "-s" "-n", "-p"], shell=True, stdin=PIPE,
                  stdout=PIPE, stderr=PIPE, close_fds=True)  # run xfoil

        # send commands to xfoil
        p.stdin.write(b'load\n')
        p.stdin.write(f'airfoils/{uid}.txt\n'.encode())
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
        p.stdin.write(f'outputs/{uid}.txt\n'.encode())
        p.stdin.write(b'\n')
        p.stdin.write(b'\n')
        p.stdin.write(b'oper\n')
        p.stdin.write(b'aseq\n')
        p.stdin.write(b'0\n')
        p.stdin.write(b'6\n')
        p.stdin.write(b'2\n')
        p.stdin.write(b'\n')
        p.stdin.write(b'!\n')
        p.stdin.write(b'\n')
        p.stdin.write(b'\n')
        p.stdin.write(b'quit\n')
        # implement a try catch block to handle the timeout error

        # run the process and wait only 5 seconds for it to finish, then kill it
        output, error = p.communicate(timeout=10)
        if p.poll() is None:
            p.kill()

        # xvfb.kill()
    except TimeoutExpired:
        logger.warning(f'   >> Xfoil process timed out for particle {uid}')
        p.kill()
        # xvfb.kill()
        # clean up the files
        if path.exists(f'airfoils/{uid}.txt'):
            os.remove(f'airfoils/{uid}.txt')

        if path.exists(f'outputs/{uid}.txt'):
            os.remove(f'outputs/{uid}.txt')

        return fitness

    res = load_results_from_file(uid)
    # clean up the files
    if path.exists(f'airfoils/{uid}.txt'):
        os.remove(f'airfoils/{uid}.txt')
    # os.remove(f'airfoils/{uid}.txt')
    if path.exists(f'outputs/{uid}.txt'):
        os.remove(f'outputs/{uid}.txt')
    # os.remove(f'outputs/{uid}.txt')

    fitness = calculate_fitness(res)
    if fitness  > 1000:
        logger.critical(f">> Fitness is too high: {fitness}")
        fitness = 1.0

    return fitness


def get_solution_coordinates(solution, uid):
    """
    This function takes a solution and returns its airfoil coordinates. num_coords specifies how many x,y pairs you prefer.
    """
    # Get airfoil coordinates from possible solution control points
    Value = solution
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

    xU = []
    yU = []
    xL = []
    yL = []

#     data = {'UpperX': [],'UpperY': []}

#     dataframe = pd.DataFrame(data)
    num_coords = 40
    u_min_upper = 1/(num_coords)
    u_max_upper = 1 + u_min_upper

    # Convert upper curve control points to airfoil coordinates and append to the dataframe
    for i in np.arange(0, u_max_upper, u_min_upper):
        u = float(i)
        x_val = generate_Bezier_Curve(
            Ux1, Ux2, Ux3, Ux4, Ux5, Ux6, Ux7, Ux8, u)
        y_val = generate_Bezier_Curve(
            Uy1, Uy2, Uy3, Uy4, Uy5, Uy6, Uy7, Uy8, u) / 1000
        xU.append(x_val)
        yU.append(y_val)
#     dataframe['UpperX'] = xU
#     dataframe['UpperY'] = yU

    # Convert lower curve control points to airfoil coordinates and append to the dataframe
    for i in np.arange(1.0, -u_min_upper, -u_min_upper):
        u = float(i)
        x_val = generate_Bezier_Curve(
            Lx1, Lx2, Lx3, Lx4, Lx5, Lx6, Lx7, Lx8, u)
        y_val = generate_Bezier_Curve(
            Ly1, Ly2, Ly3, Ly4, Ly5, Ly6, Ly7, Ly8, u) / 1000
        xL.append(x_val)
        yL.append(y_val)

    x = pd.concat([pd.Series(xU), pd.Series(xL)], ignore_index=True)
    y = pd.concat([pd.Series(yU), pd.Series(yL)], ignore_index=True)

    df = pd.DataFrame({"X": x, "Y": y})
    # np.savetxt(r'af.dat', df.values, fmt='%1.6f')

    fitness = solve(uid, df.values)

    return fitness


class ga():
    """
    This class runs a genetic algorithm
    """

    def __init__(self, Generation, Chromosome_Length, Population_Size, Mutation_Rate, Crossover_Rate, varbound, seed, function=None):
        self.number_of_generations = Generation
        self.length_of_chromosome = Chromosome_Length
        self.size_of_population = Population_Size
        self.mutation_rate = Mutation_Rate
        self.crossover_rate = Crossover_Rate
        self.varbound = varbound
        self.seed = seed
        self.fitness_func = function
#         self.maximum_allowable_thickness = Maximum_Thickness
#         self.minimum_allowable_thickness = Minimum_Thickness
#         self.maximum_allowable_camber = Maximum_Camber
#         self.minimum_allowable_camber = Minimum_Camber
#         self.bezier_control_point_parameter_check = 0

    def generate_random_initialization(self):
        """
        Function to create the first generation of solutions
        """
#         current_generation = np.random.randint(2, size=(size_of_population,length_of_chromosome))
        current_generation = np.random.randint(
            (self.varbound[0][1])+1, size=(self.size_of_population, self.varbound.shape[0]))
        current_generation[0] = self.seed
        return current_generation

    # @measure_time
    def calculate_generation_fitness(self, generation_array):
        """
        This function takes a generation array and returns the generation results
        """
        generation_results = np.empty((0, 1))

        # def uf(x):
        #     uid = str(uuid.uuid4().hex) # unique id for filenames
        #     self.fitness = self.fitness_func(x, uid)
        #     generation_results = np.append(generation_results,self.fitness)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.fitness_func, x, str(
                uuid.uuid4().hex)) for x in generation_array]
            concurrent.futures.wait(futures)

            for future in futures:
                # for future in concurrent.futures.as_completed(futures):
                fitness = future.result()
                generation_results = np.append(generation_results, fitness)

        # for x in generation_array:
        #     uid = str(uuid.uuid4().hex) # unique id for filenames
        #     self.fitness = self.fitness_func(x, uid)
        #     generation_results = np.append(generation_results,self.fitness)

        return generation_results

    def calculate_fitness_weight_distribution(self, generation_results):
        """
        Function to calculate the weight distributions of each individual's fitness against the sum of all individual's fitness
        """

        # weight_distribution = np.zeros((self.size_of_population,))
        self.generation_results = generation_results
        total_fitness = self.generation_results.sum()
        weight_distribution = self.generation_results/total_fitness
        return weight_distribution

    def selection(self, current_generation, weighted_results):
        """
        Function to make a copy of the current_gen and carry out Roulette Wheel Selection based on the provided weighted_fitness_results
        """
        self.current_generation = current_generation
        self.weights = weighted_results
        new_pool = np.copy(self.current_generation)

        # print(weighted_results)
        for i in range(0, len(self.current_generation)):
            r = np.random.rand()
            acc = 0
            idx = -1
            while acc < r:
                idx += 1
                acc += self.weights[idx]
            new_pool[i] = self.current_generation[idx]

        return new_pool

    def crossover(self, new_pool):
        """
        Function to perfom crossover on an array that has gone through Roulette Wheel Selection
        """
        self.new_pool = new_pool
        parent_pool = np.copy(self.new_pool)
        offspring_pool = np.copy(self.new_pool)
        CR = self.crossover_rate

        # The range starts from 1 to preserve best solution in zeroth position of the current_gen - pop[0]
        # The range ends at number_of_generations minus one (e.g. 5 sols means solution pop[1] and pop[3] may have crossover hence our range ends at 4 i.e. [5-1])

        for i in range(1, len(self.current_generation)-1, 2):
            r = np.random.rand()
    #             print("r = ", r)
            if r <= CR:
                # for each divider i.e. 3 divisions for 4 genes, 4 divisions for 5 genes, X-1 divisions for X genes
                crosspos = randrange(self.length_of_chromosome-1)
    #                 print("crossover_position = ", crosspos)
                for j in range(0, crosspos+1):  # from 0 to crossover position
                    offspring_pool[i][j] = parent_pool[i][j]
                    offspring_pool[i+1][j] = parent_pool[i+1][j]
                # from crossover position to end of chromosome
                for k in range(crosspos+1, self.length_of_chromosome):
                    offspring_pool[i][k] = parent_pool[i+1][k]
                    offspring_pool[i+1][k] = parent_pool[i][k]

        return offspring_pool

    def mutation(self, offspring_pool):
        """
        Function to perform mutation on an array  that has gone through Roulette Wheel Selection and Crossover
        """
        self.offspring_pool = offspring_pool
        mutated_pool = np.copy(self.offspring_pool)

        MR = self.mutation_rate

        for i in range(1, self.size_of_population):
            r1 = np.random.rand()
#                 print("r1 = ", r1 )
            if r1 <= MR:
                for j in range(0, self.length_of_chromosome):
                    r2 = np.random.rand()
#                         print("r2 = ", r2)
                    if r2 <= MR:
                        #                             print("mutation occured at Individual No: ", i, " and Gene: ", j)
                        mutated_pool[i][j] = 1 - (mutated_pool[i][j])

        return mutated_pool

    def get_best_solution_index(self, fitness_results):
        """
        Function to get the best solution from each generation
        """
        self.fitness_results = fitness_results
        top_solution = self.fitness_results.max()
        result = np.where(self.fitness_results == top_solution)
        top_solution_position = result[0][0]

        return top_solution_position

    def get_second_best_solution_index(self, fitness_results):
        """
        Function to get the second best solution from each generation
        """
        self.fitness_results = fitness_results
        copy = np.copy(self.fitness_results)
        copy[::-1].sort()
        second = 0.0
        for i in range(0,len(copy)):
            if copy[i] < copy.max():
                second = copy[i]
                break
             # sorts fitness's in ascending order
        second_best_fitness = second  # get value second best fitness
        # find position index for second best fitness
        result = np.where(self.fitness_results == second_best_fitness)
        second_best_position = result[0][0]  # get integer value of index

        return second_best_position

    def get_best_solution(self, current_gen, top_solution_position):
        """
        Function to create an array which holds the best solution
        """
        self.current_gen = current_gen
        self.top_solution_position = top_solution_position
        best_solution = np.copy(self.current_gen[self.top_solution_position])

        return best_solution

    def get_second_best_solution(self, current_gen, second_best_solution_position):
        """
        Fuction to get second best fitness value
        """
        self.current_gen = current_gen
        self.second_best_solution_position = second_best_solution_position
        second_best_solution = np.copy(
            self.current_gen[self.second_best_solution_position])

        return second_best_solution

    def seed_next_generation(self, best_solution, second_best_solution, current_gen):
        """
        Function to seed the next generation with the best solution from the previous generation
        """
        self.current_gen = current_gen
        self.best_solution = best_solution
        self.second_best_solution = second_best_solution
        new_gen = np.copy(self.current_gen)
        seed_gen = np.copy(self.current_gen)
        new_gen[0] = self.best_solution
        new_gen[1] = self.second_best_solution

        return new_gen

    def store_info(self, iteration, best_fitness):
        """
        This function keeps appending data to the store data list
        """
        info = np.array([iteration, best_fitness])
        self.stored_info = np.append(self.stored_info, info)
        return self.stored_info

    def plot_info(self, plot_data):
        """
        This plots out the current iteration and best_fitness
        """
#         iteration_info = np.array([0,5,1,11,2,12])
        import matplotlib.pyplot as plt

        plt.plot(plot_data[:, 0], plot_data[:, 1])
        plt.ylabel('fitness')
        plt.xlabel('iterations')
        plt.show()

    def run(self):
        start = time.time()
        current_gen = self.generate_random_initialization()
        current_gen[0] = self.seed
        self.stored_info = np.empty((0, 1))

        ga = Ga.objects.create()

        global_best_array = None
        global_best_fitness = 0.0
        target_fitness = 533.46

        for i in range(0, self.number_of_generations):
            logger.info(
                f'[+] Iteration {i+1} of {self.number_of_generations}..')
            try:
                # calculate fitness for entire generation
                fitness_results = self.calculate_generation_fitness(
                    current_gen)
                # calculate the weighted average according to fitness
                weighted_results = self.calculate_fitness_weight_distribution(
                    fitness_results)
                # print(weighted_results)
                # apply parent selection using Roulette Wheel methods based on weighted fitnesses
                new_pool = self.selection(current_gen, weighted_results)
                # apply crossover if random injection occurs
                xover_offspring = self.crossover(new_pool)
                # apply mutation if random injection occurs
                m_pool = self.mutation(xover_offspring)
                # get the best solution position
                best_solution_position = self.get_best_solution_index(
                    fitness_results)
                # get the best solution fitness
                best_solution = self.get_best_solution(
                    current_gen, best_solution_position)
                # get the second best solution position
                second_best_solution_position = self.get_second_best_solution_index(
                    fitness_results)
                # get the second best solution fitness
                second_best_solution = self.get_second_best_solution(
                    current_gen, second_best_solution_position)
                # store fitness information for plotting
                self.store_info(i, fitness_results.max())
                # seed next generation with best and second best seeds
                current_gen = self.seed_next_generation(
                    best_solution, second_best_solution, m_pool)

                if i == 0:
                    global_best_array = copy.deepcopy(
                        current_gen[best_solution_position])
                    global_best_fitness = fitness_results.max()
                else:
                    if fitness_results.max() > global_best_fitness and fitness_results.max() < 1000:
                        global_best_array = copy.deepcopy(
                            current_gen[best_solution_position])
                        global_best_fitness = fitness_results.max()

                dict_res = {'local': current_gen.tolist(
                ), 'global': global_best_array.tolist()}
                GaResults.objects.create(simulation=ga, solution=dict_res)

                logger.info(
                    f">> Global fitness: {global_best_fitness} || Local Fitness: {fitness_results.max()}...")
                
                diff = target_fitness - global_best_fitness
                if diff > 0:
                    logger.info(f" >> Distance to target: {diff}")
                else:
                    logger.critical(
                        f" >> Target has been exceeded by {abs(diff)} points..")

            except Exception as exc:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(f">> Exception found: {exc}")
                logger.error(exc_type)
                logger.error(exc_obj)
                logger.error(exc_tb)
                continue

            # clean up
            os.system('rm -rf outputs/*')
            os.system('rm -rf airfoils/*')

        end = time.time()
        total_time = end - start
        elapsed_time = round(total_time, 2)
        logger.info(f">> Time Elapsed: {elapsed_time} seconds")

        # plot iteration no. vs fitness after simulation ends
        # plot_data = self.stored_info.reshape(-1,2)
        # self.plot_info(plot_data)
        return current_gen


def f(X, uid):
    return get_solution_coordinates(get_control_points(X), uid)


varbound = np.random.randint(2, size=(192, 2))
# vartype=np.array([['real'],['real'],['real'],['real'],['real']])

max_num_iteration = 20000
chromosome_length = 192
population_size = 5
mutation_probability = 0.2
crossover_probability = 0.8

# seed = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
#        1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
#        0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
#        0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
#        1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
#        0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
#        1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1])

# seed = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1])

# seed = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# seed = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

seed = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1,
                1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

# 'elit_ratio': 0.01,
# 'parents_portion': 0.3,
# 'crossover_type':'uniform',
# 'max_iteration_without_improv':None

if not checkIfProcessRunning('Xvfb'):
    xvfb = Popen(["Xvfb", ":1", "-screen", "0", "1024x768x24"],
                 close_fds=True)  # start virtual display
    logger.info(">> Xvfb started")
else:
    logger.info(">> Xvfb already running")

os.environ['DISPLAY'] = ':1'  # set display

module = ga(max_num_iteration, chromosome_length, population_size,
            mutation_probability, crossover_probability, varbound, seed, function=f)

result = module.run()
