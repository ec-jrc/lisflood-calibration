import os
import numpy as np
import pandas
import multiprocessing as mp
import time

# deap related packages
import array
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from scipy.stats import ttest_ind_from_stats

class LockManager():
    """
    A class to manage locks and counters in a multiprocessing environment.

    Parameters
    ----------
    num_cpus : int
        Number of CPU cores to be used in multiprocessing.

    Attributes
    ----------
    counters : dict
        A dictionary containing multiprocessing values for run and generation counters.
    lock : multiprocessing.Lock
        A lock to ensure thread-safe operations.
    num_cpus : int
        Number of CPU cores.
    """

    def __init__(self, num_cpus):

        mgr = mp.Manager()
        self.counters = {}
        self.counters['run'] = mgr.Value('i', -1)
        self.counters['gen'] = mgr.Value('i', -1)

        self.lock = mgr.Lock()

        self.num_cpus = num_cpus

    def increment_gen(self):
        self._increment('gen')
        self._set('run', 0)

    def increment_run(self):
        return self._increment('run')

    def set_gen(self, value):
        self._set('gen', value)
        self._set('run', 0)

    def set_run(self, value):
        return self._set('run', value)

    def get_gen(self):
        return self._value('gen')

    def get_run(self):
        return self._value('run')

    def _increment(self, name):
        with self.lock:
            self.counters[name].value += 1
            value = self.counters[name].value
        return value

    def _set(self, name, value):
        with self.lock:
            self.counters[name].value = value
            value = self.counters[name].value
        return value

    def _value(self, name):
        return self.counters[name].value

    def create_mapping(self):
        if self.num_cpus > 1:
            pool = mp.Pool(processes=self.num_cpus, initargs=(self.lock,))
            return pool.map, pool
        else:
            return map, None


class Criteria():
    """
    A class to manage the termination conditions and statistics of the evolutionary algorithm.

    Parameters
    ----------
    deap_param : object
        A configuration object containing DEAP parameters.
    n_obj : int, optional
        Number of objectives.

    Attributes
    ----------
    n_obj : int
        Number of objectives.
    effmax, effmin, effavg, effstd : numpy.ndarray
        Arrays to store maximum, minimum, average, and standard deviation of efficiency for each generation.
    conditions : dict
        Dictionary to track termination conditions.

    Methods
    -------
    check_termination_conditions(gen)
        Check if the termination conditions are met for a given generation.
    update_statistics(gen, halloffame)
        Update statistics for the current generation.
        This function modifies the class attributes effmax, effmin, effavg, and effstd.
    write_front_history(path_subcatch, gen)
        Write the history of objective function changes to a CSV file.
    """

    def __init__(self, deap_param, n_obj=1):

        self.n_obj = n_obj

        self.min_gen = deap_param.min_gen
        self.max_gen = deap_param.max_gen
        self.gen_offset = deap_param.gen_offset  # 3
        self.apply_statistical_stall_check = deap_param.apply_statistical_stall_check
        self.mu = deap_param.mu

        self.effmax_tol = deap_param.effmax_tol  # 0.003

        # Initialise statistics arrays
        self.effmax = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.effmin = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.effavg = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.effstd = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN

        # Initialise population statistics arrays
        self.popmax = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.popmin = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.popavg = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.popstd = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN

        self.conditions = {"maxGen": False, "StallFit": False, "StatisticalStallFit": False}

    def check_termination_conditions(self, gen):
        # Terminate the optimization after maxGen generations
        if gen >= self.max_gen:
            print(">> Termination criterion maxGen fulfilled.")
            self.conditions['maxGen'] = True

        if gen >= self.min_gen and (gen >= self.gen_offset) and (self.effmax[gen, 0] - self.effmax[gen - self.gen_offset, 0]) < self.effmax_tol:
            if self.apply_statistical_stall_check:
                # CR optional stopping condition: even if the no-improvement KGE criterion is fulfilled, check the statistics of the latest gen_offset population to check if any overall improvement is going on
                # Calculate t-test over the last `gen_offset` generations
                mean_current = self.popavg[gen, 0]
                std_current = self.popstd[gen, 0]
                n_current = self.mu  # Assuming self.mu is the population size
                
                # Combine statistics from previous `gen_offset` generations
                means_previous = self.popavg[gen-self.gen_offset:gen, 0]
                stds_previous = self.popstd[gen-self.gen_offset:gen, 0]
                n_previous = self.mu * self.gen_offset  # Total samples in previous gen_offset generations
                
                # Compute weighted average of means and stds for the previous generations
                mean_previous = sum(means_previous) / self.gen_offset
                std_previous = (sum(stds_previous**2) / self.gen_offset)**0.5
                
                # Perform t-test
                t_stat, p_val = ttest_ind_from_stats(mean_current, std_current, n_current, mean_previous, std_previous, n_previous)
                
                # Check p-value
                if (not np.isnan(p_val)) and (mean_current > mean_previous and p_val < 0.05):
                    print(">> Significant improvement detected (p_val={:.3f}, mean_current={:.3f}, std_current={:.3f}, mean_previous={:.3f}, std_previous={:.3f}), continuing optimization." 
                          .format(p_val, mean_current, std_current, mean_previous, std_previous))
                else:
                    print(">> Termination criterion statistical no-improvement KGE fulfilled (p_val={:.3f}).".format(p_val))
                    self.conditions["StatisticalStallFit"] = True
            else:
                # # DD attempt to stop early with different criterion
                # if (effmax[gen.value,0]-effmax[gen.value-1,0]) < 0.001 and np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value, :]) > np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value - 1, :]):
                #     print(">> Termination criterion no-improvement sae fulfilled.")
                #     # conditions["StallFit"] = True
                print(">> Termination criterion no-improvement KGE fulfilled.")
                self.conditions["StallFit"] = True

    def update_statistics_population(self, gen, population):
        # Loop through the different objective functions and calculate some statistics from the current selected population
        # N.B: population is already selected using best self.mu individuals from previous population + new offspring items
        for ii in range(self.n_obj):
            self.popmax[gen, ii] = np.amax([population[x].fitness.values[ii] for x in range(len(population))])
            self.popmin[gen, ii] = np.amin([population[x].fitness.values[ii] for x in range(len(population))])
            self.popavg[gen, ii] = np.average([population[x].fitness.values[ii] for x in range(len(population))])
            self.popstd[gen, ii] = np.std([population[x].fitness.values[ii] for x in range(len(population))])
        print(">> gen: " + str(gen) + ", selected population with offsprings: KGE max={:.3f}, min={:.3f}, avg={:.3f}, std={:.3f}".format(self.popmax[gen, 0], 
                                                                                                                         self.popmin[gen, 0],
                                                                                                                         self.popavg[gen, 0],
                                                                                                                         self.popstd[gen, 0]))

    def update_statistics(self, gen, halloffame):
        # Loop through the different objective functions and calculate some statistics from the Pareto optimal population
        for ii in range(self.n_obj):
            self.effmax[gen, ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            self.effmin[gen, ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            self.effavg[gen, ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            self.effstd[gen, ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        print(">> gen: " + str(gen) + ", effmax_KGE: " + "{0:.3f}".format(self.effmax[gen, 0]))

    def write_front_history(self, path_subcatch, gen):
        front_history = pandas.DataFrame()
        front_history['gen'] = range(gen)
        front_history['effmax_R'] = self.effmax[0:gen, 0]
        front_history['effmin_R'] = self.effmin[0:gen, 0]
        front_history['effstd_R'] = self.effstd[0:gen, 0]
        front_history['effavg_R'] = self.effavg[0:gen, 0]
        front_history.to_csv(os.path.join(path_subcatch, "front_history.csv"))


class CalibrationDeap():
    """
    A class to manage the calibration process using DEAP library.

    Parameters
    ----------
    cfg : object
        A configuration object containing calibration parameters.
    fun : callable
        The objective function to evaluate.
    objective_weights : List
        List containing the weights of each objective in the multi-objective optimization.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    pop, mu, lambda_ : int
        Population size, number of individuals to select, and number of children to produce.
    cxpb, mutpb : float
        Crossover and mutation probabilities.
    param_ranges : pandas.DataFrame
        DataFrame containing parameter ranges.
    toolbox : deap.base.Toolbox
        DEAP toolbox for evolutionary operators.

    Methods
    -------
    updatePopulationFromHistory(pHistory)
        Update the population from a historical record.
    restore_calibration(halloffame, history_file)
        Restore the calibration state from a history file.
    generate_population(halloffame)
        Generate the initial population for the calibration.
    generate_offspring(gen, halloffame, population)
        Generate offspring for a new generation.
    run(path_subcatch, lock_mgr)
        Run the calibration process.
    """

    def __init__(self, cfg, fun, objective_weights, seed=None):

        print('Creating calibration object')

        if seed:
            print(f'Seeding {seed} into deap for random numbers')
            random.seed(seed)
        else:
            print('Using default deap seed')

        deap_param = cfg.deap_param

        self.pop = deap_param.pop
        self.mu = deap_param.mu
        self.lambda_ = deap_param.lambda_

        self.objective_weights = objective_weights
        self.criteria = Criteria(deap_param, len(objective_weights))

        self.cxpb = deap_param.cxpb
        self.mutpb = deap_param.mutpb

        self.param_ranges = cfg.param_ranges

        # Setup DEAP
        creator.create("FitnessMin", base.Fitness, weights=objective_weights)
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_float", random.uniform, 0, 1)

        # Structure initializers
        toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(self.param_ranges))
        toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

        def checkBounds(min, max):
            def decorator(func):
                def wrappper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        for i in range(len(child)):
                            if child[i] > max:
                                child[i] = max
                            elif child[i] < min:
                                child[i] = min
                    return offspring
                return wrappper
            return decorator

        def checkGwLossGwPerc(min, max, indexGwLoss, indexGwPerc, scaleGwLoss, offsetGwLoss, scaleGwPerc, offsetGwPerc):
            def decorator(func):
                def wrappper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        # condition in Lisflood OS: if GWloss > GwPercValue -> GwPerc = GwLoss
                        # then if GwPercValue < GWloss -> GwPerc = GwLoss
                        GwPercScaled = (child[indexGwPerc]*scaleGwPerc)+offsetGwPerc
                        GwLossScaled = (child[indexGwLoss]*scaleGwLoss)+offsetGwLoss
                        if GwPercScaled<GwLossScaled:                            
                            GwPercScaled=GwLossScaled
                            child[indexGwPerc]=(GwPercScaled-offsetGwPerc)/scaleGwPerc                            
                            assert(child[indexGwPerc]>=min)
                            assert(child[indexGwPerc]<=max)
                    return offspring
                return wrappper
            return decorator
        
        toolbox.register("evaluate", fun)
        toolbox.register("mate", tools.cxBlend, alpha=0.15)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
        toolbox.register("select", tools.selNSGA2)

        #ipar, par in enumerate(param_ranges.index):
        if ('GwLoss' in self.param_ranges.index) and ('GwPercValue' in self.param_ranges.index):
            indexGwLoss=self.param_ranges.index.get_loc('GwLoss')
            indexGwPerc=self.param_ranges.index.get_loc('GwPercValue')

            scaleGwLoss=(self.param_ranges.iloc[indexGwLoss][1]-self.param_ranges.iloc[indexGwLoss][0])
            offsetGwLoss=self.param_ranges.iloc[indexGwLoss][0]
            scaleGwPerc=(self.param_ranges.iloc[indexGwPerc][1]-self.param_ranges.iloc[indexGwPerc][0])
            offsetGwPerc=self.param_ranges.iloc[indexGwPerc][0]

            toolbox.decorate("mate", checkBounds(0, 1), checkGwLossGwPerc(0, 1, indexGwLoss, indexGwPerc, scaleGwLoss, offsetGwLoss, scaleGwPerc, offsetGwPerc))
            toolbox.decorate("mutate", checkBounds(0, 1), checkGwLossGwPerc(0, 1, indexGwLoss, indexGwPerc, scaleGwLoss, offsetGwLoss, scaleGwPerc, offsetGwPerc))
            toolbox.decorate("population", checkBounds(0, 1), checkGwLossGwPerc(0, 1, indexGwLoss, indexGwPerc, scaleGwLoss, offsetGwLoss, scaleGwPerc, offsetGwPerc))
        else:
            toolbox.decorate("mate", checkBounds(0, 1))
            toolbox.decorate("mutate", checkBounds(0, 1))

        self.toolbox = toolbox

    def updatePopulationFromHistory(self, pHistory):
        param_ranges = self.param_ranges
        n = len(pHistory)
        n_params = len(param_ranges)
        n_obj = len(self.objective_weights)
        paramvals = np.zeros(shape=(n, n_params))
        paramvals[:] = np.NaN
        invalid_ind = []
        fitnesses = []
        for ind in range(n):
            for ipar, par in enumerate(param_ranges.index):
                # # scaled to unscaled conversion
                # paramvals[ind][ipar] = pHistory.iloc[ind][par] * (float(ParamRanges.iloc[ipar,1]) - \
                #   float(ParamRanges.iloc[ipar,0]))+float(ParamRanges.iloc[ipar,0])
                # unscaled to scaled conversion
                paramvals[ind][ipar] = (pHistory.iloc[ind][par] - float(param_ranges.iloc[ipar, 0])) / \
                  (float(param_ranges.iloc[ipar, 1]) - float(param_ranges.iloc[ipar, 0]))
            # Create a fresh individual with the restored parameters
            # newInd = toolbox.Individual() # creates an individual with random numbers for the parameters
            newInd = creator.Individual(list(paramvals[ind]))  # creates a totally empty individual

            # add objectives (from file) to current individual
            objectives = pHistory.iloc[ind, n_params+1:n_params+1+n_obj].values
            newInd.fitness.values = objectives

            invalid_ind.append(newInd)

        return invalid_ind

    def restore_calibration(self, halloffame, history_file):

        param_ranges = self.param_ranges

        # Open the paramsHistory file from previous runs
        paramsHistory = pandas.read_csv(history_file, sep=",")[3:]
        print("Restoring previous calibration state")

        # Initiate the generations counter
        gen = 0

        population = None
        # reconstruct the generational evoluation
        for igen in range(int(paramsHistory["generation"].iloc[-1])+1):
            # retrieve the generation's data
            parsHistory = paramsHistory[paramsHistory["generation"] == igen]

            # we can only recover complete generations
            if (gen == 0 and len(parsHistory) == self.pop) or (gen > 0 and len(parsHistory) == self.lambda_):
                # reconstruct the invalid individuals array
                valid_ind = self.updatePopulationFromHistory(parsHistory)
                # Update the hall of fame with the generation's parameters
                halloffame.update(valid_ind)
                # prepare for the next stage
                if population is not None:
                    population[:] = self.toolbox.select(population + valid_ind, self.mu)
                else:
                    population = valid_ind
                self.criteria.update_statistics(gen, halloffame)
                self.criteria.update_statistics_population(gen, population)
                self.criteria.check_termination_conditions(gen)
                print('----> Generation {} recovered'.format(gen))
                gen = gen+1
            else:
                # in case of incomplete generation, we should delete corresponding run lines from the paramHistory.csv file
                incomplete_gen_id=gen
                if incomplete_gen_id == 0:
                    os.remove(history_file)
                else:
                    with open(history_file, "r") as f:
                        lines = f.readlines()
                    with open(history_file, "w") as f:
                        for line in lines:
                            if not line.startswith(str(incomplete_gen_id)+"_"):
                                f.write(line)
                # TODO: should we clean up also runs_log.csv?

        return population, gen

    def generate_population(self, halloffame):

        print("Generating first population")
        gen = 0

        # Start with a fresh population
        population = self.toolbox.population(n=self.pop)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid] # DD this filters the population or children for uncalculated fitnesses. We retain only the uncalculated ones to avoid recalculating those that already had a fitness. Potentially this can save time, especially if the algorithm tends to produce a child we already ran.

        # Run the first generation, first random sampling of the parameter space
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses): # DD this updates the fitness (=KGE) for the individuals in the global pool of individuals which we just calculated. ind are
            assert len(ind.fitness.weights) == len(fit)
            ind.fitness.values = fit

        halloffame.update(population) # DD this selects the best one

        self.criteria.update_statistics(gen, halloffame)
        self.criteria.update_statistics_population(gen, population)

        return population

    def generate_offspring(self, gen, halloffame, population):

        # Vary the population
        offspring = algorithms.varOr(population, self.toolbox, self.lambda_, self.cxpb, self.mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind) # DD this runs lisflood
        for ind, fit in zip(invalid_ind, fitnesses):
            assert len(ind.fitness.weights) == len(fit)
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = self.toolbox.select(population + offspring, self.mu)

        # Loop through the different objective functions and calculate some statistics
        # from the Pareto optimal population
        self.criteria.update_statistics(gen, halloffame)
        self.criteria.update_statistics_population(gen, population)

        self.criteria.check_termination_conditions(gen)

        print('Done generation {}'.format(gen))

    def run(self, path_subcatch, lock_mgr):

        t = time.time()

        print('Starting calibration')
        lock_mgr.set_gen(0)
        mapping, pool = lock_mgr.create_mapping()
        self.toolbox.register("map", mapping)

        # Start a new hall of fame
        halloffame = tools.ParetoFront()

        # Attempt to open a previous parameter history
        history_file = os.path.join(path_subcatch, "paramsHistory.csv")
        population = None
        if os.path.isfile(history_file) and os.path.getsize(history_file) > 0:
            population, gen = self.restore_calibration(halloffame, history_file)
            lock_mgr.set_gen(gen)
            
        if population is None:
            # No previous parameter history was found, so start from scratch
            population = self.generate_population(halloffame)
            lock_mgr.set_gen(1)

        # Resume the generational process from wherever we left off
        while not any(self.criteria.conditions.values()):
            self.generate_offspring(lock_mgr.get_gen(), halloffame, population)
            lock_mgr.increment_gen()

        # Finito
        if pool:
            pool.close()
        elapsed = time.time() - t
        print(">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s")

        # Save history of the change in objective function scores during calibration to csv file
        self.criteria.write_front_history(path_subcatch, lock_mgr.get_gen())

        return self.criteria.effmax[lock_mgr.get_gen()-1]
