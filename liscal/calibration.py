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


class LockManager():
    def __init__(self):
        
        mgr = mp.Manager()
        self.counters = {}
        self.counters['run'] = mgr.Value('i', -1)
        self.counters['gen'] = mgr.Value('i', -1)

        assert self.get_run() == -1
        assert self.get_gen() == -1

        self.lock = mgr.Lock()

    def increment_gen(self):
        self._increment('gen')
        self._set('run', 0)

    def increment_run(self):
        self._increment('run')

    def set_gen(self, value):
        self._set('gen', value)
        self._set('run', 0)

    def set_run(self, value):
        self._set('run', value)

    def get_gen(self):
        return self._value('gen')

    def get_run(self):
        return self._value('run')

    def _increment(self, name):
        with self.lock:
            self.counters[name].value += 1

    def _set(self, name, value):
        with self.lock:
            self.counters[name].value = value

    def _value(self, name):
        return self.counters[name].value


class DEAPParameters():

    def __init__(self, parser):
        self.use_multiprocessing = int(parser.get('DEAP','use_multiprocessing'))
        self.numCPUs = int(parser.get('DEAP','numCPUs'))
        self.minGen = int(parser.get('DEAP','minGen'))
        self.maxGen = int(parser.get('DEAP','maxGen'))
        self.pop = int(parser.get('DEAP','pop'))
        self.mu = int(parser.get('DEAP','mu'))
        self.lambda_ = int(parser.get('DEAP','lambda_'))

        self.cxpb = 0.6
        self.mutpb = 0.4


def initialise_deap(cfg, model):

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # history = tools.History()

    # Attribute generator
    toolbox.register("attr_float", random.uniform, 0, 1)

    # Structure initializers
    toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(cfg.param_ranges))
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

    toolbox.register("evaluate", model.run) #profiler) #RunModel) #DD Debug for profiling
    toolbox.register("mate", tools.cxBlend, alpha=0.15)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)
    toolbox.decorate("mate", checkBounds(0, 1))
    toolbox.decorate("mutate", checkBounds(0, 1))
    # toolbox.decorate("mate", history.decorator)
    # toolbox.decorate("mutate", history.decorator)

    return toolbox


def check_termination_conditions(gen, deap_param, effmax, conditions):

    min_gen = deap_param.minGen
    max_gen = deap_param.maxGen

    # Terminate the optimization after maxGen generations
    if gen >= max_gen:
        print(">> Termination criterion maxGen fulfilled.")
        conditions['maxGen'] = True

    if gen >= min_gen and (effmax[gen, 0] - effmax[gen - 3, 0]) < 0.003:
        # # DD attempt to stop early with different criterion
        # if (effmax[gen.value,0]-effmax[gen.value-1,0]) < 0.001 and np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value, :]) > np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value - 1, :]):
        #     print(">> Termination criterion no-improvement sae fulfilled.")
        #     # conditions["StallFit"] = True
        print(">> Termination criterion no-improvement KGE fulfilled.")
        conditions["StallFit"] = True


def updatePopulationFromHistory(pHistory, param_ranges):
    n = len(pHistory)
    paramvals = np.zeros(shape=(n, len(param_ranges)))
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
        invalid_ind.append(newInd)
        # WARNING: Change the following line when using multi-objective functions
        # also load the old KGE the individual had (works only for single objective function)
        fitnesses.append((pHistory.iloc[ind][len(param_ranges) + 1],))
    # update the score of each
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return invalid_ind


def restore_calibration(cfg, lock_mgr, toolbox, halloffame, history_file,
                        effmax, effmin, effavg, effstd, conditions):

    param_ranges = cfg.param_ranges
    deap_param = cfg.deap_param

    # Open the paramsHistory file from previous runs
    paramsHistory = pandas.read_csv(history_file, sep=",")[4:]
    print("Restoring previous calibration state")

    # Initiate the generations counter
    lock_mgr.set_gen(0)

    population = None
    # reconstruct the generational evoluation
    for igen in range(int(paramsHistory["generation"].iloc[-1])+1):
        # retrieve the generation's data
        parsHistory = paramsHistory[paramsHistory["generation"] == igen]
        # reconstruct the invalid individuals array
        valid_ind = updatePopulationFromHistory(parsHistory, param_ranges)
        # Update the hall of fame with the generation's parameters
        halloffame.update(valid_ind)
        # prepare for the next stage
        if population is not None:
            population[:] = toolbox.select(population + valid_ind, deap_param.mu)
        else:
            population = valid_ind

        gen = lock_mgr.get_gen()

        # Loop through the different objective functions and calculate some statistics from the Pareto optimal population
        for ii in range(1):
            effmax[gen, ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effmin[gen, ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effavg[gen, ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effstd[gen, ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        print(">> gen: " + str(gen) + ", effmax_KGE: " + "{0:.3f}".format(effmax[gen, 0]))

        check_termination_conditions(gen, deap_param, effmax, conditions)

        lock_mgr.increment_gen()

    return population



def generate_population(deap_param, lock_mgr, toolbox, halloffame, effmax, effmin, effavg, effstd, conditions):
    # Start with a fresh population
    population = toolbox.population(n=deap_param.pop)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid] # DD this filters the population or children for uncalculated fitnesses. We retain only the uncalculated ones to avoid recalculating those that already had a fitness. Potentially this can save time, especially if the algorithm tends to produce a child we already ran.

    lock_mgr.set_gen(0)

    # Run the first generation, first random sampling of the parameter space
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses): # DD this updates the fitness (=KGE) for the individuals in the global pool of individuals which we just calculated. ind are
        ind.fitness.values = fit

    halloffame.update(population) # DD this selects the best one

    gen = lock_mgr.get_gen()
    assert gen == 0

    # Loop through the different objective functions and calculate some statistics from the Pareto optimal population
    for ii in range(1):
        effmax[0,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effmin[0,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effavg[0,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effstd[0,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
    print(">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0]))

    # Update the generation to the first
    lock_mgr.increment_gen()

    return population


def generate_offspring(deap_param, lock_mgr, toolbox, halloffame, population, effmax, effmin, effavg, effstd, conditions):

    # Vary the population
    offspring = algorithms.varOr(population, toolbox, deap_param.lambda_, deap_param.cxpb, deap_param.mutpb)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) # DD this runs lisflood
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(offspring)

    # Select the next generation population
    population[:] = toolbox.select(population + offspring, deap_param.mu)

    gen = lock_mgr.get_gen()

    # Loop through the different objective functions and calculate some statistics
    # from the Pareto optimal population
    for ii in range(1):
        effmax[gen,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effmin[gen,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effavg[gen,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effstd[gen,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
    print(">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0]))

    check_termination_conditions(gen, deap_param, effmax, conditions)

    lock_mgr.increment_gen()


def write_front_history(path_subcatch, gen, effmax, effmin, effavg, effstd):
    front_history = pandas.DataFrame()
    front_history['gen'] = range(gen)
    front_history['effmax_R'] = effmax[0:gen,0]
    front_history['effmin_R'] = effmin[0:gen,0]
    front_history['effstd_R'] = effstd[0:gen,0]
    front_history['effavg_R'] = effavg[0:gen,0]
    front_history.to_csv(os.path.join(path_subcatch,"front_history.csv"))


def read_param_history(path_subcatch):
    pHistory = pandas.read_csv(os.path.join(path_subcatch, "paramsHistory.csv"), sep=",")[3:]
    return pHistory


def write_ranked_solution(path_subcatch, pHistory):
    # Keep only the best 10% of the runs for the selection of the parameters for the next generation
    pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
    pHistory = pHistory.head(int(max(2,round(len(pHistory) * 0.1))))
    n = len(pHistory)
    minOffset = 0.1
    maxOffset = 1.0
    # Give ranking scores to corr
    pHistory = pHistory.sort_values(by="Correlation", ascending=False)
    pHistory["corrRank"] = [minOffset + float(i + 1) * (maxOffset - minOffset) / n for i, ii in enumerate(pHistory["Correlation"].values)]
    # Give ranking scores to sae
    pHistory = pHistory.sort_values(by="sae", ascending=True)
    pHistory["saeRank"] = [minOffset + float(i + 1) * (maxOffset - minOffset) / n for i, ii in enumerate(pHistory["sae"].values)]
    # Give ranking scores to KGE
    pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
    pHistory["KGERank"] = [minOffset + float(i + 1) * (maxOffset - minOffset) / n for i, ii in enumerate(pHistory["Kling Gupta Efficiency"].values)]
    # Give pareto score
    pHistory["paretoRank"] = pHistory["corrRank"].values * pHistory["saeRank"].values * pHistory["KGERank"].values
    pHistory = pHistory.sort_values(by="paretoRank", ascending=True)
    pHistory.to_csv(os.path.join(path_subcatch, "pHistoryWRanks.csv"), ',')

    return pHistory


def write_pareto_front(param_ranges, path_subcatch, pHistory):
    # Select the best pareto candidate
    bestParetoIndex = pHistory["paretoRank"].nsmallest(1).index
    # Save the pareto front
    paramvals = np.zeros(shape=(1,len(param_ranges)))
    paramvals[:] = np.NaN
    for ipar, par in enumerate(param_ranges.index):
        paramvals[0][ipar] = pHistory.loc[bestParetoIndex][par]
    pareto_front = pandas.DataFrame(
        {
            'effover': pHistory["Kling Gupta Efficiency"].loc[bestParetoIndex],
            'R': pHistory["Kling Gupta Efficiency"].    loc[bestParetoIndex]
        }, index=[0]
    )
    for ii in range(len(param_ranges)):
        pareto_front["param_"+str(ii).zfill(2)+"_"+param_ranges.index[ii]] = paramvals[0,ii]
    pareto_front.to_csv(os.path.join(path_subcatch, "pareto_front.csv"),',')


def run_calibration(cfg, obsid, path_subcatch, station_data, model, lock_mgr):
    
    deap_param = cfg.deap_param

    toolbox = initialise_deap(cfg, model)

    observed_streamflow = model.read_observed_streamflow()

    t = time.time()

    # load forcings and input maps in cache
    # required in front of processing pool
    # otherwise each child will reload the maps
    model.init_run()

    pool_size = deap_param.numCPUs #mp.cpu_count() * 1 ## DD just restrict the number of CPUs to use manually
    pool = mp.Pool(processes=pool_size, initargs=(lock_mgr.lock,))
    toolbox.register("map", pool.map)

    #cxpb = 0.9 # For someone reason, if sum of cxpb and mutpb is not one, a lot less Pareto optimal solutions are produced
    # DD: These values are used as percentage probability, so they should add up to 1, to determine whether to mutate or cross. The former finds the parameter for the next generation by taking the average of two parameters. This could lead to convergence to a probability set by the first generation as a biproduct of low first-generation parameter spread (they are generated using a uniform-distribution random generator.
    #mutpb = 0.1

    # Initialise statistics arrays
    effmax = np.zeros(shape=(deap_param.maxGen + 1, 1)) * np.NaN
    effmin = np.zeros(shape=(deap_param.maxGen + 1, 1)) * np.NaN
    effavg = np.zeros(shape=(deap_param.maxGen + 1, 1)) * np.NaN
    effstd = np.zeros(shape=(deap_param.maxGen + 1, 1)) * np.NaN

    # Start generational process setting all stopping conditions to false
    conditions = {"maxGen": False, "StallFit": False}

    # Start a new hall of fame
    halloffame = tools.ParetoFront()

    # Attempt to open a previous parameter history
    history_file = os.path.join(path_subcatch, "paramsHistory.csv")
    if os.path.isfile(history_file) and os.path.getsize(history_file) > 0:
        population = restore_calibration(cfg, lock_mgr, toolbox, halloffame, history_file, effmax, effmin, effavg, effstd, conditions)
    else:
        # No previous parameter history was found, so start from scratch
        population = generate_population(deap_param, lock_mgr, toolbox, halloffame, effmax, effmin, effavg, effstd, conditions)

    # Resume the generational process from wherever we left off
    while not any(conditions.values()):
        generate_offspring(deap_param, lock_mgr, toolbox, halloffame, population, effmax, effmin, effavg, effstd, conditions)

    # plotSolutionTree(history, paramsHistory)

    # Finito
    pool.close()
    elapsed = time.time() - t
    print(">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s")

    ########################################################################
    #   Save calibration results
    ########################################################################

    # Save history of the change in objective function scores during calibration to csv file
    write_front_history(path_subcatch, lock_mgr.get_gen(), effmax, effmin, effavg, effstd)

    pHistory = read_param_history()
    pHistory = write_ranked_solution(path_subcatch, path_subcatch)

    write_pareto_front(cfg.param_ranges, path_subcatch, pHistory)

    return
