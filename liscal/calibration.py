import os
import numpy as np
import pandas
import time

# deap related packages
import array
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


class Criteria():

    def __init__(self, deap_param, n_obj=1):

        self.n_obj = n_obj

        self.min_gen = deap_param.min_gen
        self.max_gen = deap_param.max_gen
        self.gen_offset = 3

        self.effmax_tol = 0.003

        # Initialise statistics arrays
        self.effmax = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.effmin = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.effavg = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN
        self.effstd = np.zeros(shape=(self.max_gen + 1, self.n_obj)) * np.NaN

        self.conditions = {"maxGen": False, "StallFit": False}

    def check_termination_conditions(self, gen):
        # Terminate the optimization after maxGen generations
        if gen >= self.max_gen:
            print(">> Termination criterion maxGen fulfilled.")
            self.conditions['maxGen'] = True

        if gen >= self.min_gen and (self.effmax[gen, 0] - self.effmax[gen - self.gen_offset, 0]) < self.effmax_tol:
            # # DD attempt to stop early with different criterion
            # if (effmax[gen.value,0]-effmax[gen.value-1,0]) < 0.001 and np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value, :]) > np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value - 1, :]):
            #     print(">> Termination criterion no-improvement sae fulfilled.")
            #     # conditions["StallFit"] = True
            print(">> Termination criterion no-improvement KGE fulfilled.")
            self.conditions["StallFit"] = True

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

    def __init__(self, cfg, fun, obj_weights, scheduler):

        deap_param = cfg.deap_param

        self.pop = deap_param.pop
        self.mu = deap_param.mu
        self.lambda_ = deap_param.lambda_

        self.criteria = Criteria(deap_param, len(obj_weights))

        self.cxpb = deap_param.cxpb
        self.mutpb = deap_param.mutpb

        self.param_ranges = cfg.param_ranges

        # Setup DEAP
        creator.create("FitnessMin", base.Fitness, weights=obj_weights)
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

        toolbox.register("evaluate", fun)
        toolbox.register("mate", tools.cxBlend, alpha=0.15)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
        toolbox.register("select", tools.selNSGA2)
        toolbox.decorate("mate", checkBounds(0, 1))
        toolbox.decorate("mutate", checkBounds(0, 1))

        self.toolbox = toolbox

        self.scheduler = scheduler
        self.toolbox.register("map", scheduler.create_mapping())
    
    def updatePopulationFromHistory(self, pHistory):
        param_ranges = self.param_ranges
        n = len(pHistory)
        paramvals = np.zeros(shape=(n, len(param_ranges)))
        paramvals[:] = np.NaN
        individuals = []
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
            individuals.append(newInd)
            # WARNING: Change the following line when using multi-objective functions
            # also load the old KGE the individual had (works only for single objective function)
            fitnesses.append((pHistory.iloc[ind][len(param_ranges) + 1],))
        # update the score of each
        for ind, fit in zip(individuals, fitnesses):
            ind.fitness.values = fit
        return individuals

    def restore_calibration(self, halloffame, history_file):

        param_ranges = self.param_ranges

        # Open the paramsHistory file from previous runs
        paramsHistory = pandas.read_csv(history_file, sep=",")[4:]
        print("Restoring previous calibration state")

        # Initiate the generations counter
        gen = 0

        population = None
        # reconstruct the generational evoluation
        for igen in range(int(paramsHistory["generation"].iloc[-1])+1):
            # retrieve the generation's data
            parsHistory = paramsHistory[paramsHistory["generation"] == igen]

            print('WARNING! Partial restoring... we only consider here complete generations! We need to implement restart for uncomplete generations...')
            if (gen == 0 and len(parsHistory) == self.pop) or (gen > 0 and len(parsHistory) == self.lambda_):

                # reconstruct the recovered individuals array
                individuals = self.updatePopulationFromHistory(parsHistory)

                # Run the model for the rest of the population - not implemented
                # fitnesses = self.toolbox.map(self.toolbox.evaluate, individuals)
                # for ind, fit in zip(individuals, fitnesses): # DD this updates the fitness (=KGE) for the individuals in the global pool of individuals which we just calculated. ind are
                #     assert len(ind.fitness.weights) == len(fit)
                #     ind.fitness.values = fit

                # Update the hall of fame with the generation's parameters
                halloffame.update(individuals)

                # prepare for the next stage
                if population is not None:
                    population[:] = self.toolbox.select(population + individuals, self.mu)
                else:
                    population = individuals

                self.criteria.update_statistics(gen, halloffame)

                self.criteria.check_termination_conditions(gen)

                print('Found generation {}'.format(gen))

                gen = gen+1
            else:
                break

        return population, gen

    def backup_individuals(self, individuals):
        return

    def restore_individuals(self, individuals):
        return

    def create_individuals(self, offspring, gen):
        individuals_deap = [ind for ind in offspring if not ind.fitness.valid]
        individuals = []
        for i, child in enumerate(offspring):
            if not child.fitness.valid:
                ind = {}
                ind['value'] = np.array(child)
                ind['gen'] = gen
                ind['id'] = i
                ind['lock'] = self.scheduler.lock
                individuals.append(ind)
        return individuals, individuals_deap

    def compute_generation(self, halloffame, offspring, gen):
        self.backup_individuals(offspring)

        # Evaluate the individuals with an invalid fitness
        individuals, individuals_deap = self.create_individuals(offspring, gen)

        # Run the model (e.g. lisflood)
        self.scheduler.distribute(individuals)
        fitnesses = self.toolbox.map(self.toolbox.evaluate, individuals)
        fitnesses = self.scheduler.gather(fitnesses)
        print(self.scheduler.gather)

        # Update individuals with resulting fitnesses
        for ind, fit in zip(individuals_deap, fitnesses):
            ind_fit = ind.fitness
            assert len(ind_fit.weights) == len(fit)
            ind_fit.values = fit

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Loop through the different objective functions and calculate some statistics
        # from the Pareto optimal population
        self.criteria.update_statistics(gen, halloffame)

    def create_population(self, halloffame):

        print("Generating first population")
        gen = 0

        # Start with a fresh population
        population = self.toolbox.population(n=self.pop)

        # Compute current generation
        self.compute_generation(halloffame, population, gen)

        print("First population done")

        return population

    def generate_offspring(self, gen, halloffame, population):

        print('Starting generation {}'.format(gen))

        # Vary the population
        offspring = algorithms.varOr(population, self.toolbox, self.lambda_, self.cxpb, self.mutpb)

        # Compute current generation
        self.compute_generation(halloffame, offspring, gen)

        # Select the next generation population
        population = self.toolbox.select(population + offspring, self.mu)

        self.criteria.check_termination_conditions(gen)

        print('Done generation {}'.format(gen))

        return population

    def run(self, path_subcatch):

        t = time.time()

        print('Starting calibration')
        gen = 0

        # Start a new hall of fame
        halloffame = tools.ParetoFront()

        # Attempt to open a previous parameter history
        history_file = os.path.join(path_subcatch, "paramsHistory.csv")
        if os.path.isfile(history_file) and os.path.getsize(history_file) > 0:
            population, gen = self.restore_calibration(halloffame, history_file)

        if gen==0:
            # No previous parameter history was found, so start from scratch
            population = self.create_population(halloffame)
            gen = 1

        # Resume the generational process from wherever we left off
        while not any(self.criteria.conditions.values()):
            population = self.generate_offspring(gen, halloffame, population)
            gen += 1

        # Finito
        elapsed = time.time() - t
        print(">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s")

        # Save history of the change in objective function scores during calibration to csv file
        self.criteria.write_front_history(path_subcatch, gen)

        self.scheduler.close()

        return self.criteria.effmax[gen-1]
