'''
Created on 21-Mar-2014

@author: Jim D'Souza

Description : Uses genetic algorithms to identify users who are most likely to default
'''

from operator import itemgetter, attrgetter
from numpy import arange,array,ones,linalg


import csv
import random
import sys
import os
import math
import re
import numpy as np
import logging

# GLOBAL VARIABLES

solution_found = False
popN = 30 # n number of chromos per population
genesPerCh = 48
max_iterations = 100
### Target is a matrix which has 1 for default, 0 for non default ###
crossover_rate = 0.95
mutation_rate = 0.05


### Generates the initial chromosome - this is done by running a simple logistic regression and obtaining the parameters ###
def initialPop(training_data_set):
    target = training_data_set[:,52]
    eflkey = training_data_set[:,0]
    country = training_data_set[:,3]
    vars = training_data_set[:,4:52]
        
    w = linalg.lstsq(vars,target)[0]
    
    return w,vars,target


###Generates random population of chromos ###
def generatePop (startingPop):
    chromos, chromo = [], []
    for eachChromo in range(popN):
        chromo = []
        for bit in range(genesPerCh):
            if eachChromo == 0 :
                chromo.append(startingPop[bit])
            else:
                if (startingPop[bit] >= 0 and startingPop[bit] <= 1) or (startingPop[bit] <= 0 and startingPop[bit] >= -1) :
                    chromo.append(startingPop[bit]*random.uniform(-1.0,1.0))
                else :
                    chromo.append(startingPop[bit])
        
        chromos.append(chromo)
        
    return chromos

  
###Evaluates the mathematical expressions in number + operator blocks of two###
def evaluate(chromo, vars):
    outputs = []
       
    for var in vars :
        output = 0.0 
        try :
            for i in range(0,len(chromo)) :
                var[i] = float(var[i])
                output = output + chromo[i] * float(var[i])
        except TypeError:
            print "Type error"
            print "Var[i] :", var[i]
            print "Chromo[i] :", chromo[i]
            print "Iteration :", i
                  
        outputs.append(output)

    return outputs


def checkerror(target, outputs, non_default_limit, default_limit):
    errors = 0.0
    
    for i in range(0,len(target)-1) :
        error = 0.0
        
        if outputs[i] < 0.0:
            outputs[i] = 0.0
        elif outputs[i] > 1.0 :
            outputs[i] = 1.0
            
        if target[i] == '0' :
            if (outputs[i] - 0.0) > non_default_limit :
                error = error + 1.0
        elif target[i] == '1' :
            if (1.0 - outputs[i]) >= default_limit :
                error = error + 1.0

        errors = errors + error
    
    return errors

###Calulates fitness as a fraction of the total fitness###
def calcFitness (errors):
    fitnessScores = []
    totalError = sum(errors)
    i = 0
    # fitness scores are a fraction of the total error
    for error in errors:
        fitnessScores.append (float(errors[i])/float(totalError))
        i += 1
    
    return fitnessScores


###Takes a population of chromosomes and returns a list of tuples where each chromo is paired to its fitness scores and ranked accroding to its fitness###
def rankPop (chromos,vars,target, non_default_limit, default_limit):
    totaloutputs, totalerrors = [], []
    solution_found = False
    # translate each chromo into mathematical expression (protein), evaluate the output of the expression,
    # calculate the inverse error of the output
    # print '%s: %s\t=%s \t%s %s' %('n'.rjust(5), 'PROTEIN'.rjust(30), 'OUTPUT'.rjust(10), 'INVERSE ERROR'.rjust(17), 'GRAPHICAL INVERSE ERROR'.rjust(105))
    
    for chromo in range(0,len(chromos)) :
        outputs = evaluate(chromos[chromo],vars)
        
        min = np.min(outputs)
        max = np.max(outputs)
        for i in range(0,len(outputs)):
            outputs[i] = (((outputs[i]-min)*1)/(max-min)) + 0 # convert to range 0 to 1
        
        totaloutputs.append(outputs)
    
    
    print "Passed Output prediction stage"       
    
    print "Entered Error calculating stage"
    for outputs in totaloutputs:
        errors = checkerror(target,outputs, non_default_limit, default_limit)
        totalerrors.append(errors)
    
    print "Entered Fitness calculating stage" 
    fitnessScores = calcFitness (totalerrors) # calc fitness scores from the errors calculated
    
    print "Checking if a solution has been reached..."
    for j in range(0,len(totalerrors)):
        print "Error for chromo ",j," is : ", totalerrors[j], " and fitness score is : ", fitnessScores[j], " . The chromo is :", chromos[j]
        if totalerrors[j] <= 1 :
            print "Solution found"
            solution_found = True
            break
        else :
            continue
        
    
        
    pairedPop = zip ( chromos, fitnessScores) # pair each chromo with its fitness score
    rankedPop = sorted ( pairedPop,key = itemgetter(-1), reverse = False ) # sort the paired pop by ascending fitness score
        
    
    return rankedPop, solution_found, j



### taking a ranked population selects two of the fittest members using roulette method###
def selectFittest (fitnessScores, rankedChromos):
        
    while 1==1 :
        index1 = roulette (fitnessScores)
        index2 = roulette (fitnessScores)
        if index1 == index2 :
            continue
        elif index1 is not None and index2 is not None:
            break
        
    
    ch1 = rankedChromos[index1] # select  and return chromosomes for breeding
    ch2 = rankedChromos[index2]
        
    return ch1, ch2


###Fitness scores are fractions, their sum = 1. Fitter chromosomes have a larger fraction.  ###
def roulette (fitnessScores):
    cumalativeFitness = 0.0
    r = random.random()/3.0
    
    for i in range(len(fitnessScores)-1): # for each chromosome's fitness score
        cumalativeFitness += fitnessScores[i] # add each chromosome's fitness score to cumalative fitness
        if cumalativeFitness > r: # in the event of cumalative fitness becoming greater than r, return index of that chromo
            return i


def crossover (ch1, ch2):
    if ch1 == ch2 :
        ch2 = mutate(ch2, True)
        return ch1, ch2
    else :
        # at a random gene
        r = random.randint(0,genesPerCh)
        return ch1[:r]+ch2[r:], ch2[:r]+ch1[r:]


def mutate (ch, mutation_rate, guarantee_mutation=False):
    mutatedCh = []
    for bit in ch:
        if guarantee_mutation == False and random.random() < mutation_rate:
            if (bit >= 0 and bit <= 1) or (bit <= 0 and bit >= -1) :
                mutatedCh.append(bit*random.uniform(-1.0,1.0))
            else :
                mutatedCh.append(bit)
        elif guarantee_mutation == True and random.random() < 0.5 :
            mutatedCh.append(bit*random.uniform(-1.0,1.0))
        else:
            mutatedCh.append(bit)
    
    #assert mutatedCh != ch
    return mutatedCh
   
      
###Using breed and mutate it generates two new chromos from the selected pair###
def breed (ch1, ch2, crossover_rate, mutation_rate):
    newCh1, newCh2 = [], []
    
    if random.random() < crossover_rate: # rate dependent crossover of selected chromosomes
        newCh1, newCh2 = crossover(ch1, ch2)
    else:
        newCh1, newCh2 = ch1, ch2
    
    newnewCh1 = mutate (newCh1, mutation_rate, False) # mutate crossovered chromos
    newnewCh2 = mutate (newCh2, mutation_rate, False)
    
    return newnewCh1, newnewCh2

### Taking a ranked population return a new population by breeding the ranked one###
def iteratePop (rankedPop, crossover_rate, mutation_rate):
    
    
    fitnessScores = [ item[1] for item in rankedPop ] # extract fitness scores from ranked population
    rankedChromos = [ item[0] for item in rankedPop ] # extract chromosomes from ranked population
        
    newpop = []
    newpop = rankedChromos[:popN/3] # known as elitism, conserve the best solutions to new population
    
    
    while len(newpop) <= popN:
        ch1, ch2 = [], []
        ch1, ch2 = selectFittest (fitnessScores, rankedChromos) # select two of the fittest chromos      
        ch1, ch2 = breed (ch1, ch2, crossover_rate, mutation_rate) # breed them to create two new chromosomes 
        
        newpop.append(ch1) # and append to new population
        newpop.append(ch2)
        
        
    return newpop

def dataExtraction(trialrun): 
                    
    rownum1 = 0
    rownum2 = 0
    original_data_set = []
    complete_data_set = []
      
    ### Extract data from prepped file into data set ###
    with open('D:\EFL\EFL external data set.csv','rb') as f1:
        reader1 = csv.reader(f1)
        for row in reader1:
            if rownum1 == 0 :
                header1 = row
            else :
                original_data_set.append(row)
            
            rownum1 = rownum1 + 1
        
    
    ### Extract data from prepped file into data set ###
    with open('D:\EFL\EFL external data set-final_prep.csv','rb') as f2:
        reader2 = csv.reader(f2)
        for row in reader2:
            if rownum2 == 0 :
                header2 = row
            else :
                complete_data_set.append(row)
            
            rownum2 = rownum2 + 1
            
    return original_data_set, complete_data_set, header2


### Extract the data into different data sets for training and testing ###
def dataSplitting (complete_data_set, trialrun):

    training_data_set = []
    testing_data_set  = []
    full_data_set = []
    blanks_data_set  = []
    
        
    ### Split complete_data_set into training and testing data sets ###
    for i in range(0,len(complete_data_set)) :
        if (complete_data_set[i][52] == '') :
            blanks_data_set.append(complete_data_set[i])
        else :
            full_data_set.append(complete_data_set[i])
    
    full_data_set = np.array(full_data_set)
    blanks_data_set = np.array(blanks_data_set)
        
    
    if trialrun == True :
        for j in range(0,len(full_data_set)) :
            if j < 4000 :
                training_data_set.append(full_data_set[j])
            else:
                testing_data_set.append(full_data_set[j])
        training_data_set = np.array(training_data_set)
        testing_data_set = np.array(testing_data_set)
    
    elif trialrun == False :
        training_data_set = full_data_set
        testing_data_set = blanks_data_set
        
    
    
    ### Removing non essential information ###
    elim = []
    
    training_data_set = np.delete(training_data_set,elim,1)
    testing_data_set = np.delete(testing_data_set,elim,1)

    
    return training_data_set, testing_data_set, full_data_set, blanks_data_set
   
   
 
### Entire process in one block ###
### This is done so that the code can run separately on multiple clusters of data points ###
def cluster_run(complete_data_set, trialrun, cluster_num):
        
    training_data_set, testing_data_set, full_data_set, blanks_data_set = dataSplitting(complete_data_set, trialrun)
              
    startingPop,vars,target = initialPop(training_data_set)
        
    chromos = generatePop(startingPop) #generate new population of random chromosomes
    
    iterations = 0  
    solution_found = False
           
    ### Varying parameters according to each cluster ###
    if cluster_num == 1:
        ### Target is a matrix which has 1 for default, 0 for non default ###
        non_default_limit = 0.60
        default_limit = 0.40
        crossover_rate = 0.95
        mutation_rate = 0.05
    elif cluster_num == 2:
        ### Target is a matrix which has 1 for default, 0 for non default ###
        non_default_limit = 0.60
        default_limit = 0.40
        crossover_rate = 0.95
        mutation_rate = 0.05
    elif cluster_num == 3:
        ### Target is a matrix which has 1 for default, 0 for non default ###
        non_default_limit = 0.30
        default_limit = 0.70
        crossover_rate = 0.95
        mutation_rate = 0.05
    elif cluster_num == 4:
        ### Target is a matrix which has 1 for default, 0 for non default ###
        non_default_limit = 1.0
        default_limit = 1.0
        crossover_rate = 0.95
        mutation_rate = 0.05
    elif cluster_num == 5:
        ### Target is a matrix which has 1 for default, 0 for non default ###
        non_default_limit = 0.60
        default_limit = 0.40
        crossover_rate = 0.95
        mutation_rate = 0.05
    
        
        
    ### Running the algorithm on each cluster ###
    
    while iterations != max_iterations and solution_found != True:
        # take the pop of random chromos and rank them based on their fitness score/proximity to target output
        print '\nCurrent iterations:', iterations
        rankedPop, solution_found, solution_index = rankPop(chromos,vars,target, non_default_limit, default_limit) 

                
        if solution_found != True:
            # if solution is not found iterate a new population from previous ranked population
            chromos = []
            chromos = iteratePop(rankedPop, crossover_rate, mutation_rate)
        
        
        iterations += 1
    
    if solution_found == True:
        print chromos[solution_index]
    
    else :
        solution_index = 0
        
        
    print "Best solution obtained : ", chromos[solution_index]
        
    
    ### Predicting targets for the testing data set
    
    predicted_targets = []
    checking_data_set = complete_data_set
    
    print len(checking_data_set[0])
    
    for i in range(0,len(checking_data_set)) :
        chromo_count = 0
        predicted_target = 0.0
        for j in range(4,len(checking_data_set[i])-1) :
            predicted_target = predicted_target + float(checking_data_set[i][j]) * chromos[0][chromo_count]
            chromo_count = chromo_count + 1

        predicted_targets.append(predicted_target)
    
    predicted_targets = np.array(predicted_targets)
    
    # Converting to a range of 0 to 1
    min = np.min(predicted_targets)
    max = np.max(predicted_targets)
    for i in range(0,len(predicted_targets)):
        predicted_targets[i] = (((predicted_targets[i]-min)*1)/(max-min)) + 0 # convert to range 0 to 1

    # Initializing the data set which will be exported to csv format 
    output_data_set = []
    
    for i in range(0,len(checking_data_set)) :
        if i == 0 :
            #output_data_set.append(header)
            #output_data_set[i].append('Predicted Target')
            continue
        else :
            output_data_set.append(checking_data_set[i-1])
            output_data_set[i].append(predicted_targets[i])
    ###
            
    #return chromos[solution_index],output_data_set;
    return chromos[solution_index];
                 


### Main ###
def main(): 
    
    logging.basicConfig(filename='D:\EFL\logging.log',level=logging.DEBUG)
    
    logging.info("Starting file import.")
    
    trialrun = False
        
    ### Standardize the variables - this is done as follows :
    1) Dummy variables - Either 1/0 - these are left as they are
    2) Nominal variables - These are converted to binary dummy variables. eg: 5 will be converted to 101 (3 new variables created)
    3) Ordinal variables - These will be converted to a scale of 0 to 1
    4) Continuous variables - These will be converted to a scale of 0  to 1
    ###
    
    ### Step 1 - Convert the nominal variables to binary form dummy variables ###
    ### This step is done in advance in SAS ###
    
    
    ### Step 2 -  Find the max value in each column ###
    
            
    ### Step 3 - Normalize the selected variables ###
    
    
    ### Step 4 - Run genetic algorithm ###
    
    # This step extracts the base data into a base_data 2d array and converts it into a numpy array
    original_data_set, complete_data_set, header = dataExtraction(trialrun)
    
    complete_data_set = np.array(complete_data_set)
        
    # This step splits the data set into 5 clusters, and creates a chromosome solution for each of them individually
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    cluster_4 = []
    cluster_5 = []
    
    for i in range(0,len(complete_data_set)):
        if complete_data_set[i][1] == '1' :
            cluster_1.append(complete_data_set[i])
        elif complete_data_set[i][1] == '2' :
            cluster_2.append(complete_data_set[i])
        elif complete_data_set[i][1] == '3' :
            cluster_3.append(complete_data_set[i])
        elif complete_data_set[i][1] == '4' :
            cluster_4.append(complete_data_set[i])
        elif complete_data_set[i][1] == '5' :
            cluster_5.append(complete_data_set[i])
            
    
    cluster_1 = np.array(cluster_1)
    cluster_2 = np.array(cluster_2)
    cluster_3 = np.array(cluster_3)
    cluster_4 = np.array(cluster_4)
    cluster_5 = np.array(cluster_5)
    

    #cluster_solution_1, output_data_set_1 = cluster_run(cluster_1, trialrun)
    #cluster_solution_2, output_data_set_2 = cluster_run(cluster_2, trialrun)
    #cluster_solution_3, output_data_set_3 = cluster_run(cluster_3, trialrun)
    #cluster_solution_4, output_data_set_4 = cluster_run(cluster_4, trialrun)
    #cluster_solution_5, output_data_set_5 = cluster_run(cluster_5, trialrun)
    
    cluster_solution_1 = cluster_run(cluster_1, trialrun, 1)
    cluster_solution_2 = cluster_run(cluster_2, trialrun, 2)
    cluster_solution_3 = cluster_run(cluster_3, trialrun, 3)
    cluster_solution_4 = cluster_run(cluster_4, trialrun, 4)
    cluster_solution_5 = cluster_run(cluster_5, trialrun, 5)
    
    print "\n\n\n\n"
    print "Cluster 1 Solution : ", cluster_solution_1
    print "Cluster 2 Solution : ", cluster_solution_2
    print "Cluster 3 Solution : ", cluster_solution_3
    print "Cluster 4 Solution : ", cluster_solution_4
    print "Cluster 5 Solution : ", cluster_solution_5
    
    with open('D:\EFL\EFL external data set-final_output.csv', 'wb') as output:
        writer = csv.writer(output, delimiter=',')
        writer.writerows(output_data_set)

if __name__ == "__main__":
    main()
