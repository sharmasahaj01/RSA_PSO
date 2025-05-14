import matplotlib.pyplot as plt
import numpy as np
from algorithms import HybridRSA_PSO,OriginalRSA,WOA,GWO,MPA,PSO,COA,HO,DO
from benchmarks import CEC2014_F3,PressureVessel,WeldedBeam # Import as many functions as you want from "benchmarks.py"
from experiment import run_experiment

if __name__ == "__main__":
    # 1. Configure experiments
    algorithms = [HybridRSA_PSO, OriginalRSA,WOA,GWO,MPA,PSO,COA,HO,DO]  # Add competitors here
    benchmarks = [CEC2014_F3(dim=30),WeldedBeam()] # Add benchmark functions here as an array(Algorithms will be evaluated on all benchmarks in the array)

    # 2. Run all experiments
    results = run_experiment(algorithms, benchmarks)

    # 3. Generate convergence plot
    plt.figure(figsize=(10, 6))
    for algo, data in results['CEC2014_F3'].items(): # Mention the function for generating the convergence graph(single graph will be  generated at a time)
        median_history = np.median([h for h in data['histories']], axis=0)
        plt.plot(median_history, label=algo)

    plt.xlabel('Number of Iterations')
    plt.ylabel('Fitness Function')
    plt.yscale('log')
    plt.legend()
    plt.savefig('convergence.png')
    plt.show()
