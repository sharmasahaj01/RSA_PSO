import matplotlib.pyplot as plt
import numpy as np
from algorithms import HybridRSA_PSO,OriginalRSA,WOA,GWO,MPA,PSO,COA,HO,DO
from benchmarks import CEC2017_F2,PressureVessel,WeldedBeam
from experiment import run_experiment

if __name__ == "__main__":
    # 1. Configure experiments
    algorithms = [HybridRSA_PSO, OriginalRSA,WOA,GWO,MPA,PSO,COA,HO,DO]  # Add competitors here
    benchmarks = [WeldedBeam()]

    # 2. Run all experiments
    results = run_experiment(algorithms, benchmarks)

    # 3. Generate convergence plot
    plt.figure(figsize=(10, 6))
    for algo, data in results['WeldedBeam'].items():
        median_history = np.median([h for h in data['histories']], axis=0)
        plt.plot(median_history, label=algo)

    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Fitness')
    plt.yscale('log')
    plt.legend()
    plt.savefig('convergence.png')
    plt.show()
