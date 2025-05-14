import numpy as np

def run_experiment(algorithms, benchmarks):
    results = {}
    
    for bench in benchmarks:
        bname = bench.__class__.__name__
        results[bname] = {}
        print(f"\n=== Testing {bname} ===")
        
        for Algo in algorithms:
            all_scores = []
            all_histories = []
            
            for _ in range(50):  # 50 independent runs
                optimizer = Algo(pop_size=1200, dim=bench.dim, 
                               max_fe=60000, benchmark=bench)
                history = optimizer.run()
                final_score = history[-1]
                all_scores.append(final_score)
                all_histories.append(history)
            
            # Store metrics
            results[bname][Algo.__name__] = {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'scores': all_scores,
                'histories': all_histories
            }
            print(f"{Algo.__name__}: Mean={results[bname][Algo.__name__]['mean']:.2e} ± {results[bname][Algo.__name__]['std']:.2e}")
    
    return results

