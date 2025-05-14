import numpy as np
from scipy.special import gamma

class HybridRSA_PSO:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.position = None
        self.velocity = np.zeros((pop_size, dim))
        self.gbest = None
        self.gbest_score = np.inf
        self.feval_count = 0
        self.history = []

    def initialize(self):
        # Chaotic initialization
        chaotic = [np.random.rand()]
        for _ in range(self.pop_size-1):
            chaotic.append(np.cos(np.arccos(chaotic[-1])))
        self.position = np.array(chaotic).reshape(-1,1) * np.random.uniform(-100, 100, (self.pop_size, self.dim))
        
        # Evaluate initial population
        scores = [self.benchmark(x) for x in self.position]
        self.gbest = self.position[np.argmin(scores)]
        self.gbest_score = min(scores)
        self.feval_count = self.pop_size
        self.history.append(self.gbest_score)

    def update(self, t):
        T = self.max_fe // self.pop_size
        alpha = 0.1
        R = alpha * (t/T)
        ES = 2 * np.random.rand() * (1 - t/T)
        
        # Dynamic PSO parameters
        w = 0.9 - 0.5*(t/T)**0.5
        c1 = 2.5 * np.cos(np.pi*t/T)
        c2 = 2.5 * np.sin(np.pi*t/T)
        
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            
            # RSA phase logic
            if t <= T/4:  # High walking (70% RSA, 30% PSO)
                neighbor = self.position[np.random.randint(self.pop_size)]
                rsa_term = self.gbest*(1 - R) + np.random.rand()*(neighbor - self.position[i])
                pso_term = w*self.velocity[i] + c1*np.random.rand()*(self.gbest - self.position[i])
                new_pos = 0.7*rsa_term + 0.3*pso_term
                
            elif t <= T/2:  # Belly walking (60% RSA, 40% PSO)
                neighbor = self.position[np.random.randint(self.pop_size)]
                rsa_term = self.gbest*(1 + R) + ES*np.random.rand()*(neighbor - self.position[i])
                pso_term = w*self.velocity[i] + c2*np.random.rand()*(self.gbest - self.position[i])
                new_pos = 0.6*rsa_term + 0.4*pso_term
                
            elif t <= 3*T/4:  # Hunting coordination (80% RSA, 20% PSO)
                rsa_term = self.gbest*(1 - R) + np.random.rand()*(self.gbest - self.position[i])
                pso_term = w*self.velocity[i] + 0.5*(c1 + c2)*np.random.rand()*(self.gbest - self.position[i])
                new_pos = 0.8*rsa_term + 0.2*pso_term
                
            else:  # Hunting cooperation (90% RSA, 10% PSO)
                rsa_term = self.gbest*alpha - np.random.rand()*(self.gbest - self.position[i])
                pso_term = w*self.velocity[i] + 0.1*(c1 + c2)*np.random.rand()*(self.gbest - self.position[i])
                new_pos = 0.9*rsa_term + 0.1*pso_term
            
            # Update position and velocity
            new_pos = np.clip(new_pos, -100, 100)
            self.velocity[i] = new_pos - self.position[i]
            self.position[i] = new_pos
            
            # Evaluate
            score = self.benchmark(new_pos)
            if score < self.gbest_score:
                self.gbest = new_pos.copy()
                self.gbest_score = score
                
            self.feval_count += 1
        self.history.append(self.gbest_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history

class OriginalRSA:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.position = None
        self.best_solution = None
        self.best_score = np.inf
        self.feval_count = 0
        self.history = []

    def initialize(self):
        self.position = np.random.uniform(-100, 100, (self.pop_size, self.dim))
        scores = [self.benchmark(x) for x in self.position]
        self.best_score = np.min(scores)
        self.best_solution = self.position[np.argmin(scores)]
        self.feval_count = self.pop_size
        self.history.append(self.best_score)

    def update(self, t):
        T = self.max_fe // self.pop_size
        alpha = 0.1
        R = alpha * (t/T)
        ES = 2 * np.random.rand() * (1 - t/T)
        
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            
            if t <= T/4:  # High walking
                new_pos = self.best_solution*(1 - R) + np.random.rand() * (
                    self.position[np.random.randint(self.pop_size)] - self.position[i])
            elif t <= T/2:  # Belly walking
                new_pos = self.best_solution*(1 + R) + ES * np.random.rand() * (
                    self.position[np.random.randint(self.pop_size)] - self.position[i])
            elif t <= 3*T/4:  # Hunting coordination
                new_pos = self.best_solution*(1 - R) + np.random.rand() * (
                    self.best_solution - self.position[i])
            else:  # Hunting cooperation
                new_pos = self.best_solution*alpha - np.random.rand() * (
                    self.best_solution - self.position[i])
            
            new_pos = np.clip(new_pos, -100, 100)
            score = self.benchmark(new_pos)
            
            if score < self.best_score:
                self.best_solution = new_pos.copy()
                self.best_score = score
                
            if score < self.benchmark(self.position[i]):
                self.position[i] = new_pos
                
            self.feval_count += 1
        self.history.append(self.best_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history

class WOA:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.position = None
        self.leader_score = np.inf
        self.leader_pos = None
        self.feval_count = 0
        self.history = []

    def initialize(self):
        self.position = np.random.uniform(-100, 100, (self.pop_size, self.dim))
        scores = [self.benchmark(x) for x in self.position]
        self.leader_pos = self.position[np.argmin(scores)].copy()
        self.leader_score = min(scores)
        self.feval_count = self.pop_size
        self.history.append(self.leader_score)

    def update(self, t):
        T = self.max_fe // self.pop_size
        a = 2 - t * (2 / T)  # Decreases linearly from 2 to 0
        
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            
            r1, r2 = np.random.rand(2)
            A = 2 * a * r1 - a  # Exploration parameter
            C = 2 * r2  # Spiral shape parameter
            
            # Distance to leader
            D = np.abs(C * self.leader_pos - self.position[i])
            
            if np.random.rand() < 0.5:
                if np.abs(A) < 1:  # Exploitation: Bubble-net attacking
                    new_pos = self.leader_pos - A * D
                else:  # Exploration: Global search
                    rand_idx = np.random.randint(self.pop_size)
                    new_pos = self.position[rand_idx] - A * D
            else:  # Spiral update
                distance = np.linalg.norm(self.leader_pos - self.position[i])
                new_pos = distance * np.exp(0.5 * np.random.randn()) * np.cos(2*np.pi*np.random.rand()) + self.leader_pos
            
            new_pos = np.clip(new_pos, -100, 100)
            score = self.benchmark(new_pos)
            
            if score < self.leader_score:
                self.leader_pos = new_pos.copy()
                self.leader_score = score
                
            if score < self.benchmark(self.position[i]):
                self.position[i] = new_pos
                
            self.feval_count += 1
        self.history.append(self.leader_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history

class GWO:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.position = None
        self.alpha_score = np.inf
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        self.feval_count = 0
        self.history = []

    def initialize(self):
        self.position = np.random.uniform(-100, 100, (self.pop_size, self.dim))
        scores = np.array([self.benchmark(x) for x in self.position])
        self._update_leaders(scores)
        self.feval_count = self.pop_size
        self.history.append(self.alpha_score)

    def _update_leaders(self, scores):
        sorted_indices = np.argsort(scores)
        self.alpha_pos = self.position[sorted_indices[0]].copy()
        self.beta_pos = self.position[sorted_indices[1]].copy()
        self.delta_pos = self.position[sorted_indices[2]].copy()
        self.alpha_score = scores[sorted_indices[0]]

    def update(self, t):
        T = self.max_fe // self.pop_size
        a = 2 - t * (2 / T)  # Decreases linearly from 2 to 0
        
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            
            # Update parameters
            A1 = 2 * a * np.random.rand() - a
            C1 = 2 * np.random.rand()
            A2 = 2 * a * np.random.rand() - a
            C2 = 2 * np.random.rand()
            A3 = 2 * a * np.random.rand() - a
            C3 = 2 * np.random.rand()
            
            # Calculate new position
            X1 = self.alpha_pos - A1 * np.abs(C1 * self.alpha_pos - self.position[i])
            X2 = self.beta_pos - A2 * np.abs(C2 * self.beta_pos - self.position[i])
            X3 = self.delta_pos - A3 * np.abs(C3 * self.delta_pos - self.position[i])
            new_pos = (X1 + X2 + X3) / 3
            
            # Evaluate and update
            new_pos = np.clip(new_pos, -100, 100)
            score = self.benchmark(new_pos)
            
            if score < self.benchmark(self.position[i]):
                self.position[i] = new_pos
                
            self.feval_count += 1
            
        # Update leaders after full iteration
        scores = np.array([self.benchmark(x) for x in self.position])
        self._update_leaders(scores)
        self.history.append(self.alpha_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history

class MPA:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.prey = None
        self.best_pos = None
        self.best_score = np.inf
        self.feval_count = 0
        self.history = []
        self.lb = -100
        self.ub = 100

    def initialize(self):
        self.prey = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        scores = [self.benchmark(x) for x in self.prey]
        self.best_score = min(scores)
        self.best_pos = self.prey[np.argmin(scores)].copy()
        self.feval_count = self.pop_size
        self.history.append(self.best_score)

    def _levy_step(self, beta=1.5):
        sigma = (gamma(1+beta) * np.sin(np.pi*beta/2) / 
                (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.abs(np.random.normal(0, 1, self.dim))
        step = 0.01 * u / (v**(1/beta))
        return np.clip(step, -1, 1)


    def update(self, t):
        T = max(1, self.max_fe // self.pop_size)  # Prevent division by zero
        CF = max(0.1, (1 - t/T)**(2*t/T))  # Lower bound for CF
        
        # Phase-based movement with population scaling
        if t < T//3:
            # High velocity: Intensive exploration
            for i in range(self.pop_size):
                step = self._levy_step() * (self.best_pos - self.prey[i])
                self.prey[i] += 0.1 * step * (np.random.rand() + 0.1)  # Reduced step size
        elif t < 2*T//3:
            # Unit velocity: Balanced search
            for i in range(self.pop_size):
                if i < self.pop_size//2:
                    step = self._levy_step() * (self.best_pos - self.prey[i])
                    self.prey[i] += 0.2 * CF * step
                else:
                    step = np.random.randn(self.dim) * (self.best_pos - self.prey[i])
                    self.prey[i] += 0.5 * CF * step
        else:
            # Low velocity: Local exploitation
            for i in range(self.pop_size):
                if np.random.rand() < 0.3:
                    step = 0.1 * CF * np.random.randn(self.dim)
                    self.prey[i] += step * (self.best_pos - self.prey[i])

        # Apply boundaries
        self.prey = np.clip(self.prey, self.lb, self.ub)
        
        # Evaluate and update best
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            score = self.benchmark(self.prey[i])
            self.feval_count += 1
            if score < self.best_score:
                self.best_score = score
                self.best_pos = self.prey[i].copy()
        
        self.history.append(self.best_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history

class PSO:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_score = None
        self.gbest = None
        self.gbest_score = np.inf
        self.feval_count = 0
        self.history = []
        self.lb = -100
        self.ub = 100

    def initialize(self):
        self.position = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.velocity = np.zeros((self.pop_size, self.dim))
        self.pbest = self.position.copy()
        self.pbest_score = np.array([self.benchmark(x) for x in self.position])
        self.gbest_score = np.min(self.pbest_score)
        self.gbest = self.position[np.argmin(self.pbest_score)].copy()
        self.feval_count = self.pop_size
        self.history.append(self.gbest_score)

    def update(self, t):
        w = 0.9 - 0.5 * (t / (self.max_fe // self.pop_size))  # Inertia weight decreases
        c1 = 2.0  # Cognitive (particle)
        c2 = 2.0  # Social (swarm)

        for i in range(self.pop_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocity[i] = (
                w * self.velocity[i]
                + c1 * r1 * (self.pbest[i] - self.position[i])
                + c2 * r2 * (self.gbest - self.position[i])
            )
            self.position[i] += self.velocity[i]
            self.position[i] = np.clip(self.position[i], self.lb, self.ub)

            score = self.benchmark(self.position[i])
            self.feval_count += 1
            if score < self.pbest_score[i]:
                self.pbest_score[i] = score
                self.pbest[i] = self.position[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.position[i].copy()
        self.history.append(self.gbest_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history

class COA:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.positions = None
        self.best_pos = None
        self.best_score = np.inf
        self.feval_count = 0
        self.history = []
        self.lb = -100
        self.ub = 100

    def _chaotic_init(self):
        # Logistic map for chaotic initialization [Search Result 6]
        chaotic = np.zeros((self.pop_size, self.dim))
        x = np.random.rand()
        for i in range(self.pop_size):
            x = 4 * x * (1 - x)  # Logistic map equation
            chaotic[i] = x
        return self.lb + chaotic * (self.ub - self.lb)

    def initialize(self):
        self.positions = self._chaotic_init()
        scores = [self.benchmark(x) for x in self.positions]
        self.best_score = min(scores)
        self.best_pos = self.positions[np.argmin(scores)].copy()
        self.feval_count = self.pop_size
        self.history.append(self.best_score)

    def update(self, t):
        T = self.max_fe // self.pop_size
        a = 0.3 * (1 - t/T)  # Nonlinear inertia weight [Search Result 6]
        
        # Phase 1: Attack on iguanas (Exploration)
        for i in range(self.pop_size//2):
            # Random iguana position [Search Result 2]
            iguana = self.best_pos * (1 + np.random.randn() * 0.1)
            
            # Position update equation [Search Result 7]
            r = np.random.rand(self.dim)
            self.positions[i] += a * (iguana - 2 * r * self.positions[i])
        
        # Phase 2: Escape predators (Exploitation)
        for i in range(self.pop_size//2, self.pop_size):
            # Local search with adaptive bounds [Search Result 6]
            lb_local = self.positions[i] - 0.1*(self.ub - self.lb)*(t/T)
            ub_local = self.positions[i] + 0.1*(self.ub - self.lb)*(t/T)
            self.positions[i] = np.clip(self.positions[i] + 
                                      np.random.randn(self.dim)*a*(ub_local - lb_local),
                                      self.lb, self.ub)
        
        # Adaptive T-distribution mutation [Search Result 6]
        if np.random.rand() < 0.2:
            mutation = np.random.standard_t(self.dim, size=self.positions.shape)
            self.positions = np.clip(self.positions + 0.1*mutation, self.lb, self.ub)
        
        # Evaluate and update best
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            score = self.benchmark(self.positions[i])
            self.feval_count += 1
            if score < self.best_score:
                self.best_score = score
                self.best_pos = self.positions[i].copy()
        
        self.history.append(self.best_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history    

class HO:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.positions = None
        self.best_pos = None
        self.best_score = np.inf
        self.feval_count = 0
        self.history = []
        self.lb = -100
        self.ub = 100

    def _logistic_map(self, n):
        # Chaotic initialization [Source 2,6]
        chaotic = np.zeros(n)
        x = np.random.rand()
        for i in range(n):
            x = 4 * x * (1 - x)
            chaotic[i] = x
        return chaotic

    def initialize(self):
        # Chaotic population initialization
        self.positions = np.array([self._logistic_map(self.dim) for _ in range(self.pop_size)])
        self.positions = self.lb + self.positions * (self.ub - self.lb)
        
        scores = [self.benchmark(x) for x in self.positions]
        self.best_score = min(scores)
        self.best_pos = self.positions[np.argmin(scores)].copy()
        self.feval_count = self.pop_size
        self.history.append(self.best_score)

    def update(self, t):
        T = self.max_fe // self.pop_size
        a = 0.3 * (1 - t/T)  # Nonlinear decay factor [Source 4]
        
        # Phase 1: Position update in water (Exploration)
        r1 = np.random.rand(self.pop_size, self.dim)
        phase1 = self.positions + r1 * (self.best_pos - self.positions)
        
        # Phase 2: Defense against predators
        r2 = np.random.rand(self.pop_size, self.dim)
        predator = self.positions[np.random.randint(self.pop_size)]
        phase2 = phase1 + r2 * (predator - self.positions)
        
        # Phase 3: Evasion strategy (Exploitation)
        r3 = np.random.rand(self.pop_size, self.dim)
        phase3 = phase2 + a * r3 * (self.best_pos - phase2)
        
        # Combined update with boundary control
        self.positions = np.clip(phase3, self.lb, self.ub)
        
        # Evaluate and update best
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            score = self.benchmark(self.positions[i])
            self.feval_count += 1
            if score < self.best_score:
                self.best_score = score
                self.best_pos = self.positions[i].copy()
        
        self.history.append(self.best_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history

class DO:
    def __init__(self, pop_size, dim, max_fe, benchmark):
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.benchmark = benchmark
        self.positions = None
        self.best_pos = None
        self.best_score = np.inf
        self.feval_count = 0
        self.history = []
        self.lb = -100
        self.ub = 100

    def _levy_flight(self, beta=1.5):
        sigma = (gamma(1+beta)*np.sin(np.pi*beta/2) / 
                (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.abs(np.random.normal(0, 1, self.dim))
        return 0.01 * u / (v**(1/beta))

    def _brownian_motion(self):
        return np.random.normal(0, 1, self.dim)

    def initialize(self):
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        scores = [self.benchmark(x) for x in self.positions]
        self.best_score = min(scores)
        self.best_pos = self.positions[np.argmin(scores)].copy()
        self.feval_count = self.pop_size
        self.history.append(self.best_score)

    def update(self, t):
        T = self.max_fe // self.pop_size
        alpha = 1 - t/T  # Adaptive parameter from [Search Result 6]
        
        # Ascending phase (Exploration)
        for i in range(self.pop_size):
            if np.random.rand() < 0.7:  # Sunny weather [Search Result 4]
                LF = self._levy_flight()
                self.positions[i] += alpha * LF * (self.ub - self.lb)
            else:  # Rainy weather
                self.positions[i] += 0.1 * alpha * np.random.randn(self.dim)
        
        # Descending phase (Transition)
        mean_pos = np.mean(self.positions, axis=0)
        for i in range(self.pop_size):
            BM = self._brownian_motion()
            self.positions[i] = alpha * self.positions[i] + (1-alpha)*mean_pos + 0.1*BM
        
        # Landing phase (Exploitation)
        for i in range(self.pop_size):
            LF = self._levy_flight()
            self.positions[i] = self.best_pos + 0.1*LF*(self.ub - self.lb)
        
        # Apply boundaries
        self.positions = np.clip(self.positions, self.lb, self.ub)
        
        # Evaluate and update best
        for i in range(self.pop_size):
            if self.feval_count >= self.max_fe:
                break
            score = self.benchmark(self.positions[i])
            self.feval_count += 1
            if score < self.best_score:
                self.best_score = score
                self.best_pos = self.positions[i].copy()
        
        self.history.append(self.best_score)

    def run(self):
        self.initialize()
        T = self.max_fe // self.pop_size
        for t in range(T):
            self.update(t)
        return self.history
