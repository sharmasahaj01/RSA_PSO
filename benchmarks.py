import numpy as np
from opfunu.cec_based import cec2014, cec2017, cec2020, cec2022

########## CEC2014 ###############
class CEC2014_F1(cec2014.F12014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim

    def __call__(self, x):
        return self.evaluate(x)
    
class CEC2014_F2(cec2014.F22014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
    def __call__(self, x):
        return self.evaluate(x)

class CEC2014_F3(cec2014.F32014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
    def __call__(self, x):
        return self.evaluate(x)    

class CEC2014_F4(cec2014.F42014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
    def __call__(self, x):
        return self.evaluate(x)

class CEC2014_F5(cec2014.F52014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
    def __call__(self, x):
        return self.evaluate(x)

class CEC2014_F6(cec2014.F62014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
    def __call__(self, x):
        return self.evaluate(x)
    
class CEC2014_F23(cec2014.F232014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
    def __call__(self, x):
        return self.evaluate(x) 

class CEC2014_F24(cec2014.F242014):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
    def __call__(self, x):
        return self.evaluate(x)
################# CEC2017 ################ 
class CEC2017_F1(cec2017.F12017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2017_F2(cec2017.F22017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2017_F3(cec2017.F32017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2017_F4(cec2017.F42017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2017_F5(cec2017.F52017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2017_F6(cec2017.F62017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2017_F19(cec2017.F192017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2017_F20(cec2017.F202017):
    def __init__(self, dim=30):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

########## CEC2020 ###########
class CEC2020_F1(cec2020.F12020):
    def __init__(self, dim=10):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2020_F2(cec2020.F22020):
    def __init__(self, dim=10):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2020_F3(cec2020.F32020):
    def __init__(self, dim=10):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2020_F4(cec2020.F42020):
    def __init__(self, dim=10):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2020_F5(cec2020.F52020):
    def __init__(self, dim=10):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

########## CEC2022 ###########
class CEC2022_F1(cec2022.F12022):
    def __init__(self, dim=12):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100) 
        return self.evaluate(x)

class CEC2022_F2(cec2022.F22022):
    def __init__(self, dim=12):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2022_F3(cec2022.F32022):
    def __init__(self, dim=12):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2022_F4(cec2022.F42022):
    def __init__(self, dim=12):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)

class CEC2022_F5(cec2022.F52022):
    def __init__(self, dim=12):
        super().__init__(ndim=dim)
        self.dim = dim
        
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return self.evaluate(x)
    
##### Engineering Problems ########
class PressureVessel:
    def __init__(self):
        self.dim = 4
        self.lb = np.array([0.0625, 0.0625, 10.0, 10.0])
        self.ub = np.array([6.1875, 6.1875, 200.0, 200.0])
        
    def __call__(self, x):
        x = np.clip(x, self.lb, self.ub)  # Enforce bounds
        Ts, Th, R, L = x
        
        # Objective: Minimize total cost
        cost = (0.6224*Ts*R*L + 1.7781*Th*R**2 + 
                3.1661*Ts**2*L + 19.84*Ts**2*R)
        
        # Constraints with penalty
        penalty = 0
        penalty += 1e6 * max(0, 0.0193*R - Ts)    # Ts >= 0.0193R
        penalty += 1e6 * max(0, 0.00954*R - Th)   # Th >= 0.00954R
        penalty += 1e6 * max(0, 750*1728 - (np.pi*R**2*L + (4/3)*np.pi*R**3))
        penalty += 1e6 * max(0, L - 240)          # L <= 240
        
        return cost + penalty
    
class WeldedBeam:
    def __init__(self):
        self.dim = 4
        self.lb = np.array([0.1, 0.1, 0.1, 0.1])
        self.ub = np.array([2.0, 10.0, 10.0, 2.0])
        
    def __call__(self, x):
        x = np.clip(x, self.lb, self.ub)
        h, l, t, b = x
        
        # Objective: Minimize fabrication cost
        cost = 1.10471*h**2*l + 0.04811*t*b*(14.0 + l)
        
        # Constraints
        tau = 13600  # psi
        sigma = 30000 # psi
        delta = 0.25  # in
        
        M = 6000*(14 + 0.5*l)
        J = np.sqrt(2)*h*l*(l**2/6 + (h + t)**2/2)
        stress = 6000/(np.sqrt(2)*h*l) + 6*6000*14/(t**2*b)
        
        penalty = 0
        penalty += 1e6 * max(0, stress - tau)
        penalty += 1e6 * max(0, 6*6000*14/(t**2*b) - sigma)
        penalty += 1e6 * max(0, 2.1952/(t**3*b) - delta)
        
        return cost + penalty

class GearTrainDesign:
    def __init__(self):
        self.dim = 4  # Number of gears (T1, T2, T3, T4)
        self.lb = np.array([12, 12, 12, 12])  # Minimum teeth (AGMA standard)
        self.ub = np.array([60, 60, 60, 60])  # Maximum practical teeth count
        
    def __call__(self, x):
        # Convert to integers and enforce bounds
        x = np.clip(x, self.lb, self.ub)
        T1, T2, T3, T4 = np.round(x).astype(int)
        
        # Design parameters
        desired_ratio = 4.0      # Target gear ratio
        max_center_dist = 150    # Maximum center distance (mm)
        min_teeth = 17           # Minimum teeth to avoid undercutting
        module = 2               # Gear module (mm/tooth)
        
        # 1. Calculate actual gear ratio
        actual_ratio = (T2 * T4) / (T1 * T3)
        ratio_error = (actual_ratio - desired_ratio)**2
        
        # 2. Calculate approximate volume (proxy for material cost)
        volume = (T1 + T2 + T3 + T4) * module**2  # Proportional to teeth count
        
        # 3. Center distance calculation
        center_dist = 0.5*module*(T1 + T2 + T3 + T4)
        
        # 4. Contact ratio constraint (must be > 1.2 for smooth operation)
        contact_ratio = (np.sqrt(T1**2 - (T1-2)**2) + 
                        np.sqrt(T2**2 - (T2-2)**2)) / (2*np.pi*module)
        
        # 5. Initialize penalty
        penalty = 0
        
        # 6. Apply constraints with quadratic penalty
        penalty += 1e6 * max(0, center_dist - max_center_dist)**2
        penalty += 1e6 * sum(max(0, min_teeth - t)**2 for t in [T1, T2, T3, T4])
        penalty += 1e6 * max(0, 1.2 - contact_ratio)**2
        
        # 7. Combined objective (weighted sum)
        obj = ratio_error + 0.001*volume + penalty
        
        return obj

class ThreeBarTruss:
    def __init__(self):
        self.dim = 2  # Design variables: x1 (A1/A3) and x2 (A2)
        self.lb = np.array([0.1, 0.1])  # Lower bounds (cm^2)
        self.ub = np.array([1.0, 1.0])   # Upper bounds (cm^2)
        
        # Constants from CEC specifications
        self.P = 2.0      # Applied load (kN/cm^2)
        self.sigma = 2.0  # Allowable stress (kN/cm^2)
        self.H = 100.0    # Height (cm)
        
    def __call__(self, x):
        x = np.clip(x, self.lb, self.ub)
        x1, x2 = x
        
        # 1. Calculate volume (objective to minimize)
        volume = (2*np.sqrt(2)*x1 + x2) * self.H
        
        # 2. Calculate stress constraints
        denominator = np.sqrt(2)*x1**2 + 2*x1*x2
        
        g1 = (np.sqrt(2)*x1 + x2)/denominator * self.P - self.sigma
        g2 = x2/denominator * self.P - self.sigma
        g3 = 1/(np.sqrt(2)*x2 + x1) * self.P - self.sigma
        
        # 3. Apply penalty method
        penalty = 0
        penalty += 1e6 * max(0, g1)**2
        penalty += 1e6 * max(0, g2)**2
        penalty += 1e6 * max(0, g3)**2
        
        return volume + penalty

class SpeedReducer:
    def __init__(self):
        self.dim = 7  # Design variables
        self.lb = np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0])
        self.ub = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
        
    def __call__(self, x):
        x = np.clip(x, self.lb, self.ub)
        x1, x2, x3, x4, x5, x6, x7 = x
        x3 = int(round(x3))  # Discrete variable handling
        
        # Objective: Minimize weight (kg)
        weight = (0.7854 * x1 * x2**2 * (3.3333*x3**2 + 14.9334*x3 - 43.0934) 
                - 1.508 * x1 * (x6**2 + x7**2) 
                + 7.4777 * (x6**3 + x7**3) 
                + 0.7854 * (x4 * x6**2 + x5 * x7**2))
        
        # Constraints with penalty method
        penalty = 0
        penalty += 1e6 * max(0, 27/(x1*x2**2*x3) - 1)**2          # Bending stress
        penalty += 1e6 * max(0, 397.5/(x1*x2**2*x3**2) - 1)**2    # Surface stress
        penalty += 1e6 * max(0, (1.93*x4**3)/(x2*x3*x6**4) - 1)**2 # Shaft 1 deflection
        penalty += 1e6 * max(0, (1.93*x5**3)/(x2*x3*x7**4) - 1)**2 # Shaft 2 deflection
        penalty += 1e6 * max(0, np.sqrt((745*x4/(x2*x3))**2 + 16.9e6)/(110*x6**3) - 1)**2  # Shaft 1 stress
        penalty += 1e6 * max(0, np.sqrt((745*x5/(x2*x3))**2 + 157.5e6)/(85*x7**3) - 1)**2  # Shaft 2 stress
        penalty += 1e6 * max(0, (x2*x3)/40 - 1)**2                # Space constraint
        penalty += 1e6 * max(0, (5*x2)/x1 - 1)**2                 # Module limit
        penalty += 1e6 * max(0, x1/(12*x2) - 1)**2                # Face width ratio
        penalty += 1e6 * max(0, (1.5*x6 + 1.9)/x4 - 1)**2         # Shaft 1 diameter
        penalty += 1e6 * max(0, (1.1*x7 + 1.9)/x5 - 1)**2         # Shaft 2 diameter
        
        return weight + penalty
