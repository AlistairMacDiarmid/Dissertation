from __future__ import annotations
import math
import numpy as np

from wind_scenario import WindScenario

class KusiakEnergyEvaluator:
    """
    Python port of Java/KusiakLayoutEvaluator

    The key outputs:
    - evaluate(layout): returns energyCost (minimise)
    - evaluate_2014(layout): returns wakeFreeRatio
    - getEnergyOutput(): total energy capture (energyCapture)
    - getWakeFreeRatio(): wakeFreeRatio
    - getEnergyCost(): energy cost
    - getEnergyOutputs(): tspe[direction, turbine] annual energy contributions
    """

    FAC = math.pi / 180.0

    def __init__(self, scenario: WindScenario):
        self.initialise(scenario)

    def initialise(self, scenario: WindScenario) -> None:
        self.tspe: np.ndarray | None = None
        self.tpositions: np.ndarray | None = None
        self.energyCapture: float = 0.0
        self.wakeFreeRatio: float = 0.0
        self.energyCost: float = float("inf")
        self.scenario: WindScenario = scenario
        self.nEvals: int = 0

    def evaluate(self, layout:np.ndarray) -> float:
        """
        return the cost-based objective (energyCost).
        an invalid layout will return +inf
        :param layout:
        :return: self.energyCost:
        """
        ct = 750000.0
        cs = 8000000.0
        m = 30.0
        r = 0.03
        y = 20.0
        com = 20000.0

        wfr = self.evaluate_2014(layout)
        if wfr <= 0.0:
            self.energyCost = float("inf")
            return self.energyCost

        n = int(np.asarray(layout).shape[0])

        # Java: ((1 - (1+r)^(-y))/r)
        finance_factor = (1.0 - math.pow(1.0 + r, -y)) / r

        # Java denominator: 8760 * wakeFreeEnergy * wfr * n
        annual_energy = 8760.0 * self.scenario.wakeFreeEnergy * wfr * n

        capex = ct * n + cs * math.floor(n / m)
        scale = 0.666667 + 0.333333 * math.exp(-0.00174 * n * n)
        opex = com * n

        # Java:
        # energyCost = (((capex)*scale + opex) / finance_factor / annual_energy) + 0.1/n
        self.energyCost = ((capex * scale + opex) / finance_factor) / annual_energy + (0.1 / n)
        return self.energyCost

    def evaluate_2014(self, layout:np.ndarray) -> float:
        """
        computes energyCapture, wakeFreeRatio and tspe
        returns the wakeFreeRatio, or 0 if the layout is invalid
        :param layout:
        :return:
        """
        self.nEvals +=1

        layout = np.asarray(layout, dtype=float)
        if layout.ndim != 2 or layout.shape[1] != 2:
            raise ValueError("invalid layout. the layout must be in a (n,2) array of x,y turbine positions")

        #Java copies the layout into tpositions
        self.tpositions = np.array(layout, dtype=float, copy=True)

        self.energyCapture = 0.0

        if not self.checkConstraint(self.tpositions):
            #Java invalid-case reset
            self.energyCapture = 0.0
            self.wakeFreeRatio = 0.0
            self.tspe = None
            return 0.0

        n_turb = self.tpositions.shape[0]
        n_dirs = self.scenario.thetas.shape[0]

        #Java: tspe = new double[thetas.length][tpositions.length]
        self.tspe = np.zeros((n_dirs, n_turb), dtype=float)

        for turb in range(n_turb):
            for thets in range (n_dirs):
                #total wake velocity deficit for this turbine in this direction
                totalVdef = self.calculateWakeTurbine(turb, thets)
                #cTurb = scenario.c[thets] * (1-totalVdef)
                cTurb = float(self.scenario.c[thets]) * (1.0 - totalVdef)

                #annual power output for turbine+direction
                tint = float(self.scenario.thetas[thets][1] - self.scenario.thetas[thets][0])
                w = float(self.scenario.omegas[thets])
                ki = float(self.scenario.ks[thets])

                totalPow = 0.0
                vints = self.scenario.vints

                #integrate over speed bins using Weibull CDF differences
                for ghh in range(1, len(vints)):
                    v = (vints[ghh] + vints[ghh - 1]) / 2.0
                    P = self.powOutput(v)
                    prV = (WindScenario.wblcdf(vints[ghh], cTurb, ki) - WindScenario.wblcdf(vints[ghh-1], cTurb, ki))
                    totalPow += prV * P

                #add rated segment contribution: PRated * (1-CDF(vRated))
                totalPow += self.scenario.PRated * (1.0 - WindScenario.wblcdf(self.scenario.vRated, cTurb, ki))

                #weight by direction bin width and direction probability
                totalPow *= tint * w

                self.tspe[thets, turb] = totalPow
                self.energyCapture += totalPow

        self.wakeFreeRatio = self.energyCapture / (self.scenario.wakeFreeEnergy * n_turb)

        # Numerical safety: WFR should be in [0, 1].
        # Small overshoots above 1 can occur due to floating-point discretisation.
        if self.wakeFreeRatio < 0.0:
            self.wakeFreeRatio = 0.0
        elif self.wakeFreeRatio > 1.0:
            self.wakeFreeRatio = 1.0

        return self.wakeFreeRatio

    #GETTERS

    def getEnergyOutputs(self) -> np.ndarray | None:
        return self.tspe

    def getTurbineFitnesses(self) -> np.ndarray:
        """
        mirror the Java getTurbineFitness function
        sums tspe over directions per turbine, then divides by wakeFreeEnergy
        :return:
        """
        if self.tspe is None:
            return np.array([], dtype=float)

        #tspe shape: (n_dirs, n_turbs)
        res = np.sum(self.tspe, axis=0)
        return res / self.scenario.wakeFreeEnergy

    def getEnergyOutput(self) -> float:
        return self.energyCapture

    def getWakeFreeRatio(self) -> float:
        return self.wakeFreeRatio

    def getEnergyCost(self) -> float:
        return self.energyCost

    def getTurbineRadius(self) -> float:
        return self.scenario.R

    def getFarmWidth(self) -> float:
        return self.scenario.width

    def getFarmHeight(self) -> float:
        return self.scenario.height

    def getObstacles(self) -> np.ndarray:
        return self.scenario.obstacles


    #CONSTRAINT CHECK
    def checkConstraint(self, layout:np.ndarray) -> bool:
        width = self.getFarmWidth()
        height = self.getFarmHeight()
        obstacles = self.scenario.obstacles
        minDist = self.scenario.minDist

        n = layout.shape[0]

        for i in range(n):
            x = float(layout[i,0])
            y = float(layout[i,1])

            #NaN check
            if math.isnan(x) or math.isnan(y) or x <0.0 or y <0.0 or x > width or y > height:
                print(f"turbine {i}{x}, {y} is invalid")
                return False

            #obstacles constraints
            for j in range(obstacles.shape[0]):
                xmin, ymin, xmax, ymax = obstacles[j]
                if x > xmin and x < xmax and y > ymin and y < ymax:
                    print(f"Turbine {i}({x}, {y}) is in obstacle {j} [{xmin},{ymin},{xmax},{ymax}]")
                    return False

            #security distance constraints (squared)
            for j in range(i + 1, n):
                xj = float(layout[j, 0])
                yj = float(layout[j, 1])
                dx = x - xj
                dy = y - yj
                dist = dx * dx + dy * dy
                if dist < minDist:
                    print(
                        f"Security distance violated between turbines {i} and {j}: "
                        f"{math.sqrt(dist)} > {math.sqrt(minDist)}"
                    )
                    return False
        return True

    #WAKE MODEL METHODS
    def calculateWakeTurbine(self, turb:int, thetIndex: int) -> float:
        assert self.tpositions is not None, "tpositions is not set, call evaluate_2014 first"

        x = float(self.tpositions[turb, 0])
        y = float(self.tpositions[turb, 1])

        velDef = 0.0
        n = self.tpositions.shape[0]

        for oturb in range(n):
            if oturb == turb:
                continue

            xo = float(self.tpositions[oturb, 0])
            yo = float(self.tpositions[oturb, 1])

            beta = self.calculateBeta(x,y,xo,yo,thetIndex)
            if beta < self.scenario.atan_k:
                dij = self.calculateProjectedDistance(x,y,xo,yo,thetIndex)
                curDef = self.calculateVelocityDeficit(dij)
                velDef += curDef * curDef
        return math.sqrt(velDef)

    def calculateBeta(self,xi: float,yi:float,xj:float,yj:float,thetIndex:int) -> float:
        cos_t = self.scenario.getCosMidThetas(thetIndex)
        sin_t = self.scenario.getSinMidThetas(thetIndex)

        num = ((xi - xj) * cos_t + (yi - yj) * sin_t + self.scenario.rkRatio)
        a = (xi - xj) + self.scenario.rkRatio * cos_t
        b = (yi - yj) + self.scenario.rkRatio * sin_t
        denom = math.sqrt(a * a + b * b)

        #guard against floating rounding pushing outside [-1,1]
        if denom == 0.0:
            return math.pi #this means its effectively not "upwind"; its used as a safe fallback

        val = num / denom
        val = max(-1.0, min(1.0, val))
        return math.acos(val)

    def powOutput(self, v:float) -> float:
        if v < self.scenario.vCin:
            return 0.0
        elif self.scenario.vCin <= v <= self.scenario.vRated:
            return self.scenario.lambda_ * v + self.scenario.eta
        elif self.scenario.vRated < v < self.scenario.vCout:
            return self.scenario.PRated
        else:
            return 0.0

    def calculateProjectedDistance(self, xi:float,yi:float, xj:float, yj:float, thetIndex: int) -> float:
        cos_t = self.scenario.getCosMidThetas(thetIndex)
        sin_t = self.scenario.getSinMidThetas(thetIndex)
        return abs((xi - xj) * cos_t + (yi - yj) * sin_t)

    def calculateVelocityDeficit(self, dij:float) -> float:
        return self.scenario.trans_CT / ((1.0 + self.scenario.krRatio * dij) ** 2)