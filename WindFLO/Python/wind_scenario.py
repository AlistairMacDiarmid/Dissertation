#Port of Java/WindScenario.class

from __future__ import annotations
import math
import numpy as np
import xml.etree.ElementTree as ET

class WindScenario:
    FAC = math.pi / 180.0

    def __init__(self,xml_path: str):
        root = ET.parse(xml_path).getroot()

        #match Java's 24 bins of 15 degrees
        self.thetas = np.array([[i * 15.0, (i + 1) * 15.0] for i in range(24)], dtype=float)
        self.c = np.zeros(24, dtype=float)
        self.ks = np.zeros(24, dtype=float)
        self.omegas = np.zeros(24, dtype=float)

        #obstacles in the same format as the Java version: rows of [xmin, ymin, xmac, ymax]
        obstacles = []

        #parse the XML
        for child in root:
            if child.tag.lower() == "angles":
                for angle in child:
                    if angle.tag.lower() != "angle":
                        continue
                    #map attributes, c, k, omega, theta into arrays by theta/15 index
                    attrs = dict(angle.items())
                    theta = float(attrs["theta"])
                    idex = int(theta // 15)
                    self.c[idex] = float(attrs["c"])
                    self.ks[idex] = float(attrs["k"])
                    self.omegas[idex] = float(attrs["omega"])

            elif child.tag.lower() == "obstacles":
                for obs in child:
                    if obs.tag.lower() != "obstacle":
                        continue
                    attrs = dict(obs.items())
                    obstacles.append([
                        float(attrs["xmin"]),
                        float(attrs["ymin"]),
                        float(attrs["xmax"]),
                        float(attrs["ymax"]),
                    ])

            elif child.tag.lower() == "parameters":
                for param in child:
                    tag = param.tag
                    val = float(param.text)
                    if tag == "Width":
                        self.width = val
                    elif tag == "Height":
                        self.height = val
                    elif tag == "NTurbines":
                        self.nturbines = int(val)
                    elif tag == "WakeFreeEnergy":
                        self.wakeFreeEnergy = val

        self.obstacles = np.array(obstacles, dtype=float) if obstacles else np.zeros((0,4), dtype=float)

        #Same constraints Java version - they are the same across all scenarios, so are NOT included in the XML files
        self.CT = 0.8
        self.PRated = 1500.0
        self.R = 38.5
        self.eta = -500.0
        self.k = 0.0750
        self.lambda_ = 140.86
        self.vCin = 3.5
        self.vCout = 20.0
        self.vRated = 14.

        self.init_optimisation_parameters()

    def init_optimisation_parameters(self):
        #the cos/sin of mind-theta
        mids = np.mean(self.thetas, axis=1) * self.FAC
        self.cos_mid = np.cos(mids)
        self.sin_mid = np.sin(mids)

        self.rkRatio = self.R / self.k
        self.krRatio = self.k / self.R

        #vints : 3.5 -> 14.0, with 0.5 step
        n_vints = int(2.0 * self.vRated - 7.0 + 1.0)
        self.vints = np.array([3.5 + i * 0.5 for i in range(n_vints)], dtype=float)

        self.atan_k = math.atan(self.k)
        self.trans_CT = 1.0 - math.sqrt(1.0 - self.CT)
        self.minDist = 64.0 * self.R * self.R #(8R)^2

    @staticmethod
    def wblcdf(x: float, sc: float, sh: float) -> float:
        if x <= 0.0:
            return 0.0
        return 1.0 - math.exp(-((x / sc) ** sh))

    def getCosMidThetas(self, idx: int) -> float:
        return float(self.cos_mid[idx])

    def getSinMidThetas(self, idx: int) -> float:
        return float(self.sin_mid[idx])
