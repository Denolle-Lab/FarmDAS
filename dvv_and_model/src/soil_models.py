
import numpy as np


###  Basic extraterrestrial radiation model
def rad_a_shortterm(lat, lon, day, lz, time, duration):
    ## Short term radiation model
    ## lat, lon in degrees
    ## day in days from 1st Jan
    ## mid-point time in hours from 00:00
    ## lz in degrees
    ## duration in hours

    lat = np.radians(lat)

    ## Solar declination
    delta = 0.4093 * np.sin(2 * np.pi * day / 365 - 1.39)

    ## Sunset hour angle
    omega_s = np.arccos(-np.tan(lat) * np.tan(delta))

    ## Earth-Sun distance
    d = 1 + 0.033 * np.cos(2 * np.pi * day / 365)

    ## Seasonal correction
    b = 2 * np.pi * (day - 81) / 364
    Sc = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)

    ## Solar time angle at mid-point time
    omega = np.pi / 12 * (time + 0.06667 * (lon - lz) + Sc - 12)

    ## Solar time angle at beginning and end of the period
    omega1 = omega - duration * np.pi / 24
    omega2 = omega + duration * np.pi / 24

    ## Extra-terrestrial radiation for this period
    Ra = 12 * 60 / np.pi * 0.0820 * d * (
            (omega2 - omega1) * np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * (np.sin(omega2) - np.sin(omega1)))
    
    return Ra


### reference ET model
def eto_shortterm(rad, T, wind, RH):
    ## Short term ETo model
    ## rad in MJ/m2/min
    ## T in C
    ## wind in m/s
    ## RH in %

    ## Saturation vapour pressure
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))

    ## Actual vapour pressure
    ea = es * RH / 100

    ## Slope of the saturation vapour pressure curve
    delta = 4098 * es / np.power(T + 237.3, 2)

    ## Psychrometric constant
    gamma = 0.665e-3 * 101.325

    ## ETo
    ETo = (0.408 * delta * rad + gamma * 900/24/60 / (T + 273) * wind * (es - ea)) / (delta + gamma * (1 + 0.34 * wind))

    return ETo


## This model use the actual weather data to account for the evaporation
## The moisture at depth is forward modelled to account for the drainage and evaporation
## The PSD data is calibrated to model the water increase due to the rain
class hydro0:
    def __init__(self, prec, porosity, s0, T, I, etc, damage_index, A_p, A_e, A_d, length=2408, depth=200):
        self.depth = depth  ## mm
        self.prec = prec
        self.s = s0
        self.phi = porosity  ## Porosity
        self.T = T
        self.I = I
        self.etc = etc
        self.damage = damage_index
        self.A_p = A_p
        self.A_e = A_e
        self.A_d = A_d
        self.length = length

        self.a = 6.75e-7 * np.power(self.I, 3) - 7.71e-5 * np.power(self.I, 2) + 1.792e-2 * self.I + 0.49239


    def get_s_history(self):

        s_history = np.zeros(self.length)

        for i in range(self.length):
            ## evaporation
            # self.pet = 1.6 * np.power((10*self.T[i] / self.I), self.a)
            self.evap = self.s * self.etc[i] * self.damage ## Evaporation rate

            ## drainage
            self.drain = self.prec[i] * self.s * self.damage  ## Drainage rate

            s_rate = self.A_p * self.prec[i] - self.A_e * self.evap - self.A_d * self.drain
            
            self.s += s_rate / (self.depth * self.phi)
            self.s = min(self.s, 1) 
            s_history[i] = self.s

        return s_history
    

class litho:
    def __init__(self, bulk_density, porosity, N, f, s, s_wr, tau, igore_capillary=False):
        self.rho_d = bulk_density
        self.phi = porosity  ## derived from rho_d, actually
        self.N = N  ## number of contacts
        self.f = f  ## non-slip fraction
        self.s = s  ## Saturation (water / (air + water))
        self.s_wr = s_wr  ## residual saturation
        self.tau = tau  ## dynamic coefficient of capillary pressure

        ## Solid grains
        self.rho_s = 2650 # kg/m^3
        self.K_s = 37e9 # Pa
        self.G_s = 44e9 # Pa

        ## Water
        self.rho_w = 1000
        self.K_w = 2.2e9  ## by Copilot

        ## Air
        self.rho_a = 1.2  ## kg/m^3
        self.K_a = 1e5  ## by Copilot

        ## Water-air mixture
        self.K_f = 1 / (self.s / self.K_w  +  (1-self.s) / self.K_a)

        ## Effective density of the soil
        self.rho = (1-self.phi)*self.rho_s + self.phi*(self.s*self.rho_w + (1-self.s)*self.rho_a)

        ## Capillary and pore pressure
        if igore_capillary:  ## typical for fully saturated soil
            P_e = (self.rho - self.rho_w) * 9.8 * 0.1
        else:
            kai = (self.s - self.s_wr) / (1 - self.s_wr)
            s_rate = np.diff(self.s) / 60  ## Saturation rate, per second
            P_cdiff = np.concatenate((s_rate, [0])) * self.tau
            P_e = (self.rho - self.rho_a) * 9.8 * 0.1 - (self.rho_w - self.rho_a) * 9.8 * 0.1 * kai - P_cdiff

        ## Moduli of frame (Hertz-Mindlin) 
        nu = (3*self.K_s-2*self.G_s)/(2*(3*self.K_s+self.G_s))  ## Poisson's ratio
        self.K_d = (self.N**2 * (1-self.phi)**2 * self.G_s**2 / (18*np.pi**2 * (1-nu)**2) * P_e)**(1/3)
        self.G_d = 3 * self.K_d * (2+3*self.f-(1+3*self.f)*nu) / (2 - nu) / 5

    def get_vp_vs(self):
        
        self.K = self.K_d + (1-self.K_d/self.K_s)**2 / (self.phi/self.K_f + (1-self.phi)/self.K_s - self.K_d/self.K_s**2)
        self.G = self.G_d
        self.vp = np.sqrt((self.K + 4*self.G/3)/self.rho)
        self.vs = np.sqrt(self.G/self.rho)

        return self.vp, self.vs, self.K, self.G, self.rho