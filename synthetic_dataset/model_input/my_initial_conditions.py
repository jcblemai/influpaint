import gempyor.initial_conditions
import numpy as np

class InitialConditions(gempyor.initial_conditions.InitialConditions):

    def get_from_config(self, sim_id: int, modinf) -> np.ndarray:
        y0 = np.zeros((modinf.compartments.compartments.shape[0], modinf.nsubpops))
        S_idx = modinf.compartments.get_comp_idx({"infection_stage":"S"})
        I_idx = modinf.compartments.get_comp_idx({"infection_stage":"I"})
        R_idx = modinf.compartments.get_comp_idx({"infection_stage":"R"})
        prop_rec = np.random.uniform(low=.3,high=.8, size=modinf.nsubpops)
        n_inf = modinf.subpop_pop*0.005
        y0[S_idx, :] = modinf.subpop_pop * (1- prop_rec) - n_inf
        y0[I_idx, :] = n_inf
        y0[R_idx, :] = modinf.subpop_pop *  prop_rec 

        return y0
    
    def get_from_file(self, sim_id: int, modinf) -> np.ndarray:
        return self.draw(sim_id=sim_id, modinf=modinf)


import gempyor.initial_conditions
import numpy as np

