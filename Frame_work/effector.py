import numpy as np
from Frame_work.modular import Modular as mod
from Frame_work.modular import Modularized_Multiscale_Liquid_State_Machine as mmlsm
from Frame_work.Token import Token


class Effector(mod):
    def __init__(self, id):
        super(Effector, self).__init__(id)
        self.id = id
        return 
    
    def connect_to_state_machine(self, mmlsm: mmlsm):
        self.mmlsm = mmlsm
        
    def set_compute_queue_map(self, node_id):
        super(Effector, self).set_compute_queue_map(node_id)
        return
        
    def set_next_avail_mod_list(self, mod_list):
        super(Effector, self).set_next_avail_mod_list(mod_list)
        return
    
    def set_prev_avail_mod_list(self, mod_list):
        super(Effector, self).set_prev_avail_mod_list(mod_list)
        return 
    
    def receive_reply(self,tk: Token):
        super(Effector, self).receive_reply(tk)