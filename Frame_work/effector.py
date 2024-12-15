import numpy as np
from modular import Modular as mod
from modular import Modularized_Multiscale_Liquid_State_Machine as mmlsm
from Token import Token


class Effector(mod):
    def __init__(self, id):
        super(Effector, self).__init__(id)
        self.id = id
        return 
    
    def connect_to_state_machine(self, mmlsm: mmlsm):
        self.mmlsm = mmlsm
        
    def set_next_avail_mod_list(self, mod_list):
        self.next_avail_mod_list = mod_list
        return
    
    def set_prev_avail_mod_list(self, mod_list):
        self.prev_avail_mod_list = mod_list
        return 
    
    def receive_reply(self,tk: Token):
        super(Effector, self).receive_reply(tk)