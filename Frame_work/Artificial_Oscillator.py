import time
import queue
from Token import Token, Route
from modular import Modular as mod
class clock():
    def __init__(self,task_frequence,effector_id):
        self.task_freq = task_frequence
        self.effector_id = effector_id
        self.reply = False
    def alarm(self):
        if self.reply:
            time.sleep(1/self.task_freq)
            self.reply = False
            return self.effector_id
        else:
            self.reply = True
            return self.effector_id
    
class router():
    def __init__(self):
        self.route_dict = {}
    def get_route(self,effector_id):
        assert self.route_dict[effector_id] == None, 'illigel effector_id, try to insert and init the route'
        return self.route_dict[effector_id].copy()
    def add_route(self,effector_id,route):
        # print(effector_id, self.route_dict.keys())
        assert effector_id not in self.route_dict, 'effector has been recorded'
        self.route_dict[effector_id] = route

class artificial_oscillator():
    def __init__(self, effector_modular:mod, task_freq):
        self.clock = clock(task_freq, effector_id=effector_modular.id)
        self.router = router()
        self.FSG = None
        self.effector = effector_modular
    
    def create_token(self, time_mark):
        effector_id = clock.alarm()
        route = self.router.get_route(effector_id)
        source = effector_id
        return Token(effector_id, time_mark, route, source, None)
    
    def run(self, current_time_mark,tk:Token):
        if tk == None:
            new_token = self.create_token(self, current_time_mark)
            return self.effector.receive_Token(new_token,current_time_mark)
        else:
            print(f'get {tk.effector_id}: {tk.message.time_mark} feedback')
            return None 