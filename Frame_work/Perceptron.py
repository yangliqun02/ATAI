from Token import Token,Route,Data_Package
from modular import Modular as mod
class Perceptron(mod):
    def __init__(self, perceptron_id):
        super(Perceptron, self).__init__(perceptron_id)
        self.id = perceptron_id
        self.current_time_mark= 0
        #内容暂时置为0
        self.content = 0
    
    def reply_token(self, tk:Token, current_time_mark):
        self.current_time = current_time_mark
        route = Route(tk.message.source)
        output_token = Token(tk.effector_id,current_time_mark,route,self.id,self.content)
        return output_token
    
    def set_compute_queue_map(self, node_id):
        super(Perceptron, self).set_compute_queue_map(node_id)
        return
        
    def set_next_avail_mod_list(self, mod_list):
        super(Perceptron, self).set_next_avail_mod_list(mod_list)
        return
    
    def set_prev_avail_mod_list(self, mod_list):
        super(Perceptron, self).set_prev_avail_mod_list(mod_list)
        return 
    
    def get_percepted_data(self):
        print(f'{self.id} reach end')
        return Data_Package(self.current_time_mark, f"perception {self.id}", self.id)
    