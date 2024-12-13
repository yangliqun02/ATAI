from Token import Token,Route
class perceptron():
    def __init__(self, perceptron_id):
        self.id = perceptron_id
        self.current_time_mark= 0
        #内容暂时置为0
        self.content = 0
    
    def reply_token(self, tk:Token, current_time_mark):
        self.current_time = current_time_mark
        route = Route(tk.message.source)
        output_token = Token(tk.effector_id,current_time_mark,route,self.id,self.content)
        return output_token
    