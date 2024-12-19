import queue
from Frame_work.Framework import clock_thread, executor_thread,rand_route_init,my_aos,perceptrons,my_mmlsm,run_task
from Frame_work.modular import Modular as mod, Modularized_Multiscale_Liquid_State_Machine as mmlsm
from Frame_work.Artificial_Oscillator import artificial_oscillator as ao
from Frame_work.Token import Token, Route
import random
import threading
import Frame_work.Main_Algorithm as ma
from Frame_work.perceptron import Perceptron as ptron


# 主程序
# start python3.8 start_task.py
if __name__ == "__main__":
    print("start machine")
    perceptrons_id_list = [perceptron.id for perceptron in perceptrons]
    for i in range(len(my_aos)):
        perceptron_id_list1 = random.sample(perceptrons_id_list, 2)
        my_aos[i] = rand_route_init(my_mmlsm,my_aos[i],perceptron_id_list1)
    
    
    executor_thread_1 = threading.Thread(target=run_task, args=(my_aos[0],10))
    executor_thread_2 = threading.Thread(target=run_task, args=(my_aos[1],5))
    executor_thread_1.start()
    executor_thread_2.start()