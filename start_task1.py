import queue
from Framework import clock_thread, executor_thread
from modular import Modular as mod, Modularized_Multiscale_Liquid_State_Machine as mmlsm
from Artificial_Oscillator import artificial_oscillator as ao
from Token import Token, Route
import Main_Algorithm as ma
from perceptron import Perceptron as ptron


# 主程序
if __name__ == "__main__":
    token_queue = queue.Queue()
    modulars = [mod(i) for i in range(9)]
    my_mmlsm = mmlsm(modulars,modulars[0:3],modulars[3:6],modulars[6:])
    my_aos = [ao(effector, 5) for effector in modulars[0:3]]
    my_ptrons = modulars[6:]
    print("start task 1")