import threading
import time
import queue
from Modular import Modular as mod, Modularized_Multiscale_Liquid_State_Machine as mmlsm
from Artificial_Oscillator import artificial_oscillator as ao
from Token import Token, Route
import Main_Algorithm as ma
from Perceptron import perceptron as ptron
import datetime


# 框架内定义全局所需的变量以及各种全局函数
# 包括路径初始化，路径优化函数
token_queue = queue.Queue()
modulars = [mod(i) for i in range(9)]
my_mmlsm = mmlsm(modulars,modulars[0:3],modulars[3:6],modulars[6:])
my_aos = [ao(effector,5) for effector in modulars[0:3]]
my_ptrons = modulars[6:]

def route_init(my_ao,my_mmlsm:mmlsm,ptrons,poster_input,poster_output):
    #对于每一个任务，从modulars中寻找基于ma距离最近的modular，将其接入路线图
    #直到没有模块可以接入，或者模块的前向输入最大程度接近
    #一阶段实现单独网络替换，二阶段实现分段网络替换
    #当前为一阶段代码，将待整合网络视为一个整体，
    effector_id = my_ao.effector.id
    route = None
    min_dis = float('inf')
    for candi_route in my_mmlsm.get_candidates_routes(effector_id,[pts.id for pts in my_ptrons]):
        dis = ma.pc_distance()
        if dis<min_dis:
            route = candi_route
            min_dis = dis
    my_ao.router.add_route(effector_id,route)

for my_ao in my_aos:
    route_init(my_ao,my_mmlsm,my_ptrons,None,None)
    route_init(my_ao,my_mmlsm,my_ptrons,None,None)

# Framework 线程
def executor_thread(task_queue):
    while True:
        token = task_queue.get()
        if token is None:
            print('empty token queue')
            continue
        time_stamp = datetime.datetime.now().timestamp()
        my_mmlsm.receive_token(token,datetime.datetime.now().timestamp())
        print(f"[Framework] Received token: {token} with timestamp: {datetime.datetime.now().timestamp()}")
        

# Clock线程模拟定时器
def clock_thread(ao):
    time_stamp = datetime.datetime.now().timestamp()
    print(f'clock alarm {time_stamp}')
    tokens = ao.run(time_stamp,None)
    while not tokens.empty():
        token_queue.put(tokens)

def supervisor_thread(token_queue):
    time.sleep(5)
    print(token_queue)
# 主程序
if __name__ == "__main__":

    # 启动 Framework 线程
    for ao in my_aos:
        ao_thread = threading.Thread(target=clock_thread, args=(ao))
        ao_thread.start()
    print('ddddd')
    # 启动线程
    executor_thread_1 = threading.Thread(target=executor_thread, args=(token_queue))
    executor_thread_2 = threading.Thread(target=executor_thread, args=(token_queue))
    executor_thread_1.start()
    executor_thread_2.start()
    print('ddddd')

    supervisor_thread_1 = threading.Thread(target=supervisor_thread, args=(token_queue))
    supervisor_thread_1.start()
    print('ddddd')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        for ao in my_aos:
            ao_thread = threading.Thread(target=clock_thread, args=(ao))
            ao_thread.join()
        executor_thread_1.join()
        executor_thread_2.join()
        supervisor_thread_1.join()

        print("All threads stopped.")