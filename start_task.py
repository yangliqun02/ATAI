import queue
from Frame_work.Framework import graph_base, node_pos,rand_route_init,my_aos,perceptrons,my_mmlsm,run_task
from Frame_work.modular import Modular as mod, Modularized_Multiscale_Liquid_State_Machine as mmlsm
from Frame_work.Artificial_Oscillator import artificial_oscillator as ao
from Frame_work.Token import Token, Route
import random
import threading
import Frame_work.Main_Algorithm as ma
from Frame_work.perceptron import Perceptron as ptron
from Frame_work.modular import time_frame_plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx


# Create the figure and axis
fig, ax = plt.subplots()

# Draw the static graph
nx.draw(graph_base, node_pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)

def to_time_frame_list(resolution=1):
    new_time_frame_plot = [(int(x[0]*resolution), x[1]) for x in time_frame_plot]
    time_frame_dict = {}
    time_set = set()
    frame = -1
    for (time, node_id) in sorted(new_time_frame_plot, key=lambda x:x[0]):
        if time in time_set:
            time_frame_dict[frame].append(node_id)
        else:
            frame += 1
            time_set.add(time)
            time_frame_dict[frame] = [node_id]
    return time_frame_dict

def animate(frame):
    # This function will be called for each frame of the animation
    ax.clear()  # Clear the previous frame
    nx.draw(graph_base, node_pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
    
    # Highlight a node or edge for each frame
    # index = frame % 6
    nodes_to_highlight = frame_dict[frame]
    
    # Update node colors
    node_colors = ['lightblue' if node not in nodes_to_highlight else 'red' for node in graph_base.nodes()]
    # print(node_colors)
    # Update edge colors
    edge_colors = ['gray' for edge in graph_base.edges()]
    
    # Draw the graph with updated colors
    nx.draw(graph_base, node_pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, ax=ax)
    
# 主程序
# start python3.8 start_task.py
if __name__ == "__main__":
    print("start machine")
    perceptrons_id_list = [perceptron.id for perceptron in perceptrons]
    for i in range(len(my_aos)):
        perceptron_id_list1 = random.sample(perceptrons_id_list, 2)
        my_aos[i] = rand_route_init(my_mmlsm,my_aos[i],perceptron_id_list1)
        my_mmlsm.set_optimizer(my_aos[i].effector.id, my_aos[i].router.get_route(my_aos[i].effector.id))
    
    
    executor_thread_1 = threading.Thread(target=run_task, args=(my_aos[0],10))
    executor_thread_2 = threading.Thread(target=run_task, args=(my_aos[1],5))
    executor_thread_1.start()
    executor_thread_2.start()
    executor_thread_1.join()
    executor_thread_2.join()
    # Create the animation
    frame_dict = to_time_frame_list(resolution=1000)
    print(frame_dict)
    ani = animation.FuncAnimation(fig, animate, frames=len(frame_dict), interval=100)
    ani.save('animation.gif', writer='imagemagick')
    # Show the animation
    plt.show()