import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import random

# Create a new directed graph
G = nx.Graph()

# Add nodes (neurons)
layer1 = ["A", "B", "C"]
layer2 = ["D", "E", "F"]
layer3 = ["G", "H", "I"]
G.add_nodes_from(layer1)
G.add_nodes_from(layer2)
G.add_nodes_from(layer3)
pos = {}
i = 0
for node in layer1:
    pos[node] = (int(i/3)+1, i % 3 + 1)
    i+=1

for node in layer2:
    pos[node] = (int(i/3)+1, i % 3 + 1)
    i+=1
    
for node in layer3:
    pos[node] = (int(i/3)+1, i % 3 + 1)
    i+=1

# Add edges (connections between neurons)
edges = []

for node1  in layer1:
    for node2 in layer2:
        edges.append((node1, node2))

for node1  in layer2:
    for node2 in layer3:
        edges.append((node1, node2))
G.add_edges_from(edges)

# Position the nodes

# Create the figure and axis
fig, ax = plt.subplots()

# Draw the static graph
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)

total_list = []
# Animation function
def animate(frame):
    # This function will be called for each frame of the animation
    ax.clear()  # Clear the previous frame
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
    
    # Highlight a node or edge for each frame
    index = frame % 6
    
    # print(index)
    if index == 0:
        nodes_to_highlight = random.choices(layer1,k=2)
        total_list.append(nodes_to_highlight)
    if index == 1:
        nodes_to_highlight = random.choices(layer2,k=2)
        total_list.append(nodes_to_highlight)
    if index == 2:
        nodes_to_highlight = random.choices(layer3,k=2)
        total_list.append(nodes_to_highlight)
    if index == 3:
        nodes_to_highlight = total_list.pop()
    if index == 4:
        nodes_to_highlight = total_list.pop()
    if index == 5:
        nodes_to_highlight = total_list.pop()
    
    edge_to_highlight = edges[frame % len(edges)]
    
    # Update node colors
    node_colors = ['lightblue' if node not in nodes_to_highlight else 'red' for node in G.nodes()]
    # print(node_colors)
    # Update edge colors
    edge_colors = ['gray' if edge != edge_to_highlight else 'red' for edge in G.edges()]
    
    # Draw the graph with updated colors
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, ax=ax)
    
    # Set title to show current highlighted node/edge
    # ax.set_title(f"Node: {node_to_highlight}, Edge: {edge_to_highlight}")

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=32, interval=500)
ani.save('animation.gif', writer='imagemagick')
# Show the animation
plt.show()