from pyvis.network import Network
import networkx as nx
from flask import Flask, render_template
import random

app = Flask(__name__)

# Create a network
net = Network('500px', '500px')

@app.route('/kk', methods = ['POST'])
def index():
    # Add nodes and edges to the network
    if random.random() >= 0.5:
        net.nodes[0]['color'] = '#FF0000'
    else:
        net.nodes[0]['color'] = '#00FF00'
    net.save_graph('./templates/network.html')
    # Render the network in an HTML template
    return render_template('network.html', net=net)


# @app.route('/kks/ss')
# def index():
#     # Add nodes and edges to the network
#     if random.random() >= 0.5:
#         net.nodes[0]['color'] = '#FF0000'
#     else:
#         net.nodes[0]['color'] = '#00FF00'
#     net.save_graph('./templates/network.html')
#     # Render the network in an HTML template
#     return render_template('network.html', net=net)

if __name__ == '__main__':
    net.add_node(1,label='Node 1',x=100,y=100)
    net.add_node(2,label='Node 2',x=200,y=200)
    net.add_edge(1, 2,label='edge 1-2')
    net.save_graph('./templates/network.html')
    
    app.run(debug=True)
    