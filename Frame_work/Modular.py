from Frame_work.Token import Token, Route,Data_Package
from queue import PriorityQueue, Queue
import random
import torch.nn as nn
import torch.optim as optim
import datetime
from itertools import combinations, permutations
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import torch

time_frame_plot = []


class Modular:
    def __init__(self, id):
        self.id = id
        self.require_list = {}
        self.output_list = {}
        self.current_time_mark = 0
        self.prev_avail_mod_map = {}
        self.next_avail_mod_map = {}
        self.task_queue_map = {}
        self.compute_queue_map = {}
    
    def compute(self, dp_list):
        result_list = {}

        for key, value in dp_list.items():
            result_list[key] = self._model(value.message.content)
        
        datapackage = Data_Package(self.current_time_mark,result_list,self.id)
        time_frame_plot.append((self.current_time_mark, self.id))
        print(f'{self.id} complete computing')
        return datapackage
    
    def set_model(self, model):
        self._model = model
    
    def get_model(self):
        return self._model
    
    def has_model(self):
        if self._model:
            return True
        else:
            return False
        
    # def get_percepted_data(self):
    #     return Data_Package(self.current_time_mark, 0, self.id)
    
    def set_compute_queue_map(self, node_id):
        self.compute_queue_map[node_id] = {}
        
    def backwards(self):
        return 
    
    def set_next_avail_mod_list(self, mod_list):
        self.next_avail_mod_map = {module.id: module for module in mod_list}
        return
    
    def set_prev_avail_mod_list(self, mod_list):
        self.prev_avail_mod_map = {module.id: module for module in mod_list}
        return
    
    def add_token_to_queue_map(self, tk_queue_map, token: Token):
        if token.effector_id in tk_queue_map:
            tk_queue_map[token.effector_id].put(token)
        else:
            tk_queue_map[token.effector_id] = Queue()
            tk_queue_map[token.effector_id].put(token)
        return 
    
    def set_token_to_pre_node_queue(self, node_id, tk: Token):
        if node_id in self.prev_avail_mod_map:
            self.add_token_to_queue_map(self.prev_avail_mod_map[node_id].compute_queue_map[self.id], tk)
    
    #如果当前tk为请求，则执行如下操作
    def receive_request(self,tk:Token):
        #检查当前是否存在可以直接返回的结果，如果有，检查是否超时，如果未超时，则返回，否则删除该键值
        if tk.effector_id in self.output_list:
            datapackage = self.output_list[tk.effector_id]
            if datapackage.check_fresh(tk.message.time_mark) and len(self.prev_avail_mod_map) != 0:
                source_id = tk.message.source
                request_time_mark = tk.request_time_mark
                tk = Token(tk.effector_id,self.current_time_mark,tk.route,self.id,datapackage.content)
                tk.set_request_time_mark(request_time_mark)
                # add result token & notify
                self.set_token_to_pre_node_queue(source_id, tk)
                self.prev_avail_mod_map[source_id].receive_Token(tk)
                return
            elif datapackage.check_fresh(tk.message.time_mark) and len(self.prev_avail_mod_map) == 0:
                print(f"{tk.effector_id} success!!!")
                return
            else:
                print("delete!!!")
                del self.output_list[tk.effector_id]
                for node_id in tk.route.map[self.id]:
                    new_tk = Token(tk.effector_id,self.current_time_mark,tk.route,self.id,None)
                    new_tk.message.time_mark = self.current_time_mark
                    new_tk.message.source = self.id
                    # add task token & notify
                    self.add_token_to_queue_map(self.next_avail_mod_map[node_id].task_queue_map, new_tk)
                    self.next_avail_mod_map[node_id].receive_Token(new_tk)
                return
        #否则执行如下操作                              
        #首先从tk路线图获取当前请求的子节点信息，包含子节点id以及子节点子路由图
        #创建等待数据包，
        #创建请求令牌
        elif self.id.startswith("percept"):
            datapackage = self.get_percepted_data()
            source_id = tk.message.source
            request_time_mark = tk.request_time_mark
            tk = Token(tk.effector_id,self.current_time_mark,tk.route,self.id,datapackage.content)
            tk.set_request_time_mark(request_time_mark)
            # add result token & notify
            self.set_token_to_pre_node_queue(source_id, tk)
            self.prev_avail_mod_map[source_id].receive_Token(tk)
            return
        else:
            for node_id in tk.route.map[self.id]:
                new_tk = Token(tk.effector_id,self.current_time_mark,tk.route,self.id,None)
                new_tk.message.time_mark = self.current_time_mark
                new_tk.message.source = self.id
                # add task token & notify
                self.add_token_to_queue_map(self.next_avail_mod_map[node_id].task_queue_map, new_tk)
                self.next_avail_mod_map[node_id].receive_Token(new_tk)
            return
        
    def receive_reply(self,tk:Token, max_qsize=5):  
        #如果当前令牌为回复，则执行如下操作
        #根据令牌effector_id，找到对应等待数据包索引
        #将根据令牌与等待数据包的source，将令牌中的内容放入等待数据包
        #判断当前等待数据包是否已经完整可以计算
        
        get_all_requirements = True
        require_list = {}
        for node_id in tk.route.map[self.id]:
            get_one = False
            if tk.effector_id in self.compute_queue_map[node_id]:
                index = 0
                max_int = min(self.compute_queue_map[node_id][tk.effector_id].qsize(), max_qsize)
                while index < max_int:
                    node_tk = self.compute_queue_map[node_id][tk.effector_id].get()
                    if node_tk.request_time_mark == tk.request_time_mark:
                        get_one = True
                        require_list[node_id] = node_tk
                        break
                    else:
                        # put back
                        self.compute_queue_map[node_id][tk.effector_id].put(node_tk)
                    index += 1
            get_all_requirements = (get_all_requirements and get_one)
        
        if get_all_requirements:
            datapackage = self.compute(require_list)
            self.output_list[tk.effector_id] = datapackage
            # request_time_mark = tk.request_time_mark
            if self.id.startswith("effect") == 0:
                print(f"{tk.effector_id} success!!!")
                
                return
            
            index = 0
            get_one = False
            max_int = min(self.task_queue_map[tk.effector_id].qsize(), max_qsize)
            while index < max_int:
                task_tk = self.task_queue_map[tk.effector_id].get()
                if task_tk.request_time_mark == tk.request_time_mark:
                    get_one = True
                    # trigger second request success
                    new_tk = task_tk.copy()
                    new_tk.message.time_mark = self.current_time_mark
                    self.receive_Token(new_tk)
                    break
                else:
                    # not right put back
                    self.task_queue_map[tk.effector_id].put(task_tk)
                index+=1
        else:
            for id, item in require_list.items():
                self.compute_queue_map[id][tk.effector_id].put(item)
        return

    def receive_Token(self,tk:Token):
        self.current_time_mark = datetime.datetime.now().timestamp()
        if tk.message.content == None:
            token_queue = self.receive_request(tk)
        else:
            token_queue = self.receive_reply(tk)

        return token_queue
        
class Modularized_Multiscale_Liquid_State_Machine():
    def __init__(self, effectors, reserviors, pertrons):
        self.modulars = effectors + reserviors + pertrons
        self.mod_dict = {mod.id: mod for mod in self.modulars}
        #在这里停顿，将effector与pertron分离，只在mmlsm内部循环找路
        self.pertrons_id = [pertron.id for pertron in pertrons]
        self.reservior_id = [reservior.id for reservior in reserviors]
        self.effector_id = [effector.id for effector in effectors]
        self.effectors = effectors
        self.reserviors = reserviors
        self.pertrons = pertrons
        self.criterion = nn.CrossEntropyLoss()
        self.optimizers = dict()

    def learn_models(self, effector_id, pred_y, y):
        self.optimizers[effector_id].zero_grad()
        loss = self.criterion(pred_y, y)
        try:
            loss.backward()
            learned_count+=1
            print('learn task')
        except RuntimeError:
            print('skip this training step')
            skip_count+=1
        self.optimizers[effector_id].step()
        return
    
    def set_optimizer(self, effector_id, route):
        all_nodes = set()
        for node_id, id_list in route.map.items():
            all_nodes.add(node_id)
            for id in id_list:
                all_nodes.add(id)
        
        list_weights = []
        for effector in self.effectors:
            if effector.id in all_nodes:
                if effector.has_model():
                    list_weights += effector.get_model().parameters()
        for reservior in self.reserviors:
            if reservior.id in all_nodes:
                if reservior.has_model():
                    list_weights += reservior.get_model().parameters()
        for pertron in self.pertrons:
            if pertron.id in all_nodes:
                if pertron.has_model():
                    list_weights += pertron.get_model().parameters()
        self.optimizers[effector_id] = optim.Adam(list_weights, lr=0.001)
        
    def get_modular_id(self,layer_index, branch_index):
        return self.modulars[layer_index*self.layer_num+branch_index].id
    
    def plot_network_structure(self, layer_num=1):
        self.Graph = nx.Graph()
        self.Node_pos = {}
        i = 0
        self.Graph.add_nodes_from(self.effector_id)
        for node_id in self.effector_id:
            self.Node_pos[node_id] = (1, i+1)
            i+=1
        
        i = 0
        self.Graph.add_nodes_from(self.reservior_id)
        for node_id in self.reservior_id:
            self.Node_pos[node_id] = (int(i/layer_num)+2, i % layer_num + 1)
            i+=1
        
        i = 0
        self.Graph.add_nodes_from(self.pertrons_id)
        for node_id in self.pertrons_id:
            self.Node_pos[node_id] = (len(self.reservior_id) / layer_num + 2, i+1)
            i+=1
            
        edges = []
        for nd1 in self.effector_id:
            for nd2 in self.reservior_id[0:layer_num]:
                edges.append((nd1, nd2))
        
        for layer_inx in range(0, int(len(self.reservior_id) / layer_num)-1):
            for nd1 in self.reservior_id[layer_inx*layer_num:(layer_inx+1)*layer_num]:
                for nd2 in self.reservior_id[(layer_inx+1)*layer_num:(layer_inx+2)*layer_num]:
                    edges.append((nd1, nd2))
        
        layer_inx = int(len(self.reservior_id) / layer_num)-1
        for nd1 in self.reservior_id[layer_inx*layer_num:(layer_inx+1)*layer_num]:
            for nd2 in self.pertrons_id:
                edges.append((nd1, nd2))
        self.Graph.add_edges_from(edges)
        return self.Graph, self.Node_pos
    
    def set_network_structure(self, layer_num=1):
        self._layer_num = layer_num
        self._layer_id_map = {}
        offset = 0
        layer_size = int(len(self.reserviors) / layer_num)
        for i in range(len(self.effectors)):
            self.effectors[i].set_next_avail_mod_list(self.reserviors[offset:offset+layer_size])
            for node_id in self.reservior_id[offset:offset+layer_size]:
                self.effectors[i].set_compute_queue_map(node_id)
            
        for i in range(0, layer_num):
            if i == 0:
                for j in range(len(self.reserviors[offset:offset+layer_size])):
                    self.reserviors[offset:offset+layer_size][j].set_prev_avail_mod_list(self.effectors)
                    next_offset = offset+layer_size
                    self.reserviors[offset:offset+layer_size][j].set_next_avail_mod_list(self.reserviors[next_offset:next_offset+layer_size])
                    for node_id in self.reservior_id[next_offset:next_offset+layer_size]:
                        self.reserviors[offset:offset+layer_size][j].set_compute_queue_map(node_id)
            if i == layer_num - 1:
                for j in range(len(self.reserviors[offset:offset+layer_size])):
                    prev_offset = offset-layer_size
                    self.reserviors[offset:offset+layer_size][j].set_prev_avail_mod_list(self.reserviors[prev_offset:prev_offset+layer_size])
                    self.reserviors[offset:offset+layer_size][j].set_next_avail_mod_list(self.pertrons)
                    for node_id in self.pertrons_id:
                        self.reserviors[offset:offset+layer_size][j].set_compute_queue_map(node_id)
            else:
                for j in range(len(self.reserviors[offset:offset+layer_size])):
                    prev_offset = offset-layer_size
                    self.reserviors[offset:offset+layer_size][j].set_prev_avail_mod_list(self.reserviors[prev_offset:prev_offset+layer_size])
                    next_offset = offset+layer_size
                    self.reserviors[offset:offset+layer_size][j].set_next_avail_mod_list(self.reserviors[next_offset:next_offset+layer_size])
                    for node_id in self.reservior_id[next_offset:next_offset+layer_size]:
                        self.reserviors[offset:offset+layer_size][j].set_compute_queue_map(node_id)
            
            self._layer_id_map[i] = self.reservior_id[offset:offset+layer_size]
            offset += layer_size
        print(self._layer_id_map)
        for i in range(len(self.pertrons)):
            self.pertrons[i].set_prev_avail_mod_list(self.reserviors[offset-layer_size:offset])
    
    def has_modular(self,modular_id):
        for mod in self.modulars:
            if mod.id == modular_id:
                return True
        return False
    
    def has_modulars(self,mod_id_list):
        for mod_id in mod_id_list:
            if not self.has_modular(mod_id):
                return False
        return True
    
    def get_random_candidates_routes(self, root_nood_id, leaf_nood_id_list, sample_count=1):
        if not self.has_modular(root_nood_id):
            raise ValueError("Root node must be in the list of nodes.")
        if not self.has_modulars(leaf_nood_id_list):
            print(leaf_nood_id_list)
            print(self.mod_dict.keys())
            raise ValueError("Leaf node must be in the list of nodes.")
        route = Route(root_nood_id)
        previous_layer = []
        layer_index = 0
        while layer_index < self._layer_num:
            current_layer = set()
            if layer_index == 0:
                sampled_nodes = random.sample(self._layer_id_map[layer_index], sample_count)
                for node_id in sampled_nodes:
                    route.add_link(root_nood_id, node_id)
                    current_layer.add(node_id)
            else:
                for pre_node_id in previous_layer:
                    sampled_nodes = random.sample(self._layer_id_map[layer_index], sample_count)
                    for node_id in sampled_nodes:
                        route.add_link(pre_node_id, node_id)
                        current_layer.add(node_id)
            layer_index+=1
            previous_layer=list(current_layer)
        
        for end_node_id in leaf_nood_id_list:
            for node_id in previous_layer:
                route.add_link(node_id, end_node_id)
        return route
            
    def get_candidates_routes(self,root_nood_id, leaf_nood_id_list):
        '''
        根节点为effector，中间节点随机连接，叶节点根据leaf——nood给出
        '''
        if not self.has_modular(root_nood_id):
            raise ValueError("Root node must be in the list of nodes.")
        if not self.has_modulars(leaf_nood_id_list):
            raise ValueError("Leaf node must be in the list of nodes.")
        # Filter only valid nodes for leaf constraints
        valid_leaves = set(leaf_nood_id_list)
        all_subgraphs = []

        # Generate all possible edges
        edges = list(permutations(self.reservior_id, 2))  # All directed edges

        for r in tqdm(range(1,len(edges)+1)):
            for edge_subset in tqdm(combinations(edges, r)):
                route = Route(root_nood_id)  # Initialize the graph with root
                valid_graph = True

                # Add edges to the graph
                for start, end in edge_subset:
                    try:
                        route.add_link(start, end)
                    except ValueError:
                        valid_graph = False
                        break

                # Check for connectivity and leaf constraints
                if valid_graph:
                    reachable_nodes = set()
                    leaves = set()
                    def dfs_iterative(start_node):
                        stack = [start_node]
                        while stack:
                            node = stack.pop()
                            if node not in reachable_nodes:
                                reachable_nodes.add(node)
                                children = route.map.get(node, [])
                                if not children:
                                    leaves.add(node)
                                stack.extend(children)

                    dfs_iterative(route.root_node_id)

                    # Ensure graph is connected and leaves exactly match the valid leaves
                    if len(reachable_nodes) == len(route.map) and leaves == valid_leaves:
                        all_subgraphs.append(route)
                        #Test_code
                        if len(all_subgraphs)>5:
                            print('it is a test solution')
                            import random
                            return random.sample(all_subgraphs,2)
        print('get all candidates')
        return all_subgraphs
    
    def connect_nodes_to_leaf(route, node_list):
        import random
        """
        将 node_list 中的节点随机连接到 route 的叶节点上，
        同时保证图是连通的，并且叶节点仅包含 node_list 中的节点。
        """
        # 获取当前叶节点
        leaf_nodes = route.get_leaf_nodes()
        
        # 如果没有叶节点或者没有待连接的节点，直接返回
        if not leaf_nodes or not node_list:
            return route

        # 随机连接节点，直到所有 node_list 中的节点都连接上
        random.shuffle(node_list)
        
        for node in node_list:
            # 随机选择一个叶节点进行连接
            leaf_node = random.choice(leaf_nodes)
            
            # 连接叶节点和当前节点
            route.add_link(leaf_node, node)

            # 更新叶节点列表，删除已连接的叶节点
            leaf_nodes.remove(leaf_node)
            leaf_nodes.append(node)  # 新添加的节点会成为叶节点

            # 如果图变得不连通，则撤销连接
            if not route.is_connected():
                route.map[leaf_node].remove(node)  # 删除连接
                leaf_nodes.append(leaf_node)  # 恢复叶节点
                leaf_nodes.remove(node)  # 删除新叶节点
                continue
        return route
    
    def receive_token(self, tk:Token, current_time_mark):
        self.mod_dict[tk.route.root_node_id].receive_token(tk,current_time_mark)
