from Token import Token, Route,Data_Package
import queue
import random
from itertools import combinations, permutations
from tqdm import tqdm

class Modular:
    def __init__(self, id):
        self.id = id
        self.require_list = {}
        self.output_list = {}
        self.current_time_mark = 0
        self.prev_avail_mod_list = []
        self.next_avail_mod_list = []
    
    def compute(self,dp_list):
        print(f'{self.id} complete computing')
        datapackage = Data_Package(self.current_time_mark,0,self.id)
        
        return datapackage
    
    def set_next_avail_mod_list(self, mod_list):
        self.next_avail_mod_list = mod_list
        return
    
    def set_prev_avail_mod_list(self, mod_list):
        self.prev_avail_mod_list = mod_list
        return 
    
    #如果当前tk为请求，则执行如下操作
    def receive_request(self,tk:Token):
        token_queue = queue.Queue()
        #检查当前是否存在可以直接返回的结果，如果有，检查是否超时，如果未超时，则返回，否则删除该键值
        if tk.effector_id in self.output_list:
            datapackage = self.output_list[tk.effector_id]
            if datapackage.check_fresh(tk.message.time_mark):
                tk = Token(tk.effector_id,self.current_time_mark,Route(tk.message.source),self.id,datapackage.content)
                token_queue.put(tk)
            else:
                del self.output_list[tk.effector_id]
        #否则执行如下操作
        #首先从tk路线图获取当前请求的子节点信息，包含子节点id以及子节点子路由图
        #创建等待数据包，
        #创建请求令牌
        else:
            child_node_ids,child_node_subgraph = tk.route.get_child_node(self.id)
            for cn_id,cn_subgraph in zip(child_node_ids,child_node_subgraph):
                self.require_list[tk.effector_id].append(Data_Package(self.current_time_mark,None,cn_id))
                tk = Token(tk.effector_id,self.current_time_mark,cn_subgraph,self.id,None)
                token_queue.put(tk)
        return token_queue

    def receive_reply(self,tk:Token):
        #如果当前令牌为回复，则执行如下操作
        #根据令牌effector_id，找到对应等待数据包索引
        #将根据令牌与等待数据包的source，将令牌中的内容放入等待数据包
        #判断当前等待数据包是否已经完整可以计算
        token_queue = queue.Queue()
        get_all_requirements = True
        for datapackage in self.require_list[tk.effector_id]:
            if datapackage.source == tk.message.source:
                datapackage.content = tk.message.content
            elif datapackage.content == None:
                get_all_requirements = False
        
        if get_all_requirements:
            datapackage = self.compute(self.require_list[tk.effector_id])
            tk = Token(tk.effector_id,self.current_time_mark,Route(tk.message.source),self.id,datapackage.content)
            token_queue.put(tk)
        return token_queue

    def receive_Token(self,tk:Token,current_time_mark):
        self.current_time_mark = current_time_mark
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

    def get_modular_id(self,layer_index, branch_index):
        return self.modulars[layer_index*self.layer_num+branch_index].id
    
    def set_network_structure(self, layer_num=3):
        self._layer_num = layer_num
        self._layer_id_map = {}
        offset = 0
        for i in range(len(self.effectors)):
            self.effectors[i].set_next_avail_mod_list(self.reserviors[offset:offset+layer_num])
            
        for i in range(0, layer_num):
            if i == 0:
                for j in range(len(self.reserviors[offset:offset+layer_num])):
                    self.reserviors[offset:offset+layer_num][j].set_prev_avail_mod_list(self.effectors)
                    next_offset = offset+1
                    self.reserviors[offset:offset+layer_num][j].set_next_avail_mod_list(self.reserviors[next_offset:next_offset+layer_num])
            elif i == layer_num - 1:
                for j in range(len(self.reserviors[offset:offset+layer_num])):
                    prev_offset = offset-1
                    self.reserviors[offset:offset+layer_num][j].set_prev_avail_mod_list(self.reserviors[prev_offset:prev_offset+layer_num])
                    self.reserviors[offset:offset+layer_num][j].set_next_avail_mod_list(self.pertrons)
            else:
                for j in range(len(self.reserviors[offset:offset+layer_num])):
                    prev_offset = offset-1
                    self.reserviors[offset:offset+layer_num][j].set_prev_avail_mod_list(self.reserviors[prev_offset:prev_offset+layer_num])
                    next_offset = offset+1
                    self.reserviors[offset:offset+layer_num][j].set_next_avail_mod_list(self.reserviors[next_offset:next_offset+layer_num])
            
            self._layer_id_map[i] = self.reservior_id[offset:offset+layer_num]
            offset += layer_num
        
        for i in range(len(self.pertrons)):
            self.pertrons[i].set_prev_avail_mod_list(self.reserviors[layer_num-1:layer_num-1+layer_num])
    
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
    
    def get_random_candidates_routes(self, root_nood_id, leaf_nood_id_list, sample_count=2):
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
