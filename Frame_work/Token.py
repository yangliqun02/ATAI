class Route:
    def __init__(self, root_node_id):
        self.id = id
        self.map = {}
        self.map[root_node_id] = []
        self.root_node_id = root_node_id
        self.current_node_id = root_node_id
    def get_child_node(self):
        child_node_list = self.map[self.root_node_id]
        child_subgraph = []
        for child_node in child_node_list:
            child_subgraph.append(self.get_subgraph(child_node))

        return child_node_list,child_subgraph
    def get_subgraph(self, root):
        """
        获取以给定节点为根节点的子图
        """
        subgraph = Route(root)  # 创建一个新的子图对象
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in self.map.get(node, []):
                if neighbor not in visited:
                    subgraph.add_link(node, neighbor)
                    dfs(neighbor)

        # 从给定的根节点开始遍历子图
        dfs(root)

        return subgraph
    
    def copy(self):
        # Create a shallow copy
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        return other

    def has_cycle(self):
        visited = set()
        stack = set()

        def dfs(node, parent):
            visited.add(node)
            stack.add(node)
            for neighbor in self.map.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent and neighbor in stack:
                    return True
            stack.remove(node)
            return False

        for node in self.map.keys():
            if node not in visited:
                if dfs(node, None):
                    return True
        return False

    def add_node(self, start_node_id, end_node_id, new_node_id):
        if start_node_id not in self.map:
            self.map[start_node_id] = []
        if end_node_id not in self.map:
            self.map[end_node_id] = []
        if new_node_id in self.map:
            raise ValueError(f"Node {new_node_id} already exists.")
        
        # Temporarily add the node and links
        self.map[start_node_id].append(new_node_id)
        self.map[new_node_id] = [end_node_id]

        # Check for cycles
        if self.has_cycle():
            # Rollback changes if a cycle is detected
            self.map[start_node_id].remove(new_node_id)
            self.map.pop(new_node_id)
            raise ValueError(f"Adding node {new_node_id} creates a cycle.")

    def add_link(self, start_node_id, end_node_id):
        if start_node_id not in self.map:
            self.map[start_node_id] = []
        if end_node_id not in self.map:
            self.map[end_node_id] = []
        
        # Temporarily add the link
        self.map[start_node_id].append(end_node_id)

        # Check for cycles
        if self.has_cycle():
            # Rollback changes if a cycle is detected
            self.map[start_node_id].remove(end_node_id)
            raise ValueError(f"Adding link from {start_node_id} to {end_node_id} creates a cycle.")

    def remove_link(self, start_node_id, end_node_id):
        assert self.map[start_node_id] is not None, "Start node id is illegal"
        list_endnode_ids = self.map[start_node_id]
        if end_node_id not in list_endnode_ids:
            return
        else:
            list_endnode_ids.remove(end_node_id)

    def is_connected(self):
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in self.map.get(node, []):
                dfs(neighbor)

        dfs(self.root_node_id)
        return len(visited) == len(self.map)

    def remove_node(self, node):
        if node == self.root_node_id:
            raise ValueError("Cannot delete the root node")

        temp_map = {k: v.copy() for k, v in self.map.items()}
        for parent, children in temp_map.items():
            if node in children:
                children.remove(node)

        temp_map.pop(node, None)

        if not self.is_connected():
            raise ValueError(f"Deleting node {node} will disconnect the graph")

        self.map = temp_map

    def __str__(self):
        """
        列举所有父节点及其子节点
        """
        result = "Tree Structure:\n"
        for parent, children in self.map.items():
            result += f"{parent} -> {', '.join(map(str, children)) if children else 'None'}\n"
        result = result+f"current_node: {self.current_node_id}\n"
        result = result+f"root_node: {self.root_node_id}\n"

        return result

class Data_Package():
    def __init__(self, time_mark,content,source):
        self.content = content
        self.time_mark = time_mark
        self.source = source
    def check_fresh(self,time_mark:int,tolerance = 2):
        return pow(self.time_mark-time_mark,2)<tolerance
class Token:
    def __init__(self, effector_id, time_mark,route:Route,source,content):
        self.effector_id = effector_id
        self.request_time_mark = 0
        self.route = route
        self.message = Data_Package(time_mark,content,source)
    def set_message(self,content):
        self.message.content = content
        
    def set_request_time_mark(self, time_mark):
        self.request_time_mark = time_mark
    
    def copy(self):
        # Create a shallow copy
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        return other

def main():
    r = Route(1)
    r.add_node(1, 2, 3)  # 插入节点成功
    r.add_link(2, 4)     # 插入链接成功
    r.add_link(1,7)
    r.add_link(7,2)
    try:
        r.add_link(4, 1)  # 尝试插入导致环的链接
    except ValueError as e:
        print(e)          # 输出: Adding link from 4 to 1 creates a cycle.

    print(r)

if __name__ == "__main__":
    main()
