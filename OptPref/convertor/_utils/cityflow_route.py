import networkx as nx
from math import sqrt
def heuristic(a, b):
    global coords
    (x1, y1) = coords[a]
    (x2, y2) = coords[b]
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
class AStarPath:
    def __init__(self,roadLinksDict:dict,roadPoints:dict):
        global coords
        coords = roadPoints
        # 所有节点都有坐标
        assert all(rid in roadPoints for road_pair in roadLinksDict.keys() for rid in road_pair )
        # 有向图
        G = nx.DiGraph()
        for (_in_rid,_out_rid), (_linkLanes,_weight) in roadLinksDict.items():
            if _linkLanes:
                G.add_edge(_in_rid, _out_rid, weight=_weight)
        self.graph = G
    
    def astar_search(self, start, goal):
        res = list(nx.astar_path(self.graph, source=start, target=goal,heuristic=heuristic))
        return res