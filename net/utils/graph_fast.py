import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='fastskel',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        # self.get_edge(layout)
        # self.hop_dis = get_hop_distance(
        #     self.num_node, self.edge, max_hop=max_hop)
        # self.get_adjacency(strategy)


        self.get_edge(layout)
        self.hop_dis1 = get_hop_distance(
            self.num_node1, self.edge1, max_hop=max_hop)

        self.hop_dis2 = get_hop_distance(
            self.num_node2, self.edge2, max_hop=max_hop)

        self.hop_dis3 = get_hop_distance(
            self.num_node3, self.edge3, max_hop=max_hop)



        A1=self.get_adjacency(strategy, self.num_node1, self.hop_dis1, self.center1)
        A2=self.get_adjacency(strategy, self.num_node2, self.hop_dis2, self.center2)
        A3=self.get_adjacency(strategy, self.num_node3, self.hop_dis3, self.center3)


        self.A1=A1
        self.A2=A2
        self.A3=A3


    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1

        elif layout == 'fastskel':

            #fast transforms
            self.num_node1 = 15
            self_link = [(i, i) for i in range(self.num_node1)]
            neighbor_1base_g1 = [(1, 2), (2, 3), (3, 4), (4,5), (3,6), (6,7), (7,8), (8,9),(9,10),(8,11),(1,12),(12,13),(13,14),(14,15)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base_g1]
            self.edge1 = self_link + neighbor_link
            self.center1 = 1



            self.num_node2 = 6
            self_link = [(i, i) for i in range(self.num_node2)]
            neighbor_1base_g2 = [(1, 2), (2, 3), (3, 4),(4,5),(3,6)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base_g2]
            self.edge2 = self_link + neighbor_link
            self.center2 = 1

            self.num_node3 = 4
            self_link = [(i, i) for i in range(self.num_node3)]
            neighbor_1base_g3 = [(1, 2), (2, 3), (3, 4)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base_g3]
            self.edge3 = self_link + neighbor_link
            self.center3 = 1

        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy, num_node, hop_dis, center):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((num_node, num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, num_node, num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), num_node, num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis ==
                                                                hop]
            return A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_close = np.zeros((num_node, num_node))
                a_further = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j, i] == hop:
                            if hop_dis[j, center] == hop_dis[
                                    i, center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif hop_dis[j, center] > hop_dis[i,center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            return A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
