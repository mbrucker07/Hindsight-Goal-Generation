import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # for plotting, had to sudo apt-get install python3-tk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.art3d as art3d
import argparse
import gym
import pickle

# TODO: edit section
class DistanceMesh:

    cs_graph = None
    dist_matrix = None
    predecessors = None

    def __init__(self, field, spaces, obstacles):
        # field is of form [m_x, m_y, m_z, l, w, h]
        # obstacles is list with entries of form: [m_x, m_y, m_z, l, w, h]
        # Up to 10000 nodes can be handled memorywise (computation time is not problematic)
        # x_spaces * y_spaces * z_spaces should not increase beyond 8000
        [m_x, m_y, m_z, l, w, h] = field

        self.x_min = m_x - l
        self.y_min = m_y - w
        self.z_min = m_z - h

        self.x_max = m_x + l
        self.y_max = m_y + w
        self.z_max = m_y + h

        assert len(spaces) == 3
        self.x_spaces = spaces[0]
        self.y_spaces = spaces[1]
        self.z_spaces = spaces[2]

        self.obstacles = obstacles

        # Up to 10000 nodes can be handled memorywise (computation time is not problematic)
        # Product should not increase 8000

        # TODO: end edit

        self.dx = (self.x_max - self.x_min) / self.x_spaces
        self.dy = (self.y_max - self.y_min) / self.y_spaces
        self.dz = (self.z_max - self.z_min) / self.z_spaces

        self.x_range = self.x_spaces + 1
        self.y_range = self.y_spaces + 1
        self.z_range = self.z_spaces + 1

        self.num_nodes = self.x_range * self.y_range * self.z_range

        self.colors = np.zeros((self.x_range + 1, self.y_range + 1, self.z_range + 1))
        self.numbers = np.zeros((self.x_range + 1, self.y_range + 1, self.z_range + 1))
        #edges = np.zeros((num_nodes, num_nodes))

        self.colors[-1, :, :] = 1
        self.colors[:, -1, :] = 1
        self.colors[:, :, -1] = 1

        self.previous = list()
        for a in [-1, 0, 1]:
            for b in [-1, 0, 1]:
                self.previous.append([a, b, -1])
        for a in [-1, 0, 1]:
            self.previous.append([a, -1, 0])
        self.previous.append([-1, 0, 0])

        # sanity check whether mesh is fine enough (obstacles only detectable if e.g. l>dx/2)
        mesh_okay = True
        x_space_min = 0
        y_space_min = 0
        z_space_min = 0
        for [m_x, m_y, m_z, l, w, h] in self.obstacles:
            x_space_min = max(x_space_min, (self.x_max - self.x_min) / (2*l))
            y_space_min = max(y_space_min, (self.y_max - self.y_min) / (2*w))
            z_space_min = max(z_space_min, (self.z_max - self.z_min) / (2*h))
            if l <= self.dx/2 or w <= self.dy/2 or h <= self.dz/2:
                mesh_okay = False

        # print section
        print("Created DistanceMesh with: ")
        print("\tx: [{}, {}], y: [{}, {}], z: [{}, {}]".format(self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max))
        print("\tDx: {}, Dy: {}, Dz: {}".format(self.dx, self.dy, self.dz))
        print("\tX_spaces: {}, y_spaces: {}, z_spaces: {}".format(self.x_spaces, self.y_spaces, self.z_spaces))
        print("\tRequired spaces: x > {}, y > {}, z > {}".format(x_space_min, y_space_min, z_space_min))
        print("\tNum_nodes: {}".format(self.num_nodes))
        print("\tObstacles:")
        for obstacle in obstacles:
            print("\t\t{}".format(obstacle))

        if not mesh_okay:
            raise Exception("Mesh is not fine enough, requirements see above")

    def black(self, x, y, z):
        for [m_x, m_y, m_z, l, w, h] in self.obstacles:
            if m_x - l <= x <= m_x + l and m_y - w <= y <= m_y + w and m_z - h <= z <= m_z + h:
                return True
            else:
                return False

    def index2node(self, index):
        [i, j, k] = index
        node = i + j * self.x_range + k * self.x_range * self.y_range
        return int(node)


    def node2index(self, node):
        k = np.floor(node / (self.x_range * self.y_range))
        new = node % (self.x_range * self.y_range)
        j = np.floor(new / self.x_range)
        i = node % self.x_range
        return i, j, k


    def index2coords(self, index):
        [i, j, k] = index
        x = self.x_min + i*self.dx
        y = self.y_min + j*self.dy
        z = self.z_min + k*self.dz
        return x, y, z


    def coords2index(self, coords):
        [x, y, z] = coords
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max and self.z_min <= z <= self.z_max):
            return None
        i = np.round((x-self.x_min)/self.dx)
        j = np.round((y-self.y_min)/self.dy)
        k = np.round((z-self.z_min)/self.dz)
        return i, j, k


    def compute_cs_graph(self):
        print("Computing {}x{} cs_graph ...".format(self.num_nodes, self.num_nodes))
        start = timer()
        row = list()
        col = list()
        data = list()

        # mark points with obstacle as black (1 in self.colors)
        for i in range(self.x_range):
            for j in range(self.y_range):
                for k in range(self.z_range):
                    self.numbers[i, j, k] = self.index2node([i, j, k])
                    x, y, z = self.index2coords([i, j, k])
                    if self.black(x, y, z): # if point is black
                        self.colors[i, j, k] = 1
        # connect points which are white (0 in self.colors)
        for i in range(self.x_range):
            for j in range(self.y_range):
                for k in range(self.z_range):
                        for a, b, c in self.previous:
                            if self.colors[i, j, k] == 0:
                                if self.colors[i+a, j+b, k+c] == 0: # i.e. is white
                                    basenode = self.index2node([i, j, k])
                                    connectnode = self.index2node([i+a, j+b, k+c])
                                    d = np.sqrt(a*a*self.dx*self.dx + b*b*self.dy*self.dy + c*c*self.dz*self.dz)
                                    #edges[index2node([i, j, k]), index2node([i+a, j+b, k+c])] = np.sqrt(a*a + b*b + c*c)
                                    #edges[index2node([i+a, j+b, k+c]), index2node([i, j, k])] = np.sqrt(a*a + b*b + c*c)
                                    row.append(basenode)
                                    col.append(connectnode)
                                    data.append(d)

        #cs_graph = csr_matrix(edges)
        self.cs_graph = csr_matrix((data, (row,col)), shape=(self.num_nodes, self.num_nodes))
        end = timer()
        print("\tdone after {} secs".format(end-start))

    def compute_dist_matrix(self, compute_predecessors=False):
        print("Computing {}x{} dist_matrix ...".format(self.num_nodes, self.num_nodes))
        start = timer()
        if self.cs_graph is None:
            raise Exception("No CS_Graph available!")
        if compute_predecessors:
            self.dist_matrix, self.predecessors = dijkstra(self.cs_graph, directed=False, return_predecessors=True)
        else:
            self.dist_matrix = dijkstra(self.cs_graph, directed=False, return_predecessors=False)
        end = timer()
        print("\t done after {} secs".format(end-start))

    def get_dist(self, coords1, coords2, return_path=False):
        # coords1 and coords2 of form [1, 1, 1]
        if self.dist_matrix is None:
            raise Exception("No dist_matrix available!")
        if self.coords2index(coords1) is None or self.coords2index(coords2) is None:
            return np.inf
        goal_a = self.index2coords(self.node2index(self.index2node(self.coords2index(coords1))))
        goal_b = self.index2coords(self.node2index(self.index2node(self.coords2index(coords2))))
        #print("get_dist goals: {} // {}".format(goal_a, goal_b))
        node_a = self.index2node(self.coords2index(coords1))
        node_b = self.index2node(self.coords2index(coords2))
        if not return_path:
            return self.dist_matrix[node_a, node_b]
        else:
            if self.predecessors is None:
                raise Exception("No predecessors available!")
            path = []
            current_node = node_b
            path.append(self.index2coords(self.node2index(current_node)))
            while current_node != node_a:
                current_node = self.predecessors[node_a, current_node]
                path.append(self.index2coords(self.node2index(current_node)))
            return self.dist_matrix[node_a, node_b], path


    def plot_graph(self, path=None, mesh=False, obstacle_nodes=False, goals=None, save_path='test'):
        print("Plotting ...")
        if self.cs_graph is None:
            raise Exception("No cs_graph available")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        co_graph = coo_matrix(self.cs_graph)
        # scatter plot boundaries of field
        x_array = [self.x_min, self.x_min, self.x_min, self.x_min, self.x_max, self.x_max, self.x_max, self.x_max]
        y_array = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min, self.y_min, self.y_max, self.y_max]
        z_array = [self.z_min, self.z_max, self.z_min, self.z_max, self.z_min, self.z_max, self.z_min, self.z_max]
        ax.scatter(x_array, y_array, z_array, c='b')
        # plots obstacle
        for [m_x, m_y, m_z, l, w, h] in self.obstacles:
            # top
            side1 = Rectangle((m_x-l, m_y-w), 2*l, 2*w, color=[0,0,1,0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_z+h, zdir="z",)
            # bottom
            side1 = Rectangle((m_x-l, m_y-w), 2*l, 2*w, color=[0,0,1,0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_z-h, zdir="z")
            # back
            side1 = Rectangle((m_y-w, m_z-h), 2*w, 2*h, color=[0,0,1,0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_x+l, zdir="x")
            # front
            side1 = Rectangle((m_y-w, m_z-h), 2*w, 2*h, color=[0,0,1,0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_x-l, zdir="x")
            # right
            side1 = Rectangle((m_x-l, m_z-h), 2*l, 2*h, color=[0,0,1,0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_y+w, zdir="y")
            # left
            side1 = Rectangle((m_x-l, m_z-h), 2*l, 2*h, color=[0,0,1,0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_y-w, zdir="y")
            # plot graph edges
        if mesh:
            for i, j, v in zip(co_graph.row, co_graph.col, co_graph.data):
                a = self.index2coords(self.node2index(i))
                b = self.index2coords(self.node2index(j))
                X, Y, Z = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]
                ax.plot(X, Y, Z, c=[0, 1, 1, 0.1])
        # scatter plot nodes that are marked as black (with obstacle)
        if obstacle_nodes:
            for i in range(self.x_range):
                for j in range(self.y_range):
                    for k in range(self.z_range):
                        x, y, z = self.index2coords([i, j, k])
                        if self.colors[i, j, k] == 1:
                            ax.scatter([x], [y], [z], c='b')
        # plot goals:
        for goal in goals:
            x = goal[0]
            y = goal[1]
            z = goal[2]
            ax.scatter([x], [y], [z], c='black')

        # plot path
        if path:
            for i in range(len(path)-1):
                a = path[i]
                b = path[i+1]
                X, Y, Z = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]
                ax.plot(X, Y, Z, c=[1, 0, 0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.savefig(save_path + ".png")
        print("\tdone")


    def print_mesh(self):
        for i in range(self.z_range):
            print("z={}:".format(i))
            print(self.colors[:-1, :-1, i])

    def print_numbers(self):
        for i in range(self.z_range):
            print("z={}:".format(i))
            print(self.numbers[:-1, :-1, i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--mesh', type=bool, default=False)
    parser.add_argument('--obstacle_nodes', type=bool, default=False)
    #parser.add_argument('--path', type=bool, default=False)
    parser.add_argumetn('--pickle', type=str, default=None)
    args = parser.parse_args()
    obstacles = list()
    goals_list = None
    if args.pickle:
        with open(args.pickle, 'rb') as file:
            goals_list = pickle.load(file)


    field = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    obstacles = [[0.5, 0.5, 0.25, 0.5, 0.25, 0.25]]
    spaces = [30, 30, 5]
    goals = [[1, 0, 0.2], [0,0.1, 0.1], [0.8, 0, 0]]
    mesh = DistanceMesh(field=field, spaces=spaces, obstacles=obstacles)
    mesh.compute_cs_graph()
    mesh.compute_dist_matrix(compute_predecessors=True)

    dist, path = mesh.get_dist([0, 0, 0], [1, 1, 0], return_path=True)
    #mesh.plot_graph(path=path, mesh=True, obstacle_nodes=True)


    mesh.plot_graph(goals=goals, save_path="../log/test")
    print("Dist: {}".format(dist))

if __name__ == "__main__":
    main()
