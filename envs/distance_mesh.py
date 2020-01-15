import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from timeit import default_timer as timer

# TODO: edit section
class DistanceMesh:

    cs_graph = None
    dist_matrix = None

    def __init__(self, region=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], spaces=[10, 10, 10], obstacles=list()):
        # region is of form [m_x, m_y, m_z, l, w, h]
        # obstacles is list with entries of form: [m_x, m_y, m_z, l, w, h]
        # Up to 10000 nodes can be handled memorywise (computation time is not problematic)
        # x_spaces * y_spaces * z_spaces should not increase beyond 8000
        [m_x, m_y, m_z, l, w, h] = region

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

        for i in range(self.x_range):
            for j in range(self.y_range):
                for k in range(self.z_range):
                    self.numbers[i, j, k] = self.index2node([i, j, k])
                    x, y, z = self.index2coords([i, j, k])
                    if self.black(x, y, z): # if point is black
                        self.colors[i, j, k] = 1
                    else: # if point is white
                        for a, b, c in self.previous:
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

    def compute_dist_matrix(self):
        print("Computing {}x{} dist_matrix ...".format(self.num_nodes, self.num_nodes))
        start = timer()
        if self.cs_graph is None:
            raise Exception("No CS_Graph available!")
        self.dist_matrix = dijkstra(self.cs_graph, directed=False)
        end = timer()
        print("\t done after {} secs".format(end-start))

    def get_dist(self, coords1, coords2):
        # coords1 and coords2 of form [1, 1, 1]
        if self.dist_matrix is None:
            raise Exception("No dist_matrix available!")
        if self.coords2index(coords1) is None or self.coords2index(coords2) is None:
            return np.inf
        goal_a = self.index2coords(self.node2index(self.index2node(self.coords2index(coords1))))
        goal_b = self.index2coords(self.node2index(self.index2node(self.coords2index(coords2))))
        print("get_dist goals: {} // {}".format(goal_a, goal_b))
        return self.dist_matrix[self.index2node(self.coords2index(coords1)), self.index2node(self.coords2index(coords2))]

    def plot_mesh(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = self.colors.nonzero()
        ax.scatter(-X, Y, -Z, zdir='z', c='red')
        plt.savefig('mesh_plot.png')

    def print_mesh(self):
        for i in range(self.z_range):
            print("z={}:".format(i))
            print(self.colors[:-1, :-1, i])

    def print_numbers(self):
        for i in range(self.z_range):
            print("z={}:".format(i))
            print(self.numbers[:-1, :-1, i])


def main():
    obstacles = list()
    push_region = [1.3, 0.75, 0.6, 0.25, 0.35, 0.2]
    push_obstacles = [[1.3-0.125, 0.75, 0.6-0.18, 0.125, 0.04, 0.1]]
    mesh = DistanceMesh(region=push_region, spaces=[20, 20, 5], obstacles=push_obstacles)
    mesh.compute_cs_graph()
    mesh.compute_dist_matrix()
    #print(mesh.get_dist([0, 0.5, 0], [1, 0.5, 1]))
    mesh.print_mesh()

if __name__ == "__main__":
    main()
