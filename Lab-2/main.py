import matplotlib
matplotlib.use("TkAgg")
import networkx as nx
import random
import matplotlib.pyplot as plt

grid_size = int(input('Enter grid size (5 for default): ') or 5)
edges_to_remove = int(input('Edges to remove: ') or 0)

# -----------------------------
# 1) Генерація графа (твоя Лаба №1)
# -----------------------------
G_full = nx.grid_2d_graph(grid_size, grid_size)

pos = {}
for node in G_full.nodes():
    x = node[1]
    y = grid_size - 1 - node[0]
    pos[node] = (x, y)

G_road = G_full.copy()
all_edges = list(G_road.edges())
random.shuffle(all_edges)
removed_count = 0

for edge in all_edges:
    if removed_count >= edges_to_remove:
        break
    G_road.remove_edge(*edge)
    if nx.is_connected(G_road):
        removed_count += 1
    else:
        G_road.add_edge(*edge)

print(f"Edges removed: {removed_count}")

# -----------------------------
# 2) Agent (DFS with stack)
# -----------------------------
class Agent:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.visited = set()
        self.path = []

    def run(self):
        stack = [self.start]
        self.visited.add(self.start)

        while stack:
            current = stack[-1]

            if current == self.goal:
                self.path = stack.copy()
                return stack

            neighbors = list(self.graph.neighbors(current))
            unvisited = [n for n in neighbors if n not in self.visited]

            if unvisited:
                nxt = random.choice(unvisited)
                self.visited.add(nxt)
                stack.append(nxt)
            else:
                stack.pop()

        self.path = []
        return []

# -----------------------------
# 3) Вибір старт/фініш
# -----------------------------
start = (0, 0)
goal = (grid_size - 1, grid_size - 1)
print(f"Start = {start}, Goal = {goal}")

agent = Agent(G_road, start, goal)
path = agent.run()

if not path:
    print("Ціль недосяжна (хоча граф мав бути зв'язний).")
else:
    print(f"Знайдений шлях довжини {len(path)}")

# -----------------------------
# 4) Візуалізація кроків
# -----------------------------
def draw_step(current_node, visited_nodes, path_so_far):
    plt.clf()
    plt.title("Рух агента — покрокова візуалізація")
    nx.draw_networkx_edges(G_road, pos, alpha=0.6)
    nx.draw_networkx_nodes(G_road, pos, nodelist=list(G_road.nodes()), node_color="lightgray", node_size=200, edgecolors="black")
    if visited_nodes:
        nx.draw_networkx_nodes(G_road, pos, nodelist=list(visited_nodes), node_color="yellow", node_size=240, edgecolors="black")
    nx.draw_networkx_nodes(G_road, pos, nodelist=[current_node], node_color="green", node_size=300, edgecolors="black")
    nx.draw_networkx_nodes(G_road, pos, nodelist=[goal], node_color="red", node_size=300, edgecolors="black")
    if len(path_so_far) > 1:
        path_edges = list(zip(path_so_far[:-1], path_so_far[1:]))
        nx.draw_networkx_edges(G_road, pos, edgelist=path_edges, width=3)

    plt.scatter([], [], c='green', label='Поточна позиція')
    plt.scatter([], [], c='yellow', label='Відвідані вершини')
    plt.scatter([], [], c='red', label='Ціль')
    plt.legend(loc='upper left')
    plt.axis('off')
    plt.pause(0.5)


plt.ion()
fig = plt.figure(figsize=(6,6))

if not path:
    nx.draw(G_road, pos, node_color="lightgray", edgecolors="black", node_size=200)
    plt.show()
else:
    visited = set()
    path_so_far = []
    for node in path:
        visited.add(node)
        path_so_far.append(node)
        draw_step(node, visited, path_so_far)

    plt.ioff()
    draw_step(path[-1], visited, path_so_far)
    plt.show()

print("Візуалізація завершена.")
