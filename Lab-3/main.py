import matplotlib
matplotlib.use("TkAgg")
import networkx as nx
import random
import matplotlib.pyplot as plt


grid_size = int(input('Enter grid size (5 for default): ') or 5)
edges_to_remove = int(input('Edges to remove: ') or 0)

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


class KnowledgeBase:
    def __init__(self):
        self.facts = {}

    def tell_observation(self, node, neighbors):
        if node not in self.facts:
            self.facts[node] = {
                "neighbors": set(neighbors),
                "visited": False
            }
        else:
            self.facts[node]["neighbors"].update(neighbors)

    def tell_visited(self, node):
        if node not in self.facts:
            self.facts[node] = {
                "neighbors": set(),
                "visited": True
            }
        else:
            self.facts[node]["visited"] = True

    def ask_unvisited_neighbors(self, node):
        if node not in self.facts:
            return []
        result = []
        for n in self.facts[node]["neighbors"]:
            if not self.facts.get(n, {"visited": False})["visited"]:
                result.append(n)
        return result

    def ask_all_neighbors(self, node):
        if node not in self.facts:
            return []
        return list(self.facts[node]["neighbors"])

    def is_visited(self, node):
        return self.facts.get(node, {"visited": False})["visited"]


class KnowledgeAgent:
    def __init__(self, graph, start, goal, kb: KnowledgeBase):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.kb = kb
        self.path = []

    def heuristic(self, node):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def perceive_and_tell(self, node):
        neighbors = list(self.graph.neighbors(node))
        self.kb.tell_observation(node, neighbors)
        self.kb.tell_visited(node)

    def run(self):
        stack = [self.start]
        parents = {self.start: None}
        self.perceive_and_tell(self.start)

        while stack:
            current = stack[-1]

            if current == self.goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parents[node]
                self.path = list(reversed(path))
                return self.path

            unvisited_neighbors = self.kb.ask_unvisited_neighbors(current)

            if not unvisited_neighbors and not self.kb.ask_all_neighbors(current):
                real_neighbors = list(self.graph.neighbors(current))
                self.kb.tell_observation(current, real_neighbors)
                unvisited_neighbors = self.kb.ask_unvisited_neighbors(current)

            if unvisited_neighbors:
                nxt = min(unvisited_neighbors, key=self.heuristic)
                parents[nxt] = current
                self.perceive_and_tell(nxt)
                stack.append(nxt)
            else:
                stack.pop()

        self.path = []
        return []


start = (0, 0)
goal = (grid_size - 1, grid_size - 1)
print(f"Start = {start}, Goal = {goal}")

kb = KnowledgeBase()
agent = KnowledgeAgent(G_road, start, goal, kb)
path = agent.run()

if not path:
    print("Goal unreachable")
else:
    print(f"Found path of length {len(path)}")
    print("Path:", path)


def draw_step(current_node, visited_nodes, path_so_far):
    plt.clf()
    plt.title("Knowledge-Based Agent movement")
    nx.draw_networkx_edges(G_road, pos, alpha=0.6)
    nx.draw_networkx_nodes(
        G_road, pos,
        nodelist=list(G_road.nodes()),
        node_color="lightgray",
        node_size=200,
        edgecolors="black"
    )

    if visited_nodes:
        nx.draw_networkx_nodes(
            G_road, pos,
            nodelist=list(visited_nodes),
            node_color="yellow",
            node_size=240,
            edgecolors="black"
        )

    nx.draw_networkx_nodes(
        G_road, pos,
        nodelist=[current_node],
        node_color="green",
        node_size=300,
        edgecolors="black"
    )

    nx.draw_networkx_nodes(
        G_road, pos,
        nodelist=[goal],
        node_color="red",
        node_size=300,
        edgecolors="black"
    )

    if len(path_so_far) > 1:
        path_edges = list(zip(path_so_far[:-1], path_so_far[1:]))
        nx.draw_networkx_edges(G_road, pos, edgelist=path_edges, width=3)

    plt.scatter([], [], c='green', label='Current position')
    plt.scatter([], [], c='yellow', label='Visited (KB knows)')
    plt.scatter([], [], c='red', label='Goal')
    plt.legend(loc='upper left')
    plt.axis('off')
    plt.pause(0.5)


plt.ion()
fig = plt.figure(figsize=(6, 6))

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

print("Visualization completed.")
