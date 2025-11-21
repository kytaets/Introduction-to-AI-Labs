import matplotlib
matplotlib.use("TkAgg")

import random
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

GRID_SIZE = int(input("Enter grid size (press Enter for 5): ") or 5)
EDGES_TO_DELETE = int(input("How many walls/edges to remove (press Enter for 0): ") or 0)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

G_full = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)

pos = {}
for node in G_full.nodes():
    pos[node] = (node[1], GRID_SIZE - 1 - node[0])

road_graph = G_full.copy()
edge_list = list(road_graph.edges())
random.shuffle(edge_list)

deleted_edges = 0
for edge in edge_list:
    if deleted_edges >= EDGES_TO_DELETE:
        break
    road_graph.remove_edge(edge[0], edge[1])

    if nx.is_connected(road_graph):
        deleted_edges += 1
    else:
        road_graph.add_edge(edge[0], edge[1])

print("Edges deleted:", deleted_edges)

start_node = (0, 0)
end_node = (GRID_SIZE - 1, GRID_SIZE - 1)
print(f"Start: {start_node}, Goal: {end_node}")


def load_mnist_2to9():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_mask = (y_train >= 2)
    test_mask = (y_test >= 2)

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train_shifted = y_train - 2
    y_test_shifted = y_test - 2

    y_train_cat = tf.keras.utils.to_categorical(y_train_shifted, 8)
    y_test_cat = tf.keras.utils.to_categorical(y_test_shifted, 8)

    return (x_train, y_train_cat, y_train), (x_test, y_test_cat, y_test)


def build_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(8, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


print("Loading MNIST...")
(X_train, Y_train, y_train_raw), (X_test, Y_test, y_test_raw) = load_mnist_2to9()

cnn_model = build_cnn()

print("Training CNN...")
cnn_model.fit(X_train, Y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

_, test_acc = cnn_model.evaluate(X_test, Y_test, verbose=0)
print(f"Test accuracy: {test_acc}")

digit_images = defaultdict(list)
for image, label in zip(X_train, y_train_raw):
    if 2 <= label <= 9:
        if len(digit_images[label]) < 50:
            digit_images[label].append(image)


def get_random_image(digit):
    candidates = digit_images[digit]
    return random.choice(candidates)


def predict_digit(image):
    batch = np.expand_dims(image, axis=0)
    preds = cnn_model.predict(batch, verbose=0)
    class_idx = np.argmax(preds, axis=1)[0]
    return class_idx + 2


edge_speed_digit = {}
edge_speed_kmh = {}

for edge in road_graph.edges():
    d = random.randint(2, 9)
    edge_speed_digit[edge] = d
    edge_speed_kmh[edge] = d * 10


class KnowledgeBase:
    def __init__(self):
        self.memory = {}

    def add_observation(self, node, neighbors):
        if node not in self.memory:
            self.memory[node] = {"neighbors": set(neighbors), "visited": False}
        else:
            self.memory[node]["neighbors"].update(neighbors)

    def mark_visited(self, node):
        if node not in self.memory:
            self.memory[node] = {"neighbors": set(), "visited": True}
        else:
            self.memory[node]["visited"] = True

    def get_unvisited_neighbors(self, node):
        if node not in self.memory:
            return []
        result = []
        for neighbor in self.memory[node]["neighbors"]:
            visited = False
            if neighbor in self.memory and self.memory[neighbor]["visited"]:
                visited = True
            if not visited:
                result.append(neighbor)
        return result

    def get_all_neighbors(self, node):
        if node not in self.memory:
            return []
        return list(self.memory[node]["neighbors"])


class KnowledgeAgent:
    def __init__(self, graph, start, goal, kb):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.kb = kb
        self.path = []

    def heuristic(self, node):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def observe(self, node):
        neighbors = list(self.graph.neighbors(node))
        self.kb.add_observation(node, neighbors)
        self.kb.mark_visited(node)

    def solve(self):
        stack = [self.start]
        parents = {self.start: None}

        self.observe(self.start)

        while stack:
            current = stack[-1]

            if current == self.goal:
                tmp = current
                while tmp is not None:
                    self.path.append(tmp)
                    tmp = parents[tmp]
                self.path.reverse()
                return self.path

            unvisited = self.kb.get_unvisited_neighbors(current)

            if len(unvisited) == 0 and len(self.kb.get_all_neighbors(current)) == 0:
                real_neighbors = list(self.graph.neighbors(current))
                self.kb.add_observation(current, real_neighbors)
                unvisited = self.kb.get_unvisited_neighbors(current)

            if unvisited:
                next_node = min(unvisited, key=self.heuristic)
                parents[next_node] = current
                self.observe(next_node)
                stack.append(next_node)
            else:
                stack.pop()

        return []


kb = KnowledgeBase()
agent = KnowledgeAgent(road_graph, start_node, end_node, kb)
found_path = agent.solve()

if not found_path:
    print("Path not found")
else:
    print(f"Path found, length: {len(found_path)}")
    print(found_path)

segment_data = []
segment_log = []

if found_path:
    for i in range(len(found_path) - 1):
        u = found_path[i]
        v = found_path[i + 1]

        edge = (u, v)
        if edge not in edge_speed_digit:
            edge = (v, u)

        if edge not in edge_speed_digit:
            digit = random.randint(2, 9)
            edge_speed_digit[edge] = digit
            edge_speed_kmh[edge] = digit * 10

        true_digit = edge_speed_digit[edge]
        img = get_random_image(true_digit)
        predicted_digit = predict_digit(img)
        predicted_speed = predicted_digit * 10

        segment_data.append({
            "u": u,
            "v": v,
            "true_digit": true_digit,
            "predicted_digit": predicted_digit,
            "predicted_speed": predicted_speed,
            "img": img
        })

        segment_log.append(f"{u} -> {v}: sign {true_digit * 10}, CNN sees {predicted_speed}")

    for s in segment_log:
        print(s)
else:
    print("Nothing to draw")


def render_step(current_node, next_node, visited_nodes, path_so_far,
                img, true_digit, predicted_digit, speed_kmh):
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.title("Agent movement")

    nx.draw_networkx_edges(road_graph, pos, alpha=0.3)
    nx.draw_networkx_nodes(road_graph, pos, node_color="gray", node_size=200)

    if visited_nodes:
        nx.draw_networkx_nodes(
            road_graph, pos,
            nodelist=list(visited_nodes),
            node_color="yellow",
            node_size=230
        )

    nx.draw_networkx_nodes(
        road_graph, pos,
        nodelist=[current_node],
        node_color="green",
        node_size=260
    )
    nx.draw_networkx_nodes(
        road_graph, pos,
        nodelist=[end_node],
        node_color="red",
        node_size=260
    )

    if len(path_so_far) > 1:
        path_edges = [(path_so_far[i], path_so_far[i + 1]) for i in range(len(path_so_far) - 1)]
        nx.draw_networkx_edges(road_graph, pos, edgelist=path_edges, width=3)

    if next_node is not None:
        edge = (current_node, next_node)
        try:
            nx.draw_networkx_edges(road_graph, pos, edgelist=[edge], width=4)
            labels = {edge: f"{speed_kmh} km/h"}
            nx.draw_networkx_edge_labels(road_graph, pos, edge_labels=labels)
        except Exception:
            pass

    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Speed sign (MNIST)")

    if img is not None:
        plt.imshow(img[:, :, 0], cmap="gray")
        plt.axis("off")
        info_text = f"True: {true_digit * 10}\nPred: {predicted_digit * 10}"
        plt.text(0, 28, info_text, color="white", bbox=dict(facecolor="black"))
    else:
        plt.text(0.3, 0.5, "FINISH")
        plt.axis("off")

    plt.tight_layout()
    plt.pause(0.5)


if found_path:
    plt.ion()
    plt.figure(figsize=(10, 6))

    visited = set()
    path_so_far = []

    for idx, item in enumerate(segment_data):
        current = item["u"]
        nxt = item["v"]

        visited.add(current)
        if not path_so_far or path_so_far[-1] != current:
            path_so_far.append(current)

        render_step(
            current_node=current,
            next_node=nxt,
            visited_nodes=visited,
            path_so_far=path_so_far,
            img=item["img"],
            true_digit=item["true_digit"],
            predicted_digit=item["predicted_digit"],
            speed_kmh=item["predicted_speed"]
        )

        if idx == len(segment_data) - 1:
            visited.add(nxt)
            path_so_far.append(nxt)

    plt.ioff()
    render_step(
        current_node=path_so_far[-1],
        next_node=None,
        visited_nodes=visited,
        path_so_far=path_so_far,
        img=None,
        true_digit=0,
        predicted_digit=0,
        speed_kmh=0
    )
    plt.show()
