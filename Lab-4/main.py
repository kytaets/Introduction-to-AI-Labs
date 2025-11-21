import matplotlib
matplotlib.use("TkAgg")

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


GRID_SIZE = int(input("Enter grid size (5 for default): ") or 5)
EDGES_TO_REMOVE = int(input("Edges to remove (0 for default): ") or 0)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


G_full = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)

pos = {}
for node in G_full.nodes():
    x = node[1]
    y = GRID_SIZE - 1 - node[0]
    pos[node] = (x, y)

G_road = G_full.copy()
all_edges = list(G_road.edges())
random.shuffle(all_edges)
removed_count = 0

for edge in all_edges:
    if removed_count >= EDGES_TO_REMOVE:
        break
    G_road.remove_edge(*edge)
    if nx.is_connected(G_road):
        removed_count += 1
    else:
        G_road.add_edge(*edge)

print(f"Edges removed: {removed_count}")

start = (0, 0)
goal = (GRID_SIZE - 1, GRID_SIZE - 1)
print(f"Start = {start}, Goal = {goal}")


def load_mnist_2to9():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_mask = (y_train >= 2)
    test_mask = (y_test >= 2)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train_adj = y_train - 2
    y_test_adj = y_test - 2

    num_classes = 8
    y_train_cat = tf.keras.utils.to_categorical(y_train_adj, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test_adj, num_classes)

    return (x_train, y_train_cat, y_train), (x_test, y_test_cat, y_test)


def build_cnn_model(input_shape=(28, 28, 1), num_classes=8):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


print("Loading MNIST (digits 2–9)...")
(x_train, y_train, y_train_raw), (x_test, y_test, y_test_raw) = load_mnist_2to9()
print("Train set shape:", x_train.shape)
print("Test set shape:", x_test.shape)

model = build_cnn_model()
print(model.summary())

print("Training CNN on MNIST (2–9)...")
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy on digits 2–9: {test_acc:.3f}")


digit_to_images = defaultdict(list)
for img, lbl in zip(x_train, y_train_raw):
    if 2 <= lbl <= 9:
        if len(digit_to_images[lbl]) < 50:
            digit_to_images[lbl].append(img)
    if len(digit_to_images) == 8 and all(len(v) >= 50 for v in digit_to_images.values()):
        break


def get_random_digit_image(digit):
    imgs = digit_to_images[digit]
    return random.choice(imgs)


def recognize_speed_from_sign(img):
    x = np.expand_dims(img, axis=0)
    preds = model.predict(x, verbose=0)
    class_idx = np.argmax(preds, axis=1)[0]
    digit = class_idx + 2
    return digit


edge_speed_digit = {}
edge_speed_kmh = {}

for edge in G_road.edges():
    digit = random.randint(2, 9)
    edge_speed_digit[edge] = digit
    edge_speed_kmh[edge] = digit * 10


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


kb = KnowledgeBase()
agent = KnowledgeAgent(G_road, start, goal, kb)
path = agent.run()

if not path:
    print("Goal unreachable")
else:
    print(f"Found path of length {len(path)}")
    print("Path:", path)


segment_speeds = []
segment_speeds_pretty = []

if path:
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        if (u, v) in edge_speed_digit:
            edge = (u, v)
        elif (v, u) in edge_speed_digit:
            edge = (v, u)
        else:
            edge = (u, v)
            digit = random.randint(2, 9)
            edge_speed_digit[edge] = digit
            edge_speed_kmh[edge] = digit * 10

        true_digit = edge_speed_digit[edge]

        img = get_random_digit_image(true_digit)

        predicted_digit = recognize_speed_from_sign(img)
        predicted_speed = predicted_digit * 10

        segment_speeds.append((u, v, true_digit, predicted_digit, predicted_speed, img))
        segment_speeds_pretty.append(
            f"{u} -> {v}: true sign {true_digit*10} km/h, "
            f"CNN predicted {predicted_speed} km/h"
        )

    print("\nSpeeds on each path segment:")
    for line in segment_speeds_pretty:
        print(line)
else:
    print("No path, no speeds.")


def draw_step(current_node, next_node, visited_nodes, path_so_far,
              img, true_digit, predicted_digit, current_speed_kmh):
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.title("Agent movement (step-by-step)")

    nx.draw_networkx_edges(G_road, pos, alpha=0.3)

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
            node_size=230,
            edgecolors="black"
        )

    nx.draw_networkx_nodes(
        G_road, pos,
        nodelist=[current_node],
        node_color="green",
        node_size=260,
        edgecolors="black"
    )

    nx.draw_networkx_nodes(
        G_road, pos,
        nodelist=[goal],
        node_color="red",
        node_size=260,
        edgecolors="black"
    )

    if len(path_so_far) > 1:
        path_edges = list(zip(path_so_far[:-1], path_so_far[1:]))
        nx.draw_networkx_edges(
            G_road, pos,
            edgelist=path_edges,
            width=3
        )

    if next_node is not None:
        edge = (current_node, next_node)
        if edge not in G_road.edges():
            edge = (next_node, current_node)

        nx.draw_networkx_edges(
            G_road, pos,
            edgelist=[edge],
            width=4
        )

        current_edge_labels = {edge: f"{current_speed_kmh} km/h"}
        nx.draw_networkx_edge_labels(
            G_road, pos,
            edge_labels=current_edge_labels,
            font_size=9
        )

    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Speed sign (MNIST)")

    if img is not None:
        plt.imshow(img[:, :, 0], cmap="gray")
        plt.axis("off")
        plt.text(
            0, 27,
            f"True: {true_digit*10} km/h\nPred: {predicted_digit*10} km/h",
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.7)
        )
    else:
        plt.text(0.2, 0.5, "No sign (final node)", fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.pause(0.8)


if path:
    plt.ion()
    fig = plt.figure(figsize=(9, 5))

    visited = set()
    path_so_far = []

    for i, (u, v, true_digit, pred_digit, speed_kmh, img) in enumerate(segment_speeds):
        current_node = u
        next_node = v

        visited.add(current_node)
        if not path_so_far or path_so_far[-1] != current_node:
            path_so_far.append(current_node)

        draw_step(
            current_node=current_node,
            next_node=next_node,
            visited_nodes=visited,
            path_so_far=path_so_far,
            img=img,
            true_digit=true_digit,
            predicted_digit=pred_digit,
            current_speed_kmh=speed_kmh
        )

        if i == len(segment_speeds) - 1:
            visited.add(next_node)
            path_so_far.append(next_node)

    plt.ioff()
    draw_step(
        current_node=path_so_far[-2] if len(path_so_far) > 1 else path_so_far[-1],
        next_node=path_so_far[-1] if len(path_so_far) > 1 else None,
        visited_nodes=visited,
        path_so_far=path_so_far,
        img=None,
        true_digit=0,
        predicted_digit=0,
        current_speed_kmh=0
    )
    plt.show()

print("Step-by-step visualization completed (with edge speeds).")
