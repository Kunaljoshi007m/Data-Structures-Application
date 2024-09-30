import tkinter as tk
from tkinter import messagebox, scrolledtext
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque, defaultdict, Counter
import heapq
from typing import Any, List, Tuple
import numpy as np
import math
from itertools import permutations
from anytree import Node, RenderTree, AsciiStyle

color = "#28a745"
font_all = 16
width_all = 60

class TspBruteForce:
    def __init__(self, distanceMatrix):
        self.distanceMatrix = distanceMatrix
        self.numCities = len(distanceMatrix)

    def calculateTotalDistance(self, route):
        totalDistance = 0
        for i in range(len(route) - 1):
            totalDistance += self.distanceMatrix[route[i]][route[i+1]]
        totalDistance += self.distanceMatrix[route[-1]][route[0]]  
        return totalDistance

    def solve(self):
        minDistance = math.inf
        optimalRoute = None
        for perm in permutations(range(self.numCities)):
            perm = list(perm)
            currentDistance = self.calculateTotalDistance(perm)
            if currentDistance < minDistance:
                minDistance = currentDistance
                optimalRoute = perm
        return optimalRoute, minDistance

class TspGreedy:
    def __init__(self, distanceMatrix):
        self.distanceMatrix = distanceMatrix
        self.numCities = len(distanceMatrix)

    def findNearestPlace(self, currentPlace, unvisitedCities):
        nearestPlace = None
        minDistance = float('inf')
        for Place in unvisitedCities:
            if self.distanceMatrix[currentPlace][Place] < minDistance:
                minDistance = self.distanceMatrix[currentPlace][Place]
                nearestPlace = Place
        return nearestPlace

    def solve(self, startPlace=0):
        visitedCities = [startPlace]
        unvisitedCities = set(range(self.numCities)) - {startPlace}
        totalDistance = 0
        currentPlace = startPlace

        while unvisitedCities:
            nearestPlace = self.findNearestPlace(currentPlace, unvisitedCities)
            totalDistance += self.distanceMatrix[currentPlace][nearestPlace]
            currentPlace = nearestPlace
            visitedCities.append(currentPlace)
            unvisitedCities.remove(currentPlace)

        totalDistance += self.distanceMatrix[currentPlace][startPlace]
        visitedCities.append(startPlace)

        return visitedCities, totalDistance


def printRoute(route, distance, algorithmName, PlaceNames):
    routeStr = " -> ".join(PlaceNames[Place] for Place in route)
    return f"{algorithmName} Route: {routeStr}\nTotal Distance: {distance}\n"


def drawRouteGraph(route, distanceMatrix, PlaceNames, algorithmName, frame, startPlace):
    numCities = len(distanceMatrix)
    PlacePositions = {i: (np.cos(2 * np.pi * i / numCities), np.sin(2 * np.pi * i / numCities)) for i in range(numCities)}
    
    G_route = nx.DiGraph()

    for i in range(len(route)):
        G_route.add_node(route[i], pos=PlacePositions[route[i]])

    for i in range(len(route) - 1):
        G_route.add_edge(route[i], route[i + 1], weight=distanceMatrix[route[i]][route[i + 1]])
    G_route.add_edge(route[-1], route[0], weight=distanceMatrix[route[-1]][route[0]])

    posRoute = nx.get_node_attributes(G_route, 'pos')

    fig, ax = plt.subplots(figsize=(6, 6))
    
    node_colors = ['lightblue' if node != startPlace else 'orange' for node in G_route.nodes()]
    nx.draw(G_route, posRoute, with_labels=True, labels={i: PlaceNames[i] for i in range(len(PlaceNames))},
            node_color=node_colors, edge_color='gray', node_size=1000, font_size=12, ax=ax, arrows=True)

    edgeLabels = {(route[i], route[i + 1]): f'{distanceMatrix[route[i]][route[i + 1]]}' for i in range(len(route) - 1)}
    edgeLabels[(route[-1], route[0])] = f'{distanceMatrix[route[-1]][route[0]]}'
    nx.draw_networkx_edge_labels(G_route, posRoute, edge_labels=edgeLabels, font_color='red', ax=ax)

    plt.title(f"{algorithmName} Route")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    return canvas

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver (Brute-Force & Greedy)")
        
        inputFrame = tk.Frame(root)
        inputFrame.pack(side=tk.TOP, padx=10, pady=10)

        self.nameLabel = tk.Label(root, text="Kunal Joshi | S086", font=("Arial", 16), fg="blue")
        self.nameLabel.pack(side=tk.TOP, pady=10)

        self.numCitiesLabel = tk.Label(inputFrame, text="Enter number of places:", font=font_all, width=100)  # Increased width
        self.numCitiesLabel.grid(padx=5, pady=5)

        self.numCitiesEntry = tk.Entry(inputFrame, font=font_all, width=70)  # Increased width
        self.numCitiesEntry.grid(padx=5, pady=5)

        self.PlaceNamesLabel = tk.Label(inputFrame, text="Enter Place names (comma-separated):", font=font_all, width=100)  # Increased width
        self.PlaceNamesLabel.grid(padx=5, pady=5)

        self.PlaceNamesEntry = tk.Entry(inputFrame, font=font_all, width=70)  # Increased width
        self.PlaceNamesEntry.grid(padx=5, pady=5)

        self.distanceMatrixLabel = tk.Label(inputFrame, text="Enter the distance matrix (space or comma-separated rows):", font=font_all, width=100)  # Increased width
        self.distanceMatrixLabel.grid(padx=5, pady=5)

        # Create a Frame for the Text widget and Scrollbar
        distanceMatrixFrame = tk.Frame(inputFrame)
        distanceMatrixFrame.grid(padx=5, pady=5)

        self.distanceMatrixEntry = tk.Text(distanceMatrixFrame, height=15, width=70, font=font_all)  # Increased height and width
        self.distanceMatrixEntry.pack(side=tk.LEFT)

        self.distanceMatrixScrollbar = tk.Scrollbar(distanceMatrixFrame, command=self.distanceMatrixEntry.yview)  # Scrollbar
        self.distanceMatrixScrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.distanceMatrixEntry['yscrollcommand'] = self.distanceMatrixScrollbar.set  # Link scrollbar to text widget

        self.solveButton = tk.Button(inputFrame, text="Solve TSP", command=self.showResultsWindow, bg=color, font=font_all, width=20)
        self.solveButton.grid(pady=10)

    def showResultsWindow(self):
        resultWindow = tk.Toplevel(self.root)
        resultWindow.title("TSP Solution Results")

        numCities = int(self.numCitiesEntry.get())
        PlaceNames = self.PlaceNamesEntry.get().split(',')

        distanceMatrixInput = self.distanceMatrixEntry.get("1.0", tk.END).strip().splitlines()
        distanceMatrix = [list(map(int, filter(None, row.replace(' ', ',').split(',')))) for row in distanceMatrixInput]

        tspBruteForce = TspBruteForce(distanceMatrix)
        bruteForceRoute, bruteForceMinDistance = tspBruteForce.solve()
        bruteForceResult = printRoute(bruteForceRoute, bruteForceMinDistance, "Brute-Force", PlaceNames)

        tspGreedy = TspGreedy(distanceMatrix)
        greedyRoute, greedyMinDistance = tspGreedy.solve(startPlace=0)
        greedyResult = printRoute(greedyRoute, greedyMinDistance, "Greedy", PlaceNames)

        outputLabel = tk.Label(resultWindow, text=bruteForceResult + "\n" + greedyResult, justify="left", font=font_all)
        outputLabel.pack(side=tk.TOP, pady=10)

        graphFrameBruteForce = tk.Frame(resultWindow)
        graphFrameBruteForce.pack(side=tk.LEFT, padx=10, pady=10)

        graphFrameGreedy = tk.Frame(resultWindow)
        graphFrameGreedy.pack(side=tk.RIGHT, padx=10, pady=10)

        startPlace = 0
        canvasBruteForce = drawRouteGraph(bruteForceRoute, distanceMatrix, PlaceNames, "Brute-Force", graphFrameBruteForce, startPlace)
        canvasGreedy = drawRouteGraph(greedyRoute, distanceMatrix, PlaceNames, "Greedy", graphFrameGreedy, startPlace)

class PriorityQueue:
    def __init__(self):
        self.items: List[Tuple[int, Any]] = []

    def isEmpty(self) -> bool:
        return len(self.items) == 0

    def enqueue(self, item: Any, priority: int) -> None:
        heapq.heappush(self.items, (priority, item))

    def dequeue(self) -> Any:
        if self.isEmpty():
            return "Priority Queue is empty. Cannot dequeue."
        return heapq.heappop(self.items)[1]

    def peek(self) -> Any:
        if self.isEmpty():
            return "Priority Queue is empty."
        return heapq.nsmallest(1, self.items)[0][1]

    def size(self) -> int:
        return len(self.items)

    def traverse(self) -> List[str]:
        if self.isEmpty():
            return "Priority Queue is empty."
        return [f"Item: {item}, Priority: {priority}" for priority, item in sorted(self.items)]

class PriorityQueueGUI:
    def __init__(self, root):
        self.queue = PriorityQueue()
        
        self.root = root
        self.root.title("S086-kunal joshi - Priority Queue Operations")

        self.element_label = tk.Label(root, text="Element:", font=font_all)
        self.element_label.place(relx=0.2, rely=0.2, anchor=tk.CENTER)
        
        self.element_entry = tk.Entry(root, font=font_all)
        self.element_entry.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
        
        self.priority_label = tk.Label(root, text="Priority:", font=font_all)
        self.priority_label.place(relx=0.2, rely=0.3, anchor=tk.CENTER)
        
        self.priority_entry = tk.Entry(root, font=font_all)
        self.priority_entry.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        
        self.enqueue_button = tk.Button(root, text="Enqueue", command=self.enqueue, bg=color, font=font_all, width=width_all)
        self.enqueue_button.place(relx=0.4, rely=0.4, anchor=tk.CENTER)
        
        self.dequeue_button = tk.Button(root, text="Dequeue", command=self.dequeue, bg=color, font=font_all, width=width_all)
        self.dequeue_button.place(relx=0.6, rely=0.4, anchor=tk.CENTER)
        
        self.peek_button = tk.Button(root, text="Peek", command=self.peek, bg=color, font=font_all, width=width_all)
        self.peek_button.place(relx=0.4, rely=0.5, anchor=tk.CENTER)
        
        self.isempty_button = tk.Button(root, text="Is Empty", command=self.is_empty, bg=color, font=font_all, width=width_all)
        self.isempty_button.place(relx=0.6, rely=0.5, anchor=tk.CENTER)
        
        self.size_button = tk.Button(root, text="Get Size", command=self.get_size, bg=color, font=font_all, width=width_all)
        self.size_button.place(relx=0.4, rely=0.6, anchor=tk.CENTER)
        
        self.traverse_button = tk.Button(root, text="Traverse", command=self.traverse, bg=color, font=font_all, width=width_all)
        self.traverse_button.place(relx=0.6, rely=0.6, anchor=tk.CENTER)
        
        self.result_label = tk.Label(root, text="", font=font_all)
        self.result_label.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
        
    def enqueue(self):
        element = self.element_entry.get()
        priority = self.priority_entry.get()
        if element and priority.isdigit():
            self.queue.enqueue(element, int(priority))
            self.result_label.config(text=f"Enqueued: {element} with priority {priority}")
            self.element_entry.delete(0, tk.END)
            self.priority_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter a valid element and priority.")
    
    def dequeue(self):
        result = self.queue.dequeue()
        self.result_label.config(text=f"Dequeued: {result}")
    
    def peek(self):
        result = self.queue.peek()
        self.result_label.config(text=f"Peek: {result}")
    
    def is_empty(self):
        result = self.queue.isEmpty()
        self.result_label.config(text=f"Is Empty: {result}")
    
    def get_size(self):
        result = self.queue.size()
        self.result_label.config(text=f"Size: {result}")
    
    def traverse(self):
        result = self.queue.traverse()
        self.result_label.config(text=f"Queue: {result}")

class HashTableWTC:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def _hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash_function(key)
        if self.table[index] is None:
            self.table[index] = (key, value)
            return f"Inserted: ({key}, {value}) at index {index}"
        else:
            return f"Collision occurred for key: {key}, but collisions are not handled."

    def delete(self, key):
        index = self._hash_function(key)
        if self.table[index] is not None and self.table[index][0] == key:
            self.table[index] = None
            return f"Deleted key: {key} from index {index}"
        else:
            return f"Key: {key} not found for deletion."

    def traverse(self):
        result = []
        for index, element in enumerate(self.table):
            if element is not None:
                result.append(f"Index {index}: {element}")
            else:
                result.append(f"Index {index}: Empty")
        return "\n".join(result)


class HashTableGUIWTC:
    def __init__(self, root):
        self.root = root
        self.root.title("Hash Table GUI")
        self.hash_table = None

        name_label = tk.Label(root, text="Kunal joshi | S086", font=font_all)
        name_label.pack()

        size_frame = tk.Frame(root)
        size_frame.pack(pady=10)

        tk.Label(size_frame, text="Enter size of the Hash Table:", font=font_all).grid(row=0, column=0)
        self.size_entry = tk.Entry(size_frame, font=font_all, width=width_all)
        self.size_entry.grid(row=0, column=1)

        tk.Button(size_frame, text="Create Hash Table", font=font_all, bg=bg_button, command=self.create_hash_table).grid(row=0, column=2)

        action_frame = tk.Frame(root)
        action_frame.pack(pady=10)

        tk.Label(action_frame, text="Key:", font=font_all).grid(row=0, column=0)
        self.key_entry = tk.Entry(action_frame, font=font_all, width=width_all)
        self.key_entry.grid(row=0, column=1)

        tk.Label(action_frame, text="Value:", font=font_all).grid(row=1, column=0)
        self.value_entry = tk.Entry(action_frame, font=font_all, width=width_all)
        self.value_entry.grid(row=1, column=1)

        tk.Button(action_frame, text="Insert", font=font_all, bg="#28a745", command=self.insert_value).grid(row=0, column=2)
        tk.Button(action_frame, text="Delete", font=font_all, bg="#28a745", command=self.delete_value).grid(row=1, column=2)

        traverse_frame = tk.Frame(root)
        traverse_frame.pack(pady=10)

        tk.Button(traverse_frame, text="Traverse", font=font_all, width=width_all, bg=bg_button, command=self.traverse_table).pack()

        self.result_text = tk.Text(traverse_frame, font=font_all, height=10, width=50)
        self.result_text.pack()

    def create_hash_table(self):
        size = self.size_entry.get()
        if size.isdigit():
            self.hash_table = HashTableWTC(int(size))
            messagebox.showinfo("Success", "Hash Table created successfully.")
        else:
            messagebox.showerror("Error", "Please enter a valid size.")

    def insert_value(self):
        if not self.hash_table:
            messagebox.showerror("Error", "Please create the hash table first.")
            return

        key = self.key_entry.get()
        value = self.value_entry.get()

        if key and value:
            result = self.hash_table.insert(key, value)
            messagebox.showinfo("Insert Result", result)
        else:
            messagebox.showerror("Error", "Key and Value cannot be empty.")

    def delete_value(self):
        if not self.hash_table:
            messagebox.showerror("Error", "Please create the hash table first.")
            return
        key = self.key_entry.get()

        if key:
            result = self.hash_table.delete(key)
            messagebox.showinfo("Delete Result", result)
        else:
            messagebox.showerror("Error", "Key cannot be empty.")

    def traverse_table(self):
        if not self.hash_table:
            messagebox.showerror("Error", "Please create the hash table first.")
            return

        result = self.hash_table.traverse()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

class FamilyTreeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Huffman Family Tree")
        self.root.configure(bg="#2e8b57")
        
        self.members = {}

        self.parent_label = tk.Label(root, text="Parent Name:", bg="#2e8b57", fg="white")
        self.parent_label.pack(pady=5)
        self.parent_entry = tk.Entry(root)
        self.parent_entry.pack(pady=5)

        self.child_label = tk.Label(root, text="Child Name:", bg="#2e8b57", fg="white")
        self.child_label.pack(pady=5)
        self.child_entry = tk.Entry(root)
        self.child_entry.pack(pady=5)

        self.add_button = tk.Button(root, text="Add Relationship", command=self.add_relationship, bg="#4caf50", fg="white", borderwidth=2, relief="solid")
        self.add_button.pack(pady=10)

        self.draw_button = tk.Button(root, text="Draw Family Tree", command=self.draw_family_tree, bg="#4caf50", fg="white", borderwidth=2, relief="solid")
        self.draw_button.pack(pady=10)

        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    def add_relationship(self):
        parent_name = self.parent_entry.get()
        child_name = self.child_entry.get()
        if parent_name and child_name:
            self.add_family_member(parent_name, child_name)
            self.parent_entry.delete(0, tk.END)
            self.child_entry.delete(0, tk.END)
            messagebox.showinfo("Success", "Relationship added.")
        else:
            messagebox.showerror("Error", "Please enter both parent and child names.")

    def add_family_member(self, parent_name, child_name):
        parent_node = self.members.get(parent_name)
        if not parent_node:
            parent_node = Node(parent_name)
            self.members[parent_name] = parent_node

        child_node = self.members.get(child_name)
        if not child_node:
            child_node = Node(child_name, parent=parent_node)
            self.members[child_name] = child_node
        else:
            child_node.parent = parent_node

    def find_root(self):
        for node in self.members.values():
            if node.is_root:
                return node

    def draw_family_tree(self):
        root = self.find_root()
        if root:
            self.figure.clear()
            G = nx.DiGraph()
            self.add_edges(G, root)
            ax = self.figure.add_subplot(111)
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=1000, ax=ax, node_color="lightblue", font_size=10, font_color="black")
            self.canvas.draw()
        else:
            messagebox.showerror("Error", "No root found. Please add relationships first.")

    def add_edges(self, graph, node):
        if node is not None:
            for child in node.children:
                graph.add_edge(node.name, child.name)
                self.add_edges(graph, child)

class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, element):
        self.queue.append(element)

    def dequeue(self):
        if self.isEmpty():
            return "Queue is empty"
        return self.queue.pop(0)

    def peek(self):
        if self.isEmpty():
            return "Queue is empty"
        return self.queue[0]

    def isEmpty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

    def traverse(self):
        return self.queue

class QueueGUI:
    def __init__(self, root):
        self.queue = Queue()
        
        self.root = root
        self.root.title("Queue Operations")
        
        self.element_label = tk.Label(root, text="Element:", font=font_all, width=width_all)
        self.element_label.place(relx=0.3, rely=0.2, anchor=tk.CENTER)
        
        self.element_entry = tk.Entry(root, font=font_all, width=width_all)
        self.element_entry.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
        
        self.enqueue_button = tk.Button(root, text="Enqueue", command=self.enqueue, bg=color, font=font_all, width=width_all)
        self.enqueue_button.place(relx=0.4, rely=0.3, anchor=tk.CENTER)
        
        self.dequeue_button = tk.Button(root, text="Dequeue", command=self.dequeue, bg=color, font=font_all, width=width_all)
        self.dequeue_button.place(relx=0.6, rely=0.3, anchor=tk.CENTER)
        
        self.peek_button = tk.Button(root, text="Peek", command=self.peek, bg=color, font=font_all, width=width_all)
        self.peek_button.place(relx=0.4, rely=0.4, anchor=tk.CENTER)
        
        self.isempty_button = tk.Button(root, text="Is Empty", command=self.is_empty, bg=color, font=font_all, width=width_all)
        self.isempty_button.place(relx=0.6, rely=0.4, anchor=tk.CENTER)
        
        self.size_button = tk.Button(root, text="Get Size", command=self.get_size, bg=color, font=font_all, width=width_all)
        self.size_button.place(relx=0.4, rely=0.5, anchor=tk.CENTER)
        
        self.traverse_button = tk.Button(root, text="Traverse", command=self.traverse, bg=color, font=font_all, width=width_all)
        self.traverse_button.place(relx=0.6, rely=0.5, anchor=tk.CENTER)
        
        self.result_label = tk.Label(root, text="", font=font_all)
        self.result_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
        
    def enqueue(self):
        element = self.element_entry.get()
        if element:
            self.queue.enqueue(element)
            self.result_label.config(text=f"Enqueued: {element}")
        else:
            messagebox.showwarning("Input Error", "Please enter an element to enqueue.")
    
    def dequeue(self):
        result = self.queue.dequeue()
        self.result_label.config(text=f"Dequeued: {result}")
    
    def peek(self):
        result = self.queue.peek()
        self.result_label.config(text=f"Peek: {result}")
    
    def is_empty(self):
        result = self.queue.isEmpty()
        self.result_label.config(text=f"Is Empty: {result}")
    
    def get_size(self):
        result = self.queue.size()
        self.result_label.config(text=f"Size: {result}")
    
    def traverse(self):
        result = self.queue.traverse()
        self.result_label.config(text=f"Queue: {result}")

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, items):
        for item, priority in items:
            self.queue.append((item, priority))
        self.queue.sort(key=lambda x: x[1])  
        return [f"Enqueued: {item} with priority {priority}" for item, priority in items]

    def dequeue(self):
        if self.is_empty():
            return "Priority Queue is empty. Cannot dequeue."
        item = self.queue.pop(0)[0]
        return f"Dequeued: {item}"

    def traverse(self):
        if self.is_empty():
            return "Priority Queue is empty."
        else:
            return "Priority Queue contains:\n" + "\n".join([f"Item: {item}, Priority: {priority}" for item, priority in self.queue])

    def size(self):
        return f"Queue size: {len(self.queue)}"

    def peek(self):
        if self.is_empty():
            return "Priority Queue is empty."
        item = self.queue[0][0]
        return f"Highest priority item is: {item}"

class PriorityQueueGUI:
    def __init__(self, root):
        self.pq = PriorityQueue()
        
        self.root = root
        self.root.title("S086--Kunal joshi Priority Queue Operations")
        
        self.element_label = tk.Label(root, text="Item and Priority (item priority):", font=font_all)
        self.element_label.place(relx=0.3, rely=0.1, anchor=tk.CENTER)
        
        self.element_entry = tk.Entry(root, font=font_all)
        self.element_entry.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
        
        self.enqueue_button = tk.Button(root, text="Enqueue", command=self.enqueue, bg=color, font=font_all, width=width_all)
        self.enqueue_button.place(relx=0.4, rely=0.3, anchor=tk.CENTER)
        
        self.dequeue_button = tk.Button(root, text="Dequeue", command=self.dequeue, bg=color, font=font_all, width=width_all)
        self.dequeue_button.place(relx=0.6, rely=0.3, anchor=tk.CENTER)
        
        self.peek_button = tk.Button(root, text="Peek", command=self.peek, bg=color, font=font_all, width=width_all)
        self.peek_button.place(relx=0.4, rely=0.4, anchor=tk.CENTER)
        
        self.isempty_button = tk.Button(root, text="Is Empty", command=self.is_empty, bg=color, font=font_all, width=width_all)
        self.isempty_button.place(relx=0.6, rely=0.4, anchor=tk.CENTER)
        
        self.size_button = tk.Button(root, text="Get Size", command=self.get_size, bg=color, font=font_all, width=width_all)
        self.size_button.place(relx=0.4, rely=0.5, anchor=tk.CENTER)
        
        self.traverse_button = tk.Button(root, text="Traverse", command=self.traverse, bg=color, font=font_all, width=width_all)
        self.traverse_button.place(relx=0.6, rely=0.5, anchor=tk.CENTER)
        
        self.result_label = tk.Label(root, text="", font=font_all)
        self.result_label.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
        
    def enqueue(self):
        elements = self.element_entry.get().split()
        if len(elements) % 2 != 0:
            messagebox.showwarning("Input Error", "Please enter elements and priorities in pairs.")
            return
        
        items = []
        for i in range(0, len(elements), 2):
            item = elements[i]
            try:
                priority = int(elements[i+1])
            except ValueError:
                messagebox.showwarning("Input Error", "Priority must be an integer.")
                return
            items.append((item, priority))
        
        result = self.pq.enqueue(items)
        self.result_label.config(text="\n".join(result))
    
    def dequeue(self):
        result = self.pq.dequeue()
        self.result_label.config(text=result)
    
    def peek(self):
        result = self.pq.peek()
        self.result_label.config(text=result)
    
    def is_empty(self):
        result = self.pq.is_empty()
        self.result_label.config(text=f"Is Empty: {result}")
    
    def get_size(self):
        result = self.pq.size()
        self.result_label.config(text=result)
    
    def traverse(self):
        result = self.pq.traverse()
        self.result_label.config(text=result)

def create_node(data):
    return {'data': data, 'next': None}

def append(head, data):
    new_node = create_node(data)
    if head is None:
        return new_node
    last = head
    while last['next']:
        last = last['next']
    last['next'] = new_node
    return head

def prepend(head, data):
    new_node = create_node(data)
    if head is None:
        return new_node
    new_node['next'] = head
    return new_node

def delete(head, key):
    curr = head
    prev = None
    while curr:
        if curr['data'] == key:
            if prev:
                prev['next'] = curr['next']
            else:
                head = curr['next']
            return head
        prev = curr
        curr = curr['next']
    return head

def traverse(head):
    elems = []
    curr = head
    while curr:
        elems.append(curr['data'])
        curr = curr['next']
    return elems

class LinkedListGUISingly:
    def __init__(self, root):
        self.head = None
        self.root = root
        self.root.title("Linked List Operations")

        self.title_label = tk.Label(root, text="Singly Linked List Operations", font=("Arial", 16))
        self.title_label.grid(row=0, column=0, columnspan=3, pady=10)

        self.append_label = tk.Label(root, text="Append Values (comma-separated):", font=font_all)
        self.append_label.grid(row=1, column=0, sticky="e", padx=10)
        self.append_entry = tk.Entry(root)
        self.append_entry.grid(row=1, column=1, padx=10)
        self.append_button = tk.Button(root, text="Append", command=self.append_value, font=font_all, width=width_all)
        self.append_button.grid(row=1, column=2, padx=10)

        self.prepend_label = tk.Label(root, text="Prepend Value:", font=font_all)
        self.prepend_label.grid(row=2, column=0, sticky="e", padx=10)
        self.prepend_entry = tk.Entry(root)
        self.prepend_entry.grid(row=2, column=1, padx=10)
        self.prepend_button = tk.Button(root, text="Prepend", command=self.prepend_value, font=font_all, width=width_all)
        self.prepend_button.grid(row=2, column=2, padx=10)

        self.delete_label = tk.Label(root, text="Delete Value:", font=font_all)
        self.delete_label.grid(row=3, column=0, sticky="e", padx=10)
        self.delete_entry = tk.Entry(root)
        self.delete_entry.grid(row=3, column=1, padx=10)
        self.delete_button = tk.Button(root, text="Delete", command=self.delete_value, font=font_all, width=width_all)
        self.delete_button.grid(row=3, column=2, padx=10)

        self.display_button = tk.Button(root, text="Display List", command=self.display_list, font=font_all, width=width_all)
        self.display_button.grid(row=4, column=0, columnspan=3, pady=10)

        self.result_label = tk.Label(root, text="", font=font_all, fg="blue")
        self.result_label.grid(row=5, column=0, columnspan=3, pady=10)

    def append_value(self):
        data = self.append_entry.get().split(',')
        for i in data:
            self.head = append(self.head, i.strip())
        self.append_entry.delete(0, tk.END)
        self.result_label.config(text="Values appended.")

    def prepend_value(self):
        data = self.prepend_entry.get().strip()
        self.head = prepend(self.head, data)
        self.prepend_entry.delete(0, tk.END)
        self.result_label.config(text="Value prepended.")

    def delete_value(self):
        key = self.delete_entry.get().strip()
        self.head = delete(self.head, key)
        self.delete_entry.delete(0, tk.END)
        self.result_label.config(text=f"Value '{key}' deleted if it existed.")

    def display_list(self):
        elems = traverse(self.head)
        self.result_label.config(text="List: " + str(elems))

import tkinter as tk

color = "#28a745"
font_all = ("Arial", 16)
width_all = 15

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node
        new_node.prev = last

    def prepend(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        self.head.prev = new_node
        new_node.next = self.head
        self.head = new_node

    def delete(self, key):
        curr = self.head
        while curr:
            if curr.data == key and curr == self.head:
                if not curr.next:
                    curr = None
                    self.head = None
                    return
                else:
                    next = curr.next
                    curr.next = None
                    next.prev = None
                    curr = None
                    self.head = next
                    return
            elif curr.data == key:
                if curr.next:
                    prev = curr.prev
                    next = curr.next
                    prev.next = next
                    next.prev = prev
                    curr.next = None
                    curr.prev = None
                    curr = None
                    return
                else:
                    prev = curr.prev
                    prev.next = None
                    curr.prev = None
                    curr = None
                    return
            curr = curr.next

    def traverse(self):
        elems = []
        curr = self.head
        while curr:
            elems.append(curr.data)
            curr = curr.next
        return elems

class LinkedListGUIDoubly:
    def __init__(self, root):
        self.dll = DoublyLinkedList()
        self.root = root
        self.root.title("Doubly Linked List Operations")
        self.root.configure(bg="#f0f0f0")

        self.create_widgets()
        self.arrange_widgets()

    def create_widgets(self):
        self.append_label = tk.Label(self.root, text="Append:", bg="#f0f0f0", font=font_all, width=width_all)
        self.append_entry = tk.Entry(self.root, font=font_all, width=width_all)
        self.append_button = tk.Button(self.root, text="Append", command=self.append_value, font=font_all, bg=color, fg="white", width=width_all)

        self.prepend_label = tk.Label(self.root, text="Prepend:", bg="#f0f0f0", font=font_all, width=width_all)
        self.prepend_entry = tk.Entry(self.root, font=font_all, width=width_all)
        self.prepend_button = tk.Button(self.root, text="Prepend", command=self.prepend_value, font=font_all, bg=color, fg="white", width=width_all)

        self.delete_label = tk.Label(self.root, text="Delete:", bg="#f0f0f0", font=font_all, width=width_all)
        self.delete_entry = tk.Entry(self.root, font=font_all, width=width_all)
        self.delete_button = tk.Button(self.root, text="Delete", command=self.delete_value, font=font_all, bg=color, fg="white", width=width_all)

        self.display_button = tk.Button(self.root, text="Display", command=self.display_list, font=font_all, bg=color, fg="white", width=width_all)

        self.result_label = tk.Label(self.root, text="", bg="#f0f0f0", font=font_all, width=width_all)

    def arrange_widgets(self):
        self.append_label.grid(row=0, column=0, padx=10, pady=10)
        self.append_entry.grid(row=0, column=1, padx=10, pady=10)
        self.append_button.grid(row=0, column=2, padx=10, pady=10)

        self.prepend_label.grid(row=1, column=0, padx=10, pady=10)
        self.prepend_entry.grid(row=1, column=1, padx=10, pady=10)
        self.prepend_button.grid(row=1, column=2, padx=10, pady=10)

        self.delete_label.grid(row=2, column=0, padx=10, pady=10)
        self.delete_entry.grid(row=2, column=1, padx=10, pady=10)
        self.delete_button.grid(row=2, column=2, padx=10, pady=10)

        self.display_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.result_label.grid(row=4, column=0, columnspan=3, pady=10)

        for i in range(3):
            self.root.grid_columnconfigure(i, weight=1)
        self.root.grid_rowconfigure(4, weight=1)

    def append_value(self):
        data = self.append_entry.get().split(',')
        for i in data:
            self.dll.append(i.strip())
        self.append_entry.delete(0, tk.END)

    def prepend_value(self):
        data = self.prepend_entry.get()
        self.dll.prepend(data.strip())
        self.prepend_entry.delete(0, tk.END)

    def delete_value(self):
        key = self.delete_entry.get()
        self.dll.delete(key.strip())
        self.delete_entry.delete(0, tk.END)

    def display_list(self):
        elems = self.dll.traverse()
        self.result_label.config(text=str(elems))

width_all=15
font_all=("Helvetica",15)
bg_button="#007bff"

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, root, key):
        if key < root.val:
            if root.left is None:
                root.left = Node(key)
            else:
                self._insert(root.left, key)
        else:
            if root.right is None:
                root.right = Node(key)
            else:
                self._insert(root.right, key)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, root, key):
        if root is None:
            return root
        if key < root.val:
            root.left = self._delete(root.left, key)
        elif key > root.val:
            root.right = self._delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            min_larger_node = self._get_min(root.right)
            root.val = min_larger_node.val
            root.right = self._delete(root.right, min_larger_node.val)
        return root

    def _get_min(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder_traversal(self):
        return self._inorder_traversal(self.root)

    def _inorder_traversal(self, root):
        res = []
        if root:
            res = self._inorder_traversal(root.left)
            res.append(root.val)
            res = res + self._inorder_traversal(root.right)
        return res

    def preorder_traversal(self):
        return self._preorder_traversal(self.root)

    def _preorder_traversal(self, root):
        res = []
        if root:
            res.append(root.val)
            res = res + self._preorder_traversal(root.left)
            res = res + self._preorder_traversal(root.right)
        return res

    def postorder_traversal(self):
        return self._postorder_traversal(self.root)

    def _postorder_traversal(self, root):
        res = []
        if root:
            res = self._postorder_traversal(root.left)
            res = res + self._postorder_traversal(root.right)
            res.append(root.val)
        return res

class BinaryTreeGUI:
    def __init__(self, root):
        self.tree = BinaryTree()

        self.root = root
        self.root.title("Binary Tree GUI Kunal joshi_S086")

        self.label = tk.Label(root, text="Enter a number:",font=font_all)
        self.label.pack()

        self.entry = tk.Entry(root,font=font_all,width=width_all)
        self.entry.pack()

        self.insert_button = tk.Button(root, text="Insert", command=self.insert,font=font_all,width=width_all,bg="#28a745")
        self.insert_button.pack()

        self.delete_button = tk.Button(root, text="Delete", command=self.delete,font=font_all,width=width_all,bg="#28a745")
        self.delete_button.pack()

        self.traversal_label = tk.Label(root, text="Traversal:",width=width_all,font=font_all)
        self.traversal_label.pack()

        self.inorder_button = tk.Button(root, text="Inorder Traversal", command=self.show_inorder,font=font_all,width=width_all,bg=bg_button)
        self.inorder_button.pack()

        self.preorder_button = tk.Button(root, text="Preorder Traversal", command=self.show_preorder,font=font_all,width=width_all,bg=bg_button)
        self.preorder_button.pack()

        self.postorder_button = tk.Button(root, text="Postorder Traversal", command=self.show_postorder,font=font_all,width=width_all,bg=bg_button)
        self.postorder_button.pack()

        self.result_label = tk.Label(root, text="",font=font_all,width=50)
        self.result_label.pack()

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.update_tree_diagram()

    def insert(self):
        try:
            key = int(self.entry.get())
            self.tree.insert(key)
            messagebox.showinfo("Success", f"Inserted {key} into the tree.")
            self.entry.delete(0, tk.END)
            self.update_tree_diagram()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer.")

    def delete(self):
        try:
            key = int(self.entry.get())
            self.tree.delete(key)
            messagebox.showinfo("Success", f"Deleted {key} from the tree.")
            self.entry.delete(0, tk.END)
            self.update_tree_diagram()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer.")

    def show_inorder(self):
        result = self.tree.inorder_traversal()
        self.result_label.config(text=f"Inorder: {result}")

    def show_preorder(self):
        result = self.tree.preorder_traversal()
        self.result_label.config(text=f"Preorder: {result}")

    def show_postorder(self):
        result = self.tree.postorder_traversal()
        self.result_label.config(text=f"Postorder: {result}")

    def update_tree_diagram(self):
        def add_edges(graph, node):
            if node is not None:
                if node.left:
                    graph.add_edge(node.val, node.left.val)
                    add_edges(graph, node.left)
                if node.right:
                    graph.add_edge(node.val, node.right.val)
                    add_edges(graph, node.right)

        G = nx.DiGraph()
        add_edges(G, self.tree.root)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=1000, ax=ax, node_color="lightblue", font_size=10, font_color="black")
        self.canvas.draw()

class Node:
    def __init__(self, char=None, freq=None):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Huffman Encoding & Decoding")
        self.root.configure(bg="#f0f0f0")
        self.setup_ui()

    def setup_ui(self):
        # Input Frame
        frame_input = tk.Frame(self.root, bg="#f0f0f0")
        frame_input.pack(pady=10)

        tk.Label(frame_input, text="Kunal Joshi S086", font=30, bg="#f0f0f0").grid(row=0, column=0, columnspan=2, padx=5, pady=(0, 10))

        tk.Label(frame_input, text="Enter Text:", font=font_all, bg="#f0f0f0").grid(row=1, column=0, padx=5)
        self.input_text = scrolledtext.ScrolledText(frame_input, height=5, width=50, font=font_all)  # Multi-line input
        self.input_text.grid(row=1, column=1, padx=5)

        # Button Frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Run Huffman", command=self.run_huffman, font=font_all, width=width_all, bg="#28a745").grid(row=0, column=0, padx=5)

        # Output Frame
        frame_output = tk.Frame(self.root, bg="#f0f0f0")
        frame_output.pack(pady=10)

        tk.Label(frame_output, text="Original Bits:", font=font_all, bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.original_text = scrolledtext.ScrolledText(frame_output, height=5, width=50, font=font_all)
        self.original_text.grid(row=0, column=1, padx=5)

        tk.Label(frame_output, text="Encoded Data:", font=font_all, bg="#f0f0f0").grid(row=1, column=0, padx=5)
        self.encoded_text = scrolledtext.ScrolledText(frame_output, height=5, width=50, font=font_all)
        self.encoded_text.grid(row=1, column=1, padx=5)

        tk.Label(frame_output, text="Decoded Data:", font=font_all, bg="#f0f0f0").grid(row=2, column=0, padx=5)
        self.decoded_text = scrolledtext.ScrolledText(frame_output, height=5, width=50, font=font_all)
        self.decoded_text.grid(row=2, column=1, padx=5)

        tk.Label(frame_output, text="Codebook:", font=font_all, bg="#f0f0f0").grid(row=3, column=0, padx=5)
        self.codebook_text = scrolledtext.ScrolledText(frame_output, height=5, width=50, font=font_all)
        self.codebook_text.grid(row=3, column=1, padx=5)

    def run_huffman(self):
        data = self.input_text.get("1.0", tk.END).strip()  
        
        if not data:
            messagebox.showwarning("Input Error", "Please enter some text.")
            return

        original_bits = self.print_bits(data)
        encoded_data, codebook = self.huffman_encoding(data)
        decoded_data = self.huffman_decoding(encoded_data, codebook)

        self.original_text.delete(1.0, tk.END)
        self.original_text.insert(tk.END, original_bits)

        self.encoded_text.delete(1.0, tk.END)
        self.encoded_text.insert(tk.END, encoded_data)

        self.decoded_text.delete(1.0, tk.END)
        self.decoded_text.insert(tk.END, decoded_data)

        self.codebook_text.delete(1.0, tk.END)
        self.codebook_text.insert(tk.END, str(codebook))

    @staticmethod
    def print_bits(data):
        return ' '.join(format(ord(char), '08b') for char in data)

    @staticmethod
    def huffman_encoding(data):
        if not data:
            return "", {}

        frequencies = Counter(data)
        root = HuffmanApp.build_huffman_tree(frequencies)
        codebook = HuffmanApp.generate_codes(root)
        encoded_data = ''.join(codebook[char] for char in data)

        return encoded_data, codebook

    @staticmethod
    def huffman_decoding(encoded_data, codebook):
        reverse_codebook = {v: k for k, v in codebook.items()}
        decoded_data = ""
        current_code = ""

        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_codebook:
                decoded_data += reverse_codebook[current_code]
                current_code = ""

        return decoded_data

    @staticmethod
    def build_huffman_tree(frequencies):
        heap = [Node(char, freq) for char, freq in frequencies.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = Node(freq=left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)

        return heap[0]

    @staticmethod
    def generate_codes(node, prefix="", codebook={}):
        if node:
            if node.char is not None:
                codebook[node.char] = prefix
            HuffmanApp.generate_codes(node.left, prefix + "0", codebook)
            HuffmanApp.generate_codes(node.right, prefix + "1", codebook)
        return codebook

class GraphBFS:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.graph and vertex2 in self.graph:
            self.graph[vertex1].append(vertex2)
            self.graph[vertex2].append(vertex1)

    def bfs_tree(self, start):
        visited = set()
        bfs_tree = defaultdict(list)
        queue = deque([start])
        visited.add(start)

        while queue:
            current = queue.popleft()
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    bfs_tree[current].append(neighbor)
                    queue.append(neighbor)
        return bfs_tree

    def visualize(self, bfs_tree=None):
        G = nx.Graph(self.graph)
        pos = nx.spring_layout(G)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Graph Visualization')
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10, font_weight='bold', ax=axs[0])
        axs[0].set_title("Graph")

        if bfs_tree:
            T = nx.DiGraph(bfs_tree)
            pos_tree = nx.spring_layout(T)
            nx.draw(T, pos_tree, with_labels=True, node_color='lightgreen', edge_color='blue', node_size=1000, font_size=10, font_weight='bold', arrows=True, ax=axs[1])
            axs[1].set_title("BFS Tree")
        plt.show()

class GraphGUIBFS:
    def __init__(self, root):
        self.graph = GraphBFS()

        self.root = root
        self.root.title("Graph GUI - kunal S086")

        self.label = tk.Label(root, text="kunal S086", font=font_all)
        self.label.pack()

        self.label = tk.Label(root, text="Enter a vertex:", font=font_all)
        self.label.pack()

        self.vertex_entry = tk.Entry(root, font=font_all, width=width_all)
        self.vertex_entry.pack()

        self.add_vertex_button = tk.Button(root, text="Add Vertex", command=self.add_vertex, font=font_all, width=width_all, bg="#28a745")
        self.add_vertex_button.pack()

        self.label2 = tk.Label(root, text="Enter edge (vertex1 vertex2):", font=font_all)
        self.label2.pack()

        self.edge_entry = tk.Entry(root, font=font_all, width=width_all)
        self.edge_entry.pack()

        self.add_edge_button = tk.Button(root, text="Add Edge", command=self.add_edge, font=font_all, width=width_all, bg="#28a745")
        self.add_edge_button.pack()

        self.bfs_label = tk.Label(root, text="Enter start vertex for BFS:", font=font_all)
        self.bfs_label.pack()

        self.bfs_entry = tk.Entry(root, font=font_all, width=width_all)
        self.bfs_entry.pack()

        self.bfs_button = tk.Button(root, text="Run BFS", command=self.run_bfs, font=font_all, width=width_all, bg=bg_button)
        self.bfs_button.pack()

        self.result_label = tk.Label(root, text="", font=font_all, width=50)
        self.result_label.pack()

    def add_vertex(self):
        vertex = self.vertex_entry.get()
        if vertex:
            self.graph.add_vertex(vertex)
            messagebox.showinfo("Success", f"Added vertex: {vertex}")
            self.vertex_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Please enter a valid vertex.")

    def add_edge(self):
        edge = self.edge_entry.get()
        vertices = edge.split()
        if len(vertices) == 2:
            vertex1, vertex2 = vertices
            self.graph.add_edge(vertex1, vertex2)
            messagebox.showinfo("Success", f"Added edge: {vertex1} - {vertex2}")
            self.edge_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Please enter a valid edge in the format 'vertex1 vertex2'.")

    def run_bfs(self):
        start_vertex = self.bfs_entry.get()
        if start_vertex:
            bfs_tree = self.graph.bfs_tree(start_vertex)
            self.result_label.config(text=f"BFS from {start_vertex}: {dict(bfs_tree)}")
            self.show_graph(bfs_tree)
        else:
            messagebox.showerror("Error", "Please enter a valid start vertex for BFS.")

    def show_graph(self, bfs_tree=None):
        new_window = tk.Toplevel(self.root)
        new_window.title("Graph Visualization")

        figure = plt.Figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=new_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.graph.visualize(bfs_tree)
        canvas.draw()

class GraphDFS:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.graph and vertex2 in self.graph:
            self.graph[vertex1].append(vertex2)
            self.graph[vertex2].append(vertex1)

    def dfs_tree(self, start):
        visited = set()
        dfs_tree = defaultdict(list)

        def dfs(v):
            visited.add(v)
            for neighbor in self.graph[v]:
                if neighbor not in visited:
                    dfs_tree[v].append(neighbor)
                    dfs(neighbor)

        dfs(start)
        return dfs_tree

    def visualize(self, dfs_tree=None):
        G = nx.Graph(self.graph)
        pos = nx.spring_layout(G)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Graph Visualization')

        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10, font_weight='bold', ax=axs[0])
        axs[0].set_title("Graph")

        if dfs_tree:
            T = nx.DiGraph(dfs_tree)
            pos_tree = nx.spring_layout(T)
            nx.draw(T, pos_tree, with_labels=True, node_color='lightgreen', edge_color='blue', node_size=1000, font_size=10, font_weight='bold', arrows=True, ax=axs[1])
            axs[1].set_title("DFS Tree")

        plt.show()

class GraphGUIDFS:
    def __init__(self, root):
        self.graph = GraphDFS()

        self.root = root
        self.root.title("Graph GUI - DFS Visualization")

        self.label = tk.Label(root, text="Kunal joshi S086", font=font_all)
        self.label.pack()

        self.label = tk.Label(root, text="Enter a vertex:", font=font_all)
        self.label.pack()

        self.vertex_entry = tk.Entry(root, font=font_all, width=width_all)
        self.vertex_entry.pack()

        self.add_vertex_button = tk.Button(root, text="Add Vertex", command=self.add_vertex, font=font_all, width=width_all, bg="#28a745")
        self.add_vertex_button.pack()

        self.label2 = tk.Label(root, text="Enter edge (vertex1 vertex2):", font=font_all)
        self.label2.pack()

        self.edge_entry = tk.Entry(root, font=font_all, width=width_all)
        self.edge_entry.pack()

        self.add_edge_button = tk.Button(root, text="Add Edge", command=self.add_edge, font=font_all, width=width_all, bg="#28a745")
        self.add_edge_button.pack()

        self.dfs_label = tk.Label(root, text="Enter start vertex for DFS:", font=font_all)
        self.dfs_label.pack()

        self.dfs_entry = tk.Entry(root, font=font_all, width=width_all)
        self.dfs_entry.pack()

        self.dfs_button = tk.Button(root, text="Run DFS", command=self.run_dfs, font=font_all, width=width_all, bg=bg_button)
        self.dfs_button.pack()

        self.result_label = tk.Label(root, text="", font=font_all, width=50)
        self.result_label.pack()

    def add_vertex(self):
        vertex = self.vertex_entry.get()
        if vertex:
            self.graph.add_vertex(vertex)
            messagebox.showinfo("Success", f"Added vertex: {vertex}")
            self.vertex_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Please enter a valid vertex.")

    def add_edge(self):
        edge = self.edge_entry.get()
        vertices = edge.split()
        if len(vertices) == 2:
            vertex1, vertex2 = vertices
            self.graph.add_edge(vertex1, vertex2)
            messagebox.showinfo("Success", f"Added edge: {vertex1} - {vertex2}")
            self.edge_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Please enter a valid edge in the format 'vertex1 vertex2'.")

    def run_dfs(self):
        start_vertex = self.dfs_entry.get()
        if start_vertex:
            dfs_tree = self.graph.dfs_tree(start_vertex)
            self.result_label.config(text=f"DFS from {start_vertex}: {dict(dfs_tree)}")
            self.show_graph(dfs_tree)
        else:
            messagebox.showerror("Error", "Please enter a valid start vertex for DFS.")

    def show_graph(self, dfs_tree=None):
        new_window = tk.Toplevel(self.root)
        new_window.title("Graph Visualization")

        figure = plt.Figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=new_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.graph.visualize(dfs_tree)

        canvas.draw()

class HashTableWC:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]  

    def _hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash_function(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)  
                return f"Updated: ({key}, {value}) at index {index}"

        self.table[index].append((key, value))
        return f"Inserted: ({key}, {value}) at index {index}"

    def delete(self, key):
        index = self._hash_function(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return f"Deleted key: {key} from index {index}"
        return f"Key: {key} not found for deletion."

    def traverse(self):
        result = []
        for index, chain in enumerate(self.table):
            if chain:
                chain_str = ', '.join([f"({k}, {v})" for k, v in chain])
                result.append(f"Index {index}: {chain_str}")
            else:
                result.append(f"Index {index}: Empty")
        return "\n".join(result)

class HashTableGUIWC:
    def __init__(self, root):
        self.root = root
        self.root.title("Hash Table with Chaining - GUI")
        self.hash_table = None

        # Colors
        self.bg_color = "#f0f0f5"  # Light grey
        self.label_color = "#333399"  # Dark blue
        self.button_color = "#3366cc"  # Light blue
        self.button_text_color = "#ffffff"  # White
        self.textbox_bg = "#e6e6ff"  # Light lavender
        self.textbox_fg = "#000000"  # Black
        
        # Set background color for the root window
        self.root.configure(bg=self.bg_color)

        # Label for name
        name_label = tk.Label(root, text="Kunal Joshi | S086", bg=self.bg_color, fg=self.label_color, font=("Arial", 16, "bold"))
        name_label.pack()

        # Size frame
        size_frame = tk.Frame(root, bg=self.bg_color)
        size_frame.pack(pady=10)

        tk.Label(size_frame, text="Enter size of the Hash Table:", bg=self.bg_color, fg=self.label_color).grid(row=0, column=0)
        self.size_entry = tk.Entry(size_frame)
        self.size_entry.grid(row=0, column=1)

        tk.Button(size_frame, text="Create Hash Table", command=self.create_hash_table, bg=self.button_color, fg=self.button_text_color).grid(row=0, column=2)

        # Action frame for insert/delete
        action_frame = tk.Frame(root, bg=self.bg_color)
        action_frame.pack(pady=10)

        tk.Label(action_frame, text="Key:", bg=self.bg_color, fg=self.label_color).grid(row=0, column=0)
        self.key_entry = tk.Entry(action_frame)
        self.key_entry.grid(row=0, column=1)

        tk.Label(action_frame, text="Value:", bg=self.bg_color, fg=self.label_color).grid(row=1, column=0)
        self.value_entry = tk.Entry(action_frame)
        self.value_entry.grid(row=1, column=1)

        tk.Button(action_frame, text="Insert", command=self.insert_value, bg=self.button_color, fg=self.button_text_color).grid(row=0, column=2)
        tk.Button(action_frame, text="Delete", command=self.delete_value, bg=self.button_color, fg=self.button_text_color).grid(row=1, column=2)

        # Traverse frame
        traverse_frame = tk.Frame(root, bg=self.bg_color)
        traverse_frame.pack(pady=10)

        tk.Button(traverse_frame, text="Traverse", command=self.traverse_table, bg=self.button_color, fg=self.button_text_color).pack()

        # Result text area
        self.result_text = tk.Text(traverse_frame, height=10, width=50, bg=self.textbox_bg, fg=self.textbox_fg)
        self.result_text.pack()

    def create_hash_table(self):
        size = self.size_entry.get()
        if size.isdigit():
            self.hash_table = HashTableWC(int(size))
            messagebox.showinfo("Success", "Hash Table created successfully.")
        else:
            messagebox.showerror("Error", "Please enter a valid size.")

    def insert_value(self):
        if not self.hash_table:
            messagebox.showerror("Error", "Please create the hash table first.")
            return

        key = self.key_entry.get()
        value = self.value_entry.get()

        if key and value:
            result = self.hash_table.insert(key, value)
            messagebox.showinfo("Insert Result", result)
        else:
            messagebox.showerror("Error", "Key and Value cannot be empty.")

    def delete_value(self):
        if not self.hash_table:
            messagebox.showerror("Error", "Please create the hash table first.")
            return
        key = self.key_entry.get()

        if key:
            result = self.hash_table.delete(key)
            messagebox.showinfo("Delete Result", result)
        else:
            messagebox.showerror("Error", "Key cannot be empty.")

    def traverse_table(self):
        if not self.hash_table:
            messagebox.showerror("Error", "Please create the hash table first.")
            return

        result = self.hash_table.traverse()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

width_all = 20
font_all = ("Helvetica", 15)
color = "#007bff"

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Menu")
        self.root.state("zoomed")
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Select an Application", font=("Arial", 24))
        self.label.pack(pady=20)

        self.label = tk.Label(self.root, text="Kunal Joshi S086", font=("Arial", 24))
        self.label.pack(pady=20)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill="y")

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill="y")

        self.linked_list_button_singly = tk.Button(self.left_frame, text="Linked List GUI Singly", command=self.open_linked_list_guiSingly, font=font_all, width=width_all, bg=color)
        self.linked_list_button_singly.pack(pady=10)

        self.linked_list_button_doubly = tk.Button(self.left_frame, text="Linked List GUI Doubly", command=self.open_linked_list_guiDoubly, font=font_all, width=width_all, bg=color)
        self.linked_list_button_doubly.pack(pady=10)

        self.priority_queue_button = tk.Button(self.left_frame, text="Priority Queue GUI", command=self.open_priority_queue_gui, font=font_all, width=width_all, bg=color)
        self.priority_queue_button.pack(pady=10)

        self.tsp_button = tk.Button(self.left_frame, text="TSP Application", command=self.open_tsp_app, font=font_all, width=width_all, bg=color)
        self.tsp_button.pack(pady=10)

        self.binary_tree_button = tk.Button(self.left_frame, text="Binary Tree GUI", command=self.open_binary_tree_gui, font=font_all, width=width_all, bg=color)
        self.binary_tree_button.pack(pady=10)

        self.huffman_button = tk.Button(self.left_frame, text="Huffman Encoding GUI", command=self.open_huffman_app, font=font_all, width=width_all, bg=color)
        self.huffman_button.pack(pady=10)

        self.hash_table_button_wtc = tk.Button(self.right_frame, text="Hash Table WTC GUI", command=self.open_hash_table_guiWTC, font=font_all, width=width_all, bg=color)
        self.hash_table_button_wtc.pack(pady=10)

        self.hash_table_button_wc = tk.Button(self.right_frame, text="Hash Table WC GUI", command=self.open_hash_table_guiWC, font=font_all, width=width_all, bg=color)
        self.hash_table_button_wc.pack(pady=10)

        self.family_tree_button = tk.Button(self.right_frame, text="Family Tree GUI", command=self.open_family_tree_gui, font=font_all, width=width_all, bg=color)
        self.family_tree_button.pack(pady=10)

        self.queue_button = tk.Button(self.right_frame, text="Queue GUI", command=self.open_queue_gui, font=font_all, width=width_all, bg=color)
        self.queue_button.pack(pady=10)

        self.graph_button_bfs = tk.Button(self.right_frame, text="Graph GUI BFS", command=self.open_graph_guiBFS, font=font_all, width=width_all, bg=color)
        self.graph_button_bfs.pack(pady=10)

        self.graph_button_dfs = tk.Button(self.right_frame, text="Graph GUI DFS", command=self.open_graph_guiDFS, font=font_all, width=width_all, bg=color)
        self.graph_button_dfs.pack(pady=10)

    def open_linked_list_guiSingly(self):
        root1 = tk.Toplevel(self.root)
        root1.state("zoomed")
        app1 = LinkedListGUISingly(root1)

    def open_linked_list_guiDoubly(self):
        root2 = tk.Toplevel(self.root)
        root2.state("zoomed")
        app2 = LinkedListGUIDoubly(root2)

    def open_priority_queue_gui(self):
        root3 = tk.Toplevel(self.root)
        root3.state("zoomed")
        app3 = PriorityQueueGUI(root3)

    def open_tsp_app(self):
        root4 = tk.Toplevel(self.root)
        root4.state("zoomed")
        app4 = TSPApp(root4)

    def open_hash_table_guiWTC(self):
        root5 = tk.Toplevel(self.root)
        root5.state("zoomed")
        app5 = HashTableGUIWTC(root5)

    def open_hash_table_guiWC(self):
        root6 = tk.Toplevel(self.root)
        root6.state("zoomed")
        app6 = HashTableGUIWC(root6)  

    def open_family_tree_gui(self):
        root7 = tk.Toplevel(self.root)
        root7.state("zoomed")
        app7 = FamilyTreeGUI(root7)

    def open_queue_gui(self):
        root8 = tk.Toplevel(self.root)
        root8.state("zoomed")
        app8 = QueueGUI(root8)

    def open_graph_guiBFS(self):
        root9 = tk.Toplevel(self.root)
        root9.state("zoomed")
        app9 = GraphGUIBFS(root9)

    def open_graph_guiDFS(self):
        root10 = tk.Toplevel(self.root)
        root10.state("zoomed")
        app10 = GraphGUIDFS(root10)

    def open_binary_tree_gui(self):
        root11 = tk.Toplevel(self.root)
        root11.state("zoomed")
        app11 = BinaryTreeGUI(root11)

    def open_huffman_app(self):
        root12 = tk.Toplevel(self.root)
        root12.state("zoomed")
        app12 = HuffmanApp(root12)

if __name__ == "__main__":
    main_root = tk.Tk()
    app = MainMenu(main_root)
    main_root.mainloop()