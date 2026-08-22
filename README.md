# Data Structures and Algorithms GUI

A Python-based **Data Structures and Algorithms visualization project** with interactive graphical interfaces built using **Tkinter**.

The project provides separate GUI applications for practicing and visualizing fundamental data structures, algorithms, graph traversal techniques, hashing, compression, and the Travelling Salesman Problem.

## 👨‍💻 Student

**Kunal Joshi | S086**

---

# 📌 Features

The project contains interactive implementations of:

* Travelling Salesman Problem
* Brute-Force TSP
* Greedy TSP
* Queue
* Priority Queue
* Hash Table without collision handling
* Hash Table with Chaining
* Singly Linked List
* Doubly Linked List
* Binary Search Tree
* Huffman Encoding and Decoding
* Breadth-First Search (BFS)
* Depth-First Search (DFS)
* Family Tree
* Graph Visualization

The application provides a main menu from which the individual GUI modules can be opened.

---

# 🖥️ Graphical User Interface

The project uses **Tkinter** to provide interactive windows for each data structure and algorithm.

Users can enter data through input fields and perform operations using buttons rather than interacting only through the command line.

The application also uses graphical visualization for structures such as graphs, trees, and TSP routes.

---

# 🧩 Modules

## 1. Travelling Salesman Problem

The project demonstrates two approaches for solving the Travelling Salesman Problem:

### Brute-Force Approach

The brute-force implementation evaluates possible routes and selects the route with the minimum total distance.

### Greedy Approach

The greedy implementation starts from a selected location and repeatedly chooses the nearest unvisited location.

Both approaches display:

* Route
* Total distance
* Graphical route visualization

The TSP interface allows the user to enter the number of places, place names, and distance matrix.

---

## 2. Queue

A basic Queue implementation is provided with the following operations:

* Enqueue
* Dequeue
* Peek
* Check if empty
* Get size
* Traverse

The GUI provides buttons for each operation and displays the resulting queue state.

---

## 3. Priority Queue

The project includes Priority Queue implementations that allow elements to be associated with priorities.

Supported operations include:

* Enqueue
* Dequeue
* Peek
* Check if empty
* Get size
* Traverse

The priority queue GUI accepts items along with their priority values.

---

# 🔑 4. Hash Tables

Two different hashing implementations are included.

## Hash Table Without Collision Handling

This implementation demonstrates basic hashing where a key is mapped to an index.

Operations include:

* Insert
* Delete
* Traverse

If two keys produce the same index, the implementation reports a collision rather than resolving it.

## Hash Table With Chaining

The second implementation handles collisions using **separate chaining**.

Multiple key-value pairs can therefore exist at the same hash index.

Operations include:

* Insert
* Update
* Delete
* Traverse

The chaining implementation stores multiple entries within a bucket when required.

---

# 🌳 5. Family Tree

The Family Tree module allows users to create parent-child relationships.

Features include:

* Add parent-child relationship
* Create family members
* Identify the root
* Generate a graphical family tree

The tree structure is represented using nodes and displayed graphically using NetworkX and Matplotlib.

---

# 🔗 6. Singly Linked List

The Singly Linked List module demonstrates:

* Append
* Prepend
* Delete
* Traverse / Display

Users can enter multiple values and interact with the linked list through the GUI.

---

# 🔗 7. Doubly Linked List

The Doubly Linked List contains nodes with connections in both directions.

Supported operations include:

* Append
* Prepend
* Delete
* Display

The implementation maintains both forward and backward links between nodes.

---

# 🌲 8. Binary Search Tree

The Binary Search Tree module supports:

* Insert
* Delete
* Inorder Traversal
* Preorder Traversal
* Postorder Traversal

The current tree is also represented graphically in the GUI.

### Traversals

**Inorder**

Produces the values in sorted order for a Binary Search Tree.

**Preorder**

Visits the root before its subtrees.

**Postorder**

Visits the subtrees before the root.

The GUI provides separate controls for all three traversal methods.

---

# 🔐 9. Huffman Encoding and Decoding

The project demonstrates **Huffman Coding**, a lossless data compression technique.

The Huffman module allows users to enter text and obtain:

* Original binary representation
* Encoded data
* Decoded data
* Huffman codebook

The implementation builds a Huffman tree based on character frequencies and generates binary codes for the characters.

The application also performs decoding to verify that the encoded data can be converted back to the original text.

---

# 🔍 10. Breadth-First Search — BFS

The BFS module allows users to create a graph and perform Breadth-First Search from a selected starting vertex.

Features include:

* Add vertex
* Add edge
* Select starting vertex
* Run BFS
* Generate BFS tree
* Visualize the graph and BFS tree

The BFS implementation uses a queue-based traversal approach.

---

# 🔎 11. Depth-First Search — DFS

The DFS module provides:

* Add vertex
* Add edge
* Select starting vertex
* Run DFS
* Generate DFS tree
* Visualize the graph and DFS tree

The DFS implementation uses recursive traversal and tracks visited vertices to construct the DFS tree.

---

# 📊 Visualization

Graphical visualization is an important part of the project.

The project uses:

* **NetworkX** for graph representation and drawing
* **Matplotlib** for graphical visualization
* **Tkinter** for embedding visualizations into GUI applications

The TSP module displays routes graphically, while BFS and DFS display both the original graph and the corresponding traversal tree.

---

# 🛠️ Technologies Used

* **Python**
* **Tkinter**
* **NetworkX**
* **Matplotlib**
* **NumPy**
* **AnyTree**

The project imports these libraries for GUI development, graph visualization, numerical operations, tree representation, and algorithm implementation.

---

# 📚 Data Structures Covered

| Data Structure           | Operations / Concepts                          |
| ------------------------ | ---------------------------------------------- |
| Queue                    | Enqueue, Dequeue, Peek, Size, Traverse         |
| Priority Queue           | Priority-based insertion and removal           |
| Singly Linked List       | Append, Prepend, Delete, Traverse              |
| Doubly Linked List       | Bidirectional linking, Append, Prepend, Delete |
| Hash Table               | Hashing, Insert, Delete, Traverse              |
| Hash Table with Chaining | Collision handling                             |
| Binary Search Tree       | Insert, Delete, Traversals                     |
| Huffman Tree             | Encoding, Decoding, Code Generation            |
| Graph                    | Vertices, Edges, Traversals                    |
| Family Tree              | Parent-Child Relationships                     |

---

# 🧠 Algorithms Covered

| Algorithm                     | Purpose                                                               |
| ----------------------------- | --------------------------------------------------------------------- |
| Brute-Force TSP               | Find an optimal travelling route by evaluating possible routes        |
| Greedy TSP                    | Construct a route by repeatedly selecting the nearest unvisited place |
| BFS                           | Traverse a graph level by level                                       |
| DFS                           | Traverse a graph depth-first                                          |
| Huffman Coding                | Perform lossless text encoding and decoding                           |
| Hashing                       | Map keys to table indices                                             |
| Binary Search Tree Operations | Maintain and traverse ordered tree data                               |

---

# 🖱️ Main Application

The project includes a central GUI menu that provides access to the different modules.

Available modules include:

* Singly Linked List
* Doubly Linked List
* Priority Queue
* TSP
* Hash Table without Chaining
* Hash Table with Chaining
* Family Tree
* Queue
* BFS
* DFS
* Binary Tree
* Huffman Coding

The main application launches the selected module in a separate window.

---

# 🎯 Learning Objectives

This project demonstrates practical understanding of:

* Fundamental data structures
* Algorithm implementation
* Object-oriented programming
* Recursion
* Graph traversal
* Tree traversal
* Hashing techniques
* Collision handling
* Priority-based data structures
* Linked list manipulation
* Lossless compression
* GUI application development
* Graph and tree visualization
* Algorithm comparison

---

# 🚀 Project Highlights

* Interactive GUI-based learning environment
* Multiple data structures in a single application
* Graphical representation of algorithms
* TSP algorithm comparison
* BFS and DFS visualization
* Hash table collision demonstration
* Huffman encoding and decoding
* Binary tree visualization
* User-friendly input and operation controls


# ⚙️ Installation

Install Python 3.x and the required libraries.

Required packages:

* `networkx`
* `matplotlib`
* `numpy`
* `anytree`

Tkinter is generally included with standard Python installations on Windows.

---

# ▶️ Running the Project

1. Clone or download the repository.
2. Install the required dependencies.
3. Run the main Python application.
4. Select the desired data structure or algorithm from the main menu.
5. Enter the required input.
6. Perform operations using the GUI controls.
7. View the results and visualizations.

---

# 📖 Educational Purpose

This project is designed as a practical implementation of **Data Structures and Algorithms** concepts.

Instead of implementing each concept only through terminal-based programs, the project provides interactive graphical interfaces that make it easier to understand how the underlying structures and algorithms behave.

---

# 👨‍💻 Author

**Kunal Joshi007m**
---

## ⭐ Conclusion

This project brings together multiple fundamental **Data Structures and Algorithms** in one interactive Python GUI application.

It demonstrates queues, priority queues, linked lists, hash tables, trees, graphs, graph traversal, Huffman coding, and TSP approaches while providing graphical visualization wherever applicable.

The project serves as a practical reference for understanding and experimenting with core DSA concepts through Python.
