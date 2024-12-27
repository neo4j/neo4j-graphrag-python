import pygraphviz as pgv

# Create a new graph
G = pgv.AGraph()

# Add some nodes and edges
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 1)

# Set the layout
G.layout(prog='dot')

# Save the graph as SVG
G.draw('test.svg')

print("Graph has been created and saved as test.svg")
