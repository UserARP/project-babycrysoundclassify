# importing graph from previous chapter but directed version
from graph_directed import Graph
from graph_directed import Node  # see code of previous chapter
from graph_directed import Color


def dfs_visit(g, i):
    i.color = Color.Gray

    temp = i.head
    while(temp != None):
        if(g.heads[temp.index_of_item].color == Color.White):
            dfs_visit(g, g.heads[temp.index_of_item])
        temp = temp.next
    i.color = Color.Black
    print(i.data)


def dfs(g):
    for i in range(0, g.number_of_vertices):
        g.heads[i].color = Color.White

    for i in range(0, g.number_of_vertices):
        if(g.heads[i].color == Color.White):
            dfs_visit(g, g.heads[i])


if __name__ == '__main__':
    g = Graph(7)

    g.add_edge(1, 2)
    g.add_edge(1, 5)
    g.add_edge(1, 3)
    g.add_edge(2, 6)
    g.add_edge(2, 4)
    g.add_edge(5, 4)
    g.add_edge(3, 4)
    g.add_edge(3, 7)

    dfs(g)
