# Packages
import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
from networkx.algorithms import bipartite
from community import community_louvain


# Fonction pour calculer la densité d'un graphe
def calculate_graph_density(similarity_matrix):
    # Conversion de la matrice de similarité en graphe non orienté
    graph = nx.from_numpy_array(similarity_matrix)
    # Calcul de la densité du graphe
    density = nx.density(graph)

    return density


# Fonction pour comparer la densité des graphes et avoir un aperçu
def visualize_and_compare_densities(similarity_matrices, similarity_labels):
    densities = [calculate_graph_density(similarity_matrix) for similarity_matrix in similarity_matrices]

    # Comparaison des densités
    max_density = max(densities)
    max_density_label = similarity_labels[densities.index(max_density)]
    density_df = pd.DataFrame({"Similarité": similarity_labels,
                               "Densité du graphe": densities})

    # Affichage des graphiques individuels de densité
    print("Aperçu des graphes des matrices de similarité pour"
          f" mise en perspective avec les calculs de densités: ")
    fig, axs = plt.subplots(1, len(similarity_matrices), figsize=(12, 4))
    for i, similarity_matrix in enumerate(similarity_matrices):
        graph = nx.from_numpy_array(similarity_matrix)
        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, pos=pos, with_labels=False,
                         node_size=100, alpha=0.7, ax=axs[i])
        axs[i].set_title(f"Graphe de Similarité '{similarity_labels[i]}'")
        axs[i].axis('off')

    # Affichage du graphique de comparaison des densités
    plt.figure(figsize=(5, 2))
    plt.bar(similarity_labels, densities)
    plt.xlabel("Similarité")
    plt.ylabel("Densité du graphe")
    plt.title("Comparaison des Densités des Graphes de Similarités")

    # Vérification de la différence de densité significative
    if len(densities) >= 2:
        diff_density = max_density - min(densities)
        if diff_density > 0.1:
            plt.text(0.5, -0.2, "Différence de densité significative",
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes)
    plt.show()
    print(f"\nLe graphe de similarité '{max_density_label}' a la plus grande densité de graphe.")

    print("Tableau de valeurs des densités calculées : ")
    return density_df


# Fonction d'aperçu de la fonction de similarité finale
def visualize_similarity_graph(graph_sim):
    # Détection des communautés dans le graphe avec l'algorithme de Louvain
    partition = community_louvain.best_partition(graph_sim)

    # Couleurs pour chaque communauté
    colors = [partition[node] for node in graph_sim.nodes()]

    # Affichage du graphe avec les points colorés selon la communauté
    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(graph_sim, seed=42)
    nx.draw_networkx(graph_sim, pos=pos, with_labels=False,
                     node_color=colors, node_size=100, alpha=0.7)
    plt.title(f'Aperçu du graphe non orienté généré à partir de la matrice '
              f'finale de similarité (avec détection des communautés)')
    plt.axis('off')
    plt.show()


# fonction pour l'identification des utilisateurs leaders ayant le maximum de voisins
def find_top_users_leaders(graph_sim, num_leaders=5):
    leaders = sorted(graph_sim.degree,
                     key=lambda x: x[1], reverse=True)[:num_leaders]

    return leaders


# fonction pour l'aperçu visuel des utilisateurs leaders
def visualize_leader_subgraph(graph, num_leaders=5):
    # Identification des utilisateurs leaders avec leurs voisins
    leaders = find_top_users_leaders(graph, num_leaders=num_leaders)
    # Affichage des utilisateurs leaders avec leurs voisins
    print("Les 5 utilisateurs leaders ayant le maximum de voisins :")
    leaders_df = pd.DataFrame(leaders, columns=["Utilisateur", "Nb Voisins"])
    print(leaders_df)
    # Création d'un sous-graphe avec les utilisateurs leaders et leurs voisins
    leader_nodes = [leader[0] for leader in leaders]
    subgraph_leaders = graph.subgraph(leader_nodes)
    # Nombre de voisins pour chaque nœud
    num_neighbors = {node: graph.degree[node] for node in leader_nodes}
    # Affichage du sous-graphe des utilisateurs leaders avec les nombres de voisins
    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(subgraph_leaders, seed=42)
    nx.draw_networkx(subgraph_leaders, pos=pos, with_labels=True, node_size=200, alpha=0.7)
    plt.title(f"Sous-graphe des Utilisateurs Leaders ({num_neighbors[leader_nodes[0]]} voisins)")
    plt.axis('off')
    plt.show()

    # fonction pour identifier les utilisateurs connecteurs
    def find_top_users_connectors(graph, num_connectors=5):
        # Identification des utilisateurs connecteurs reliant le maximum de communautés
        connectors = sorted(
            nx.betweenness_centrality(graph).items(),
            key=lambda x: x[1], reverse=True)[:num_connectors]

        return connectors


    # fonction pour l'aperçu visuel des utilisateurs  connecteurs
    def visualize_connector_subgraph(graph, num_connectors=5):
        # Identification des utilisateurs connecteurs reliant le maximum de communautés
        connectors = find_top_users_connectors(graph, num_connectors=num_connectors)
        # Affichage des utilisateurs leaders avec leurs voisins
        print("Les 5 utilisateurs connecteurs reliant le maximum de communautés :")
        connectors_df = pd.DataFrame(connectors, columns=["Utilisateurs", "Coef de centralité"])
        print(connectors_df)
        # Création d'un sous-graphe avec les utilisateurs connecteurs et leurs voisins
        connector_nodes = [connector[0] for connector in connectors]
        subgraph_connectors = graph.subgraph(connector_nodes)
        # Calcul des coefficients de centralité des utilisateurs connecteurs
        centrality = {nodes: centrality for nodes, centrality in connectors}
        # Affichage du sous-graphe des utilisateurs connecteurs avec les arêtes
        plt.figure(figsize=(4, 4))
        pos = nx.spring_layout(subgraph_connectors, seed=42)
        # Affichage des arêtes du sous-graphe
        nx.draw_networkx_edges(subgraph_connectors, pos, width=1.0, alpha=0.7)
        # Affichage des nœuds avec l'étiquette de l'identifiant utilisateur
        nx.draw_networkx_nodes(subgraph_connectors, pos, node_size=300, alpha=0.8)
        nx.draw_networkx_labels(subgraph_connectors, pos, font_size=8, font_color='black')
        # Formatage et affichage des coefficients de centralité au-dessus des nœuds
        for node, (x, y) in pos.items():
            # Formatage en notation scientifique
            centrality_value = f"Centrality: {centrality[node]:.2e}"
            plt.text(x, y + 0.05, centrality_value, ha='center', fontsize=8)
        plt.title(f"Sous-graphe des Utilisateurs Connecteurs ({num_connectors} connecteurs)")
        plt.axis('off')
        plt.show()

        def find_top_information_flow(graph, num_relations=5):
            # Calculer les coefficients de centralité pour toutes les arêtes du graphe
            edge_centralities = nx.edge_betweenness_centrality(graph)

            # Trier les arêtes par coefficient de centralité décroissant
            sorted_edges = sorted(edge_centralities.items(), key=lambda x: x[1], reverse=True)

            # Récupérer les 5 meilleures relations
            top_relations = sorted_edges[:num_relations]

            return top_relations

        def visualize_information_flow_subgraph(graph, num_relations=5):
            # Sous-graphe des 5 meilleures relations d'information
            subgraph = nx.DiGraph()

            # Identifier les 5 meilleures relations d'information
            top_relations = find_top_information_flow(graph, num_relations=num_relations)

            # Affichage des utilisateurs leaders avec leurs voisins
            print("Les 5 meilleures relations d'information :")
            top_relations_df = pd.DataFrame(top_relations, columns=["Relations Utilisateurs", "Coef de centralité"])
            print(top_relations_df)

            # Ajouter les arêtes et les nœuds correspondants aux meilleures relations
            for edge, centrality in top_relations:
                source, target = edge
                if source in graph.nodes() and target in graph.nodes():
                    subgraph.add_edge(source, target)
                    subgraph.edges[source, target]['centrality'] = centrality

            # Détecter les communautés dans le sous-graphe
            communities = nx.algorithms.community.greedy_modularity_communities(subgraph)
            community_colors = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_colors[node] = i

            # Affichage du sous-graphe avec étiquettes, couleurs de communauté et coefficient de centralité
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(subgraph)

            # Dessiner les nœuds avec les étiquettes et les couleurs de communauté
            node_labels = nx.get_node_attributes(subgraph, 'userId')
            node_colors = [community_colors[node] for node in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos=pos,
                                   node_color=node_colors, node_size=300, alpha=0.8)
            nx.draw_networkx_labels(subgraph, pos=pos,
                                    labels=node_labels, font_size=8, font_color='black')

            # Dessiner les arêtes avec les couleurs de communauté correspondantes et afficher le coefficient de centralité
            for edge in subgraph.edges():
                centrality = subgraph.edges[edge]['centrality']
                edge_color = community_colors[edge[0]]
                nx.draw_networkx_edges(subgraph, pos=pos,
                                       edgelist=[edge], edge_color=str(edge_color), alpha=0.7)
                if 'userId' in subgraph.nodes[edge[0]]:
                    source_user = subgraph.nodes[edge[0]]['userId']
                else:
                    source_user = edge[0]
                if 'userId' in subgraph.nodes[edge[1]]:
                    target_user = subgraph.nodes[edge[1]]['userId']
                else:
                    target_user = edge[1]
                plt.text((pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2,
                         f"Centrality: {centrality:.2e}\n{source_user} -> {target_user}", ha='center', fontsize=8)

            plt.title("Sous-graphe des Meilleures Relations d'Information")
            plt.axis('off')
            plt.show()


