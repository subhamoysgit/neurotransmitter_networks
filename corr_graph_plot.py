import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.stats import pearsonr


def pearsonr_pval(x, y):
    '''
    Parameters:
        x, y: lists of same length

    Returns:
        p-value of pearson correlation between x and y

    '''
    return pearsonr(x, y)[1]


def corr_graph(filename, conc, type):
    '''
    Parameters:
        filename (str): filename
        conc (str): k+ concentration
        type (str): type of GCR exposure

    Returns:
        correlation network of neurotransmitters

    '''
    df = pd.read_excel(filename)
    names = df.columns.values.tolist()
    df = df[df['Concentration'] == conc]
    df = df.reset_index()
    NT = names[2:7]
    classes = df['Target Class']
    data = df[NT][classes == type]
    pval = np.matrix(data.corr(method=pearsonr_pval))
    corr = np.matrix(data.corr())
    adjacency_matrix = np.abs(corr)
    adjacency_matrix[pval >= 0.05] = 0
    adjacency_matrix[np.eye(5) == 1] = 0
    adjacency_matrix[adjacency_matrix < 0.5] = 0  # new addition
    G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.Graph)
    return G


def main():
    filename = '/home/schatterjee/Desktop/iRF/Neurotransmitters_conc.xlsx'
    cla = ['Control', 'Acute', 'Chronic']
    concs = ['4 mM', '30 mM', '60 mM', '120 mM']
    # t = ['DA', '5-HT', 'NE', 'GLU', 'GABA']
    dct_label = {'Dopamine': 'DA', '5-HT': '5-HT', 'Norepinephrine': 'NE',
                 'Glutamate': 'GLU', 'GABA': 'GABA'}
    df = pd.read_excel(filename)
    names = df.columns.values.tolist()
    NT = names[2:7]
    int2label = {}
    for i in range(5):
        int2label[i] = dct_label[NT[i]]
    # pos = {'DA': (-1.5, -0.5), '5-HT': (-1.5, 0), 'NE': (2.5, -0.5), 'GLU': (2.5, 0), 'GABA': (0.5, 0.5)}
    pos = {'DA': (-1.5, -0.5), '5-HT': (-1.5, 0.5), 'NE': (2.5, -0.5),
           'GLU': (2.5, 0.5), 'GABA': (0.5, 1.0)}
    fig, axn = plt.subplots(4, 3, figsize=(8, 16))  # , sharex=True, sharey=True)
    for i, ax in enumerate(axn.flat):
        xx = i % len(cla)
        yy = i//len(cla)
        G = corr_graph(filename, concs[yy], cla[xx])
        G = nx.relabel_nodes(G, int2label)
        weights = nx.get_edge_attributes(G, "weight")

        # getting edges
        edges = G.edges()
        # weight and color of edges
        scaling_factor = 5  # to emphasise differences 
        alphas = [weights[edge] * scaling_factor for edge in edges]
        nx.draw(G, ax=ax, with_labels=True, pos=pos, width=alphas,
                node_size=600, font_size=9,
                node_color=['r', '#83c995', '#17becf', 'yellow', 'orange'])
        ax.set_xlim([-3, 4])
        ax.set_ylim([-1, 1.3])
        # nx.draw_networkx_edge_labels(G, pos=pos)
        if i == 10:
            ax.text(-6, -1.1,
                    'Edges shown with |Pearson Correlation| $\geq$ 0.5'
                    ' and p-value<0.05')
        if xx == 0:
            ax.text(-3, -0.25, concs[yy], rotation='vertical')
        if yy == 0:
            ax.set_title(cla[xx])
    plt.show()
    fig.savefig('/home/schatterjee/Desktop/iRF/correlation_graph.eps',
                dpi=300, format='eps')


if __name__ == '__main__':
    main()
