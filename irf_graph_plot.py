import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.stats import pearsonr
from irf import irf_utils
from irf.ensemble import RandomForestRegressorWithWeights
#import netgraph
import matplotlib.cm as cmx
from sklearn.linear_model import LinearRegression
import math


def pearsonr_pval(x, y):
    '''
    Parameters:
        x, y: lists of same length

    Returns:
        p-value of pearson correlation between x and y

    '''
    return pearsonr(x, y)[1]


def irf_adj(filename, conc, type):
    '''
    Parameters:
        filename (str): filename
        conc (str): k+ concentration
        type (str): type of GCR exposure

    Returns:
        iRF network of neurotransmitters

    '''
    df = pd.read_excel(filename)
    names = df.columns.values.tolist()
    NT = names[2:7]
    df = df[df['Concentration'] == conc]
    df = df.reset_index()
    classes = df['Target Class']
    l = len(classes)
    features = np.zeros((l, 5))
    # cla = list(set(classes))
    for i in range(5):
        features[:, i] = df[NT[i]]

    ind = np.linspace(0, l-1, l)
    np.random.seed(203)
    np.random.shuffle(ind)
    ind = ind.astype('int')
    m = int(l//5)
    trn = features[ind[:(l-m)], :]
    tst = features[ind[(l-m):], :]
    classes_trn = classes[ind[:(l-m)]]
    classes_tst = classes[ind[(l-m):]]
    idx = np.where((classes_trn == type) == True)[0]
    idxx = np.where((classes_tst == type) == True)[0]
    imp_mat = np.zeros((5, 5))
    sign_mat = np.zeros((5, 5))
    for i in range(5):
        cl = list({0, 1, 2, 3, 4} - {i})
        X_train = trn[:, cl]
        X_train = X_train[idx, :]
        y_train = trn[:, i]
        y_train = y_train[idx]
        X_test = tst[:, cl]
        X_test = X_test[idxx, :]
        y_test = tst[:, i]
        y_test = y_test[idxx]
        _, all_K_iter_rf_data, _, _, _ = irf_utils.run_iRF(X_train=X_train,
                                                           X_test=X_test,
                                                           y_train=y_train,
                                                           y_test=y_test,
                                                           K=5,
                                                           rf=RandomForestRegressorWithWeights(n_estimators=20,
                                                                                               random_state=2018),
                                                           B=30,
                                                           random_state_classifier=2018,
                                                           propn_n_samples=.2,
                                                           bin_class_type=None,
                                                           M=20,
                                                           max_depth=5,
                                                           noisy_split=False,
                                                           num_splits=2,)

        reg = LinearRegression().fit(X_train, y_train)
        feature_importances = all_K_iter_rf_data['rf_iter5']['feature_importances']
        imp_mat[i, cl] = feature_importances
        sign_mat[i, cl] = np.sign(reg.coef_)
    adjacency_matrix = np.transpose(np.matrix(imp_mat))
    sign_matrix = np.transpose(np.matrix(sign_mat))
    return adjacency_matrix, sign_matrix


def main():
    filename = '/home/schatterjee/Desktop/iRF/Neurotransmitters_conc.xlsx'
    cla = ['Control', 'Acute', 'Chronic']
    concs = ['4 mM', '30 mM', '60 mM', '120 mM']
    # t = ['DA','5-HT','NE','GLU','GABA']
    dct_label = {'Dopamine': 'DA', '5-HT': '5-HT',
                 'Norepinephrine': 'NE', 'Glutamate': 'GLU', 'GABA': 'GABA'}
    df = pd.read_excel(filename)
    names = df.columns.values.tolist()
    NT = names[2:7]
    int2label = {}
    for i in range(5):
        int2label[i] = dct_label[NT[i]]

    pos = {'DA': (-1.5, -0.5), '5-HT': (-1.5, 0.5), 'NE': (2.5, -0.5),
           'GLU': (2.5, 0.5), 'GABA': (0.5, 1.25)}

    adj_dict = {}
    adj_s_dict = {}
    th = 0.2
    for i in concs:
        adj_mat = np.zeros((5, 5, 3))
        k = 0
        for j in cla:
            mat, mat_s = irf_adj(filename, i, j)
            mat[mat <= th] = 0
            mat_s[mat <= th] = 0
            adj_dict[i, j] = mat
            adj_s_dict[i, j] = mat_s
            adj_mat[:, :, k] = adj_dict[i, j]
            k = k + 1
        adj_dict[i, 'Common'] = adj_mat.min(axis=2)
        adj_dict[i, 'Change'] = adj_mat.max(axis=2) - adj_mat.min(axis=2)
        adj_dict[i, 'Acute vs. Control'] = np.abs(adj_mat[:, :, 1]
                                                  - adj_mat[:, :, 0])
        adj_dict[i, 'Chronic vs. Control'] = np.abs(adj_mat[:, :, 2]
                                                    - adj_mat[:, :, 0])
        adj_dict[i, 'Chronic vs. Acute'] = np.abs(adj_mat[:, :, 2]
                                                  - adj_mat[:, :, 1])

    # cla_f = cla + ['Common','Change']
    # cla_f = cla #+ ['Acute vs. Control','Chronic vs. Control','Chronic vs. Acute']
    # cla_f = ['Acute vs. Control','Chronic vs. Control','Chronic vs. Acute']
    cla_f = ['Control', 'Acute', 'Chronic']
    r_pos = 0.6
    nt_keys = list(pos.keys())
    fig, axn = plt.subplots(len(concs), len(cla_f),
                            figsize=(2.8*len(cla_f), 16)) # , sharex=True, sharey=True)
    for i, ax in enumerate(axn.flat):
        xx = i % len(cla_f)
        yy = i//len(cla_f)
        adjacency_matrix = adj_dict[concs[yy], cla_f[xx]]
        s_matrix = adj_s_dict[concs[yy], cla_f[xx]]
        print(concs[yy], cla_f[xx])
        print(s_matrix)
        # th = 0.2
        # if xx<3:
        #     adjacency_matrix[adjacency_matrix<th] = 0
        G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)
        G = nx.relabel_nodes(G, int2label)
        weights = nx.get_edge_attributes(G, "weight")

        # getting edges 
        edges = G.edges()
        
        # weight and color of edges 
        scaling_factor = 3 # to emphasise differences 
        alphas = [weights[edge] * scaling_factor for edge in edges]

        values = alphas
        # These values could be seen as dummy edge weights

        cNorm = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.cm.binary)
        colorList = []

        for n in range(len(values)):
            colorVal = scalarMap.to_rgba(values[n])
            colorList.append(colorVal)

        nx.draw(G, ax=ax, with_labels=True, pos=pos,
                node_size=600, font_size=8, arrowsize=8,
                node_color=['r', '#83c995', '#17becf', 'yellow', 'orange'],
                width=alphas, edge_color=colorList)# connectionstyle="arc3,rad=0.06",
        ax.set_xlim([-3, 4])
        ax.set_ylim([-1, 1.6])
        # nx.draw_networkx_edge_labels(G, pos=pos)
        if i == len(concs)*len(cla_f)-2:
            ax.text(-4.5, -1.1,
                    'Edges shown with feature importance > '+str(th))
        # if i==len(concs)*len(cla_f)-4: ax.text(-2,-1.1,'Edges shown with feature importance > '+str(th))
        if xx == 0:
            ax.text(-3, -0.25, concs[yy],
                    rotation='vertical')
        if yy == 0:
            ax.set_title(cla_f[xx])

        delta = 0.05
        for jj in range(5):
            to_node = pos[nt_keys[jj]]
            for ii in range(5):
                if s_matrix[ii, jj] != 0:
                    from_node = pos[nt_keys[ii]]
                    rad = math.atan2(from_node[1]-to_node[1],
                                     from_node[0]-to_node[0])
                    coord_x = to_node[0] - delta + r_pos*math.cos(rad)
                    coord_y = to_node[1] - delta + r_pos*math.sin(rad)
                    # ax.text(to_node[0] - delta, to_node[1] - delta, 'o')
                    if s_matrix[ii, jj] < 0:
                        ax.text(coord_x, coord_y, '-', color='b', fontsize=15)
                    else:
                        ax.text(coord_x, coord_y, '+', color='r', fontsize=12)

    plt.show()
    # fig.savefig('/home/schatterjee/Desktop/iRF/irf_graph_pairwise_change_'+str(th)+'.eps', format='eps', dpi=100)
    # fig.savefig('/home/schatterjee/Desktop/iRF/irf_graph_pairwise_change_'+str(th)+'.pdf', dpi=300)
    fig.savefig('/home/schatterjee/Desktop/iRF/irf_graphs_'+str(th)+'.eps',
                dpi=300, format='eps')


if __name__ == '__main__':
    main()
