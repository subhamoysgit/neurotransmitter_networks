import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns

import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import numpy as np
import matplotlib
from scipy.special import cbrt

font = {'family': 'normal',
        'size': 14}

matplotlib.rc('font', **font)

##### utils
class CubeRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating cube root scale.
    """
 
    name = 'cuberoot'
 
    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)
 
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())
 
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax
 
    class CubeRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform_non_affine(self, a): 
            return cbrt(np.array(a))
 
        def inverted(self):
            return CubeRootScale.InvertedCubeRootTransform()
 
    class InvertedCubeRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform(self, a):
            return np.array(a)**3
 
        def inverted(self):
            return CubeRootScale.SquareRootTransform()
 
    def get_transform(self):
        return self.CubeRootTransform()
mscale.register_scale(CubeRootScale)

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


path = '/home/schatterjee/Desktop/causal_discovery/Neurotransmitters_conc.xlsx'

df = pd.read_excel(path)

target_class = ['Control', 'Acute', 'Chronic']
features = ['Concentration', 'Dopamine', '5-HT',
            'Norepinephrine', 'Glutamate', 'GABA']
ff = ['5-HT', 'NE', 'GLU', 'GABA']

#fig, ax = plt.subplots(1,3,figsize = (15,5))

fig = plt.figure(figsize = (15, 5))

ax1 = fig.add_axes([0.08, 0.1, 0.3, 0.8])
ax2 = fig.add_axes([0.38, 0.1, 0.3, 0.8])
ax3 = fig.add_axes([0.68, 0.1, 0.3, 0.8])
ax = [ax1,ax2, ax3]


v = [0,1]
mat_pval = {}
colors = ['k', 'b', 'r']

df_p_value = {'Target Class': [], 
              '5-HT': [], 'NE': [], 'GLU': [], 
              'GABA': []}

for n in range(3): 
    medianprops = dict(color=colors[n],linewidth=2)
    boxprops=dict(color=colors[n])
    df_1 = df[features][df['Target Class'] == target_class[n]]
    data = df_1.to_numpy()
    data[:, 0] = [x[:-3] for x in data[:,0]]
    data[:, 0] = data[:, 0].astype('float')

    n_subjects = data.shape[0]//16
    pc_dataset_all = np.zeros((16, 6, n_subjects))

    for i in range(n_subjects):
        pc_dataset_all[:,0,i] = data[:16, 0]
        for j in range(5):
            pc_dataset_all[:, 1+j, i] =  data[16*i:16*(i+1), 1+j]


    mat = np.zeros((4,n_subjects))



    count = 0
    for k in range(n_subjects):
        pc_dataset = pc_dataset_all[:,:,k]
        if np.sum(pc_dataset[:,1]==0)<8:
            for nn in range(2,6):
                df_pc_dataset = {}
                for i in [1,nn]:
                    df_pc_dataset[features[i]] = pc_dataset[:,i] 

                df_pc_dataset = pd.DataFrame(df_pc_dataset)
                maxlag = 3
                df_g = grangers_causation_matrix(df_pc_dataset, variables = df_pc_dataset.columns)
                print(df_g)
                mat[nn-2,count] = df_g.to_numpy()[v[0],v[1]]
            count += 1
    mat = mat[:,:count]

    for j in range(count):
        df_p_value['Target Class'].append(target_class[n])
        for i,ftr in enumerate(ff):
            df_p_value[ftr].append(mat[i,j])

    

    ax[n].boxplot(np.transpose(mat),
                  showfliers = False,
                  medianprops = medianprops, 
                  boxprops=boxprops, 
                  whiskerprops=boxprops, 
                  capprops=boxprops)
    #sns.violinplot(np.transpose(mat),ax=ax[n])
    ax[n].plot([0.5,4.5],[0.05,0.05],'--k',lw =1)
    ax[n].set_ylim([-0.02,1.7])
    ax[n].set_title(target_class[n])
    ax[n].set_xticklabels(ff)
    if n==0: 
        if v==[1,0]:
            ax[n].set_ylabel('Granger causality p-value \n (DA \u27F6 NT)')
        else:
            ax[n].set_ylabel('Granger causality p-value \n (NT \u27F6 DA)')

    ax[n].set_yscale('cuberoot')
    if n == 0:
        ax[n].set_yticks([0.0, 0.05, 0.4, 1.4])
    else:
        ax[n].set_yticks([])

plt.show()
if v == [1,0]:
    fig.savefig(f'/home/schatterjee/Desktop/causal_discovery/Granger_causality_DA_to_NT_{maxlag}.eps', dpi=300, format = 'eps')
else:
    fig.savefig(f'/home/schatterjee/Desktop/causal_discovery/Granger_causality_NT_to_DA_{maxlag}.eps', dpi=300, format = 'eps')


df_p_value = pd.DataFrame(df_p_value)

if v == [1,0]:
    df_p_value.to_csv('/home/schatterjee/Desktop/causal_discovery/Causality_DA_to_NT_pval.csv')
else:
    df_p_value.to_csv('/home/schatterjee/Desktop/causal_discovery/Causality_NT_to_DA_pval.csv')