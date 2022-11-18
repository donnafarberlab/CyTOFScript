import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import phenograph
import umap

from matplotlib import colors as mcolors
from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

pd.set_option("max_columns", 50)

D455=pd.read_csv("D455_Tcell_CD3gate_bld_spl_LLN_MLN_ILN_lung_jej_skin_PBMC.csv")
D455.head(10)
parameters=list(D455)

parameters

markers = []
for p in parameters:
    spl = p.split('_')
    if len(spl) > 1:
        markers.append(spl[1])
    else:
         markers.append(p)
markers

D455.columns = markers
D455.head(10)

sel_columns =  'CD45','CD57', 'CD28', 'CD19', 'CD45RA', 'CD103', 'CD4', 'CD8', 'Perforin', 'CD16', 'CD127','CD1c','CD123', 'CD66b', 'TIGIT', 'ICOS', 'CD27', 'CCR5', 'Tcf1', 'CD14', 'CD56', 'gdTCR', 'CXCR5', 'CD69', 'CRTH2', 'CD25', 'CCR7', 'CD3', 'Tbet', 'CD38', 'CD95', 'LAG3','CXCR4','HLADR', 'PD-1','GranzymeB', 'CD11b', 'Classification Identifier'
D455_sub_data = D455.loc[:,sel_columns]
D455_sub_data.head(10)

D455_sub_data['Classification Identifier'].value_counts()
D455_sub_data['Donor']='D455'

D457=pd.read_csv("D457_Tcell_gatedonCD3_bld_spl_LLN_MLN_ILN_lung_jej_skin_PBMC.csv")
D457.head(10)
parameters=list(D457)

parameters

markers = []
for p in parameters:
    spl = p.split('_')
    if len(spl) > 1:
        markers.append(spl[1])
    else:
         markers.append(p)
markers

D457.columns = markers
D457.head(10)

sel_columns =  'CD45','CD57', 'CD28', 'CD19', 'CD45RA', 'CD103', 'CD4', 'CD8', 'Perforin', 'CD16', 'CD127','CD1c','CD123', 'CD66b', 'TIGIT', 'ICOS', 'CD27', 'CCR5', 'Tcf1', 'CD14', 'CD56', 'gdTCR', 'CXCR5', 'CD69', 'CRTH2', 'CD25', 'CCR7', 'CD3', 'Tbet', 'CD38', 'CD95', 'LAG3','CXCR4','HLADR', 'PD-1','GranzymeB', 'CD11b', 'Classification Identifier'
D457_sub_data = D457.loc[:,sel_columns]
D457_sub_data.head(10)

D457_sub_data['Classification Identifier'].value_counts()
D457_sub_data['Donor']='D457'


D461=pd.read_csv("D461_Tcell_bld_spl_LLN_MLN_ILN_lung_jej_skin_PBMC.csv")
D461.head(10)
parameters=list(D461)

parameters

markers = []
for p in parameters:
    spl = p.split('_')
    if len(spl) > 1:
        markers.append(spl[1])
    else:
         markers.append(p)
markers

D461.columns = markers
D461.head(10)

sel_columns =  'CD45','CD57', 'CD28', 'CD19', 'CD45RA', 'CD103', 'CD4', 'CD8', 'Perforin', 'CD16', 'CD127','CD1c','CD123', 'CD66b', 'TIGIT', 'ICOS', 'CD27', 'CCR5', 'Tcf1', 'CD14', 'CD56', 'gdTCR', 'CXCR5', 'CD69', 'CRTH2', 'CD25', 'CCR7', 'CD3', 'Tbet', 'CD38', 'CD95', 'LAG3','CXCR4','HLADR', 'PD-1','GranzymeB', 'CD11b', 'Classification Identifier'
D461_sub_data = D461.loc[:,sel_columns]
D461_sub_data.head(10)

D461_sub_data['Classification Identifier'].value_counts()
D461_sub_data['Donor']='D461'

D455_blood= D455_sub_data[D455_sub_data['Classification Identifier'] == 0]
D455_blood = D455_blood.sample(n=5531)
D455_blood.shape
D455_blood.head()

D455_spleen= D455_sub_data[D455_sub_data['Classification Identifier'] == 1]
D455_spleen = D455_spleen.sample(n=5531)
D455_spleen.shape
D455_spleen.head()

D455_LLN= D455_sub_data[D455_sub_data['Classification Identifier'] == 2]
D455_LLN = D455_LLN.sample(n=5531)
D455_LLN.shape
D455_LLN.head()


D455_MLN= D455_sub_data[D455_sub_data['Classification Identifier'] == 3]
D455_MLN = D455_MLN.sample(n=5531)
D455_MLN.shape
D455_MLN.head()

D455_ILN= D455_sub_data[D455_sub_data['Classification Identifier'] == 4]
D455_ILN = D455_ILN.sample(n=5531)
D455_ILN.shape
D455_ILN.head()

D455_lung= D455_sub_data[D455_sub_data['Classification Identifier'] == 5]
D455_lung = D455_lung.sample(n=5531)
D455_lung.shape
D455_lung.head()

D455_jejunum= D455_sub_data[D455_sub_data['Classification Identifier'] == 6]
D455_jejunum = D455_jejunum.sample(n=5531)
D455_jejunum.shape
D455_jejunum.head()

D455_skin= D455_sub_data[D455_sub_data['Classification Identifier'] == 7]
D455_skin = D455_skin.sample(n=5531)
D455_skin.shape
D455_skin.head()


D455_sub_sub_data = pd.concat([D455_blood, D455_spleen, D455_LLN, D455_MLN, 
                               D455_ILN, D455_lung, D455_jejunum, D455_skin], 
                            ignore_index=True, sort=False)
D455_sub_sub_data.shape
D455_sub_sub_data.head()

D457_blood= D457_sub_data[D457_sub_data['Classification Identifier'] == 0]
D457_blood = D457_blood.sample(n=5531)
D457_blood.shape
D457_blood.head()

D457_spleen= D457_sub_data[D457_sub_data['Classification Identifier'] == 1]
D457_spleen = D457_spleen.sample(n=5531)
D457_spleen.shape
D457_spleen.head()

D457_LLN= D457_sub_data[D457_sub_data['Classification Identifier'] == 2]
D457_LLN = D457_LLN.sample(n=5531)
D457_LLN.shape
D457_LLN.head()


D457_MLN= D457_sub_data[D457_sub_data['Classification Identifier'] == 3]
D457_MLN = D457_MLN.sample(n=5531)
D457_MLN.shape
D457_MLN.head()

D457_ILN= D457_sub_data[D457_sub_data['Classification Identifier'] == 4]
D457_ILN = D457_ILN.sample(n=5531)
D457_ILN.shape
D457_ILN.head()

D457_lung= D457_sub_data[D457_sub_data['Classification Identifier'] == 5]
D457_lung = D457_lung.sample(n=5531)
D457_lung.shape
D457_lung.head()

D457_jejunum= D457_sub_data[D457_sub_data['Classification Identifier'] == 6]
D457_jejunum = D457_jejunum.sample(n=5531)
D457_jejunum.shape
D457_jejunum.head()

D457_skin= D457_sub_data[D457_sub_data['Classification Identifier'] == 7]
D457_skin = D457_skin.sample(n=5531)
D457_skin.shape
D457_skin.head()


D457_sub_sub_data = pd.concat([D457_blood, D457_spleen, D457_LLN, D457_MLN, 
                               D457_ILN, D457_lung, D457_jejunum, D457_skin], 
                            ignore_index=True, sort=False)
D457_sub_sub_data.shape
D457_sub_sub_data.tail()

D461_blood= D461_sub_data[D461_sub_data['Classification Identifier'] == 0]
D461_blood = D461_blood.sample(n=5531)
D461_blood.shape
D461_blood.head()

D461_spleen= D461_sub_data[D461_sub_data['Classification Identifier'] == 1]
D461_spleen = D461_spleen.sample(n=5531)
D461_spleen.shape
D461_spleen.head()

D461_LLN= D461_sub_data[D461_sub_data['Classification Identifier'] == 2]
D461_LLN = D461_LLN.sample(n=5531)
D461_LLN.shape
D461_LLN.head()


D461_MLN= D461_sub_data[D461_sub_data['Classification Identifier'] == 3]
D461_MLN = D461_MLN.sample(n=5531)
D461_MLN.shape
D461_MLN.head()

D461_ILN= D461_sub_data[D461_sub_data['Classification Identifier'] == 4]
D461_ILN = D461_ILN.sample(n=5531)
D461_ILN.shape
D461_ILN.head()

D461_lung= D461_sub_data[D461_sub_data['Classification Identifier'] == 5]
D461_lung = D461_lung.sample(n=5531)
D461_lung.shape
D461_lung.head()

D461_jejunum= D461_sub_data[D461_sub_data['Classification Identifier'] == 6]
D461_jejunum = D461_jejunum.sample(n=5531)
D461_jejunum.shape
D461_jejunum.head()

D461_skin= D461_sub_data[D461_sub_data['Classification Identifier'] == 7]
D461_skin = D461_skin.sample(n=5531)
D461_skin.shape
D461_skin.head()


D461_sub_sub_data = pd.concat([D461_blood, D461_spleen, D461_LLN, D461_MLN, 
                               D461_ILN, D461_lung, D461_jejunum, D461_skin], 
                            ignore_index=True, sort=False)
D461_sub_sub_data.shape
D461_sub_sub_data.tail()


D455_sub_sub_data['site'] = D455_sub_sub_data['Classification Identifier'].values
def site_name(df):
    if df['site'] == 0:
        return "blood"
    elif df['site'] == 1:
        return "spleen"
    elif df['site'] == 2:
        return "LLN"
    elif df['site'] == 3:
        return "MLN"
    elif df['site'] == 4:
        return "ILN"
    elif df['site'] == 5:
        return "lung"
    elif df['site'] == 6:
        return "jejunum"
    elif df['site'] == 7:
        return "skin"    
D455_sub_sub_data['Tissue'] = D455_sub_sub_data.apply(site_name, axis = 1)
D455_sub_sub_data.tail()

D457_sub_sub_data['site'] = D457_sub_sub_data['Classification Identifier'].values
def site_name(df):
    if df['site'] == 0:
        return "blood"
    elif df['site'] == 1:
        return "spleen"
    elif df['site'] == 2:
        return "LLN"
    elif df['site'] == 3:
        return "MLN"
    elif df['site'] == 4:
        return "ILN"
    elif df['site'] == 5:
        return "lung"
    elif df['site'] == 6:
        return "jejunum"
    elif df['site'] == 7:
        return "skin"    
D457_sub_sub_data['Tissue'] = D457_sub_sub_data.apply(site_name, axis = 1)
D457_sub_sub_data.tail()

D461_sub_sub_data['site'] = D461_sub_sub_data['Classification Identifier'].values
def site_name(df):
    if df['site'] == 0:
        return "blood"
    elif df['site'] == 1:
        return "spleen"
    elif df['site'] == 2:
        return "LLN"
    elif df['site'] == 3:
        return "MLN"
    elif df['site'] == 4:
        return "ILN"
    elif df['site'] == 5:
        return "lung"
    elif df['site'] == 6:
        return "jejunum"
    elif df['site'] == 7:
        return "skin"    
D461_sub_sub_data['Tissue'] = D461_sub_sub_data.apply(site_name, axis = 1)
D461_sub_sub_data.tail()

sub_sub_data = pd.concat([D455_sub_sub_data, 
                          D457_sub_sub_data, D461_sub_sub_data], 
                            ignore_index=True, sort=False)

sub_sub_data.to_csv("sub_sub_data.csv")
sub_sub_data.columns
#removing the Classification Identifier from the data table so that transforming the data doesn't mess up the site information
dat_cols = sel_columns[0:-1]
sel_data = sub_sub_data.loc[:,dat_cols]
sel_data.shape
sel_data.head(5)
cofac = 5
trans_data = np.arcsinh(sel_data.values/cofac)
#putting the transformed data values into a data frame with markers as column titles
trans_data_df = pd.DataFrame(trans_data, columns= dat_cols)
trans_data_df.head()
trans_data_df.to_csv("trans_data_df.csv")
#to plot scatter plot like in FCS express
x = 'CD8'
y = 'CD19'
plt.scatter(trans_data_df[x], trans_data_df[y], s = 5)
# run tSNE
tsne = TSNE(perplexity = 30, random_state=6, n_jobs = 4, negative_gradient_method="bh")
embed = tsne.fit(trans_data_df)
# Pranay: random_state it tells it to randomly start at the 6th cell and affects the orientation of the plot
tsne_df = pd.DataFrame(embed, columns = ['tsne_x','tsne_y'])
tsne_df.head()
#plot TSNE
fig, ax = plt.subplots(figsize = (10,10))
ax.grid(False)
plt.scatter(tsne_df['tsne_x'], tsne_df['tsne_y'], c= 'gray', s = 1)
plt.xlabel('tSNE X')
plt.ylabel('tSNE Y')
#save TSNE
plt.savefig("tsne_trans_data.pdf")

#associate tsne coordinates with tissue site
tsne_df['Tissue'] = sub_sub_data['Tissue'].values
tsne_df['Donor'] = sub_sub_data['Donor'].values

tsne_df.head()

site_list = tsne_df.Tissue.unique()
print(site_list)
for site in site_list:
    #subset data
    df = tsne_df[tsne_df['Tissue'] == site]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.grid(False)
    plt.scatter(df['tsne_x'], df['tsne_y'], c= 'gray', s = 1)
    plt.title(site, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("tsne_trans_data_{}.pdf" .format(site))


donor_list = tsne_df.Donor.unique()
print(donor_list)
for donor in donor_list:
    #subset data
    df = tsne_df[tsne_df['Donor'] == donor]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.grid(False)
    plt.scatter(df['tsne_x'], df['tsne_y'], c= 'gray', s = 1)
    plt.title(donor, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("tsne_Tcell_alldata_{}.pdf" .format(donor))

tsne_df.to_csv("tsne_trans_data.csv")



# Read saved data in #
#tsne_data = pd.read_csv("tsne_df.csv", index_col = 0)
#
#trans_data = pd.read_csv("trans_data_df.csv", index_col =0)
p_data = trans_data_df.values
p_data

# Run phenograph
communities, graph, Q = phenograph.cluster(trans_data_df, k = 100)
print(set(communities))

tsne_df['cluster'] = communities
trans_data_df['cluster']=communities
tsne_df.head()
#defining colors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
palette = []
for k, v in colors.items():
    if "#" in v:
        palette.append(v)
palette = palette[1::4]
len(palette)
clust_color = dict(zip(tsne_df['cluster'].unique(), palette))
sample_color = tsne_df['cluster'].map(clust_color)
tsne_df['clust_color'] = sample_color

tsne_df.to_csv("tsne_trans_data.csv", index = False)

#tsne_df = pd.read_csv("tsne_data_communities_and_color.csv")
#tsne_df.head()
sub_sub_data['cluster_all']=tsne_df['cluster']
sub_sub_data['cluster_all_color']=tsne_df['clust_color']
sub_sub_data.tail()

sub_sub_data.columns
columns='CD45', 'CD57', 'CD28', 'CD19', 'CD45RA', 'CD103', 'CD4', 'CD8','Perforin', 'CD16', 'CD127', 'CD1c', 'CD123', 'CD66b', 'TIGIT', 'ICOS','CD27', 'CCR5', 'Tcf1', 'CD14', 'CD56', 'gdTCR', 'CXCR5', 'CD69','CRTH2', 'CD25', 'CCR7', 'CD3', 'Tbet', 'CD38', 'CD95', 'LAG3', 'CXCR4','HLADR', 'PD-1', 'GranzymeB', 'CD11b', 'Donor', 'Tissue', 'cluster_all','cluster_all_color'
sub_sub_data=sub_sub_data.loc[:,columns]

#trans_data_df=pd.read_csv("trans_data_df.csv", index_col="Unnamed: 0")
#trans_data_df.head()

#plot clustered tsne
fig, ax = plt.subplots(figsize = (10,10))
ax.set_facecolor('lightgray')
ax.grid(False)
scatter = plt.scatter(tsne_df['tsne_x'], tsne_df['tsne_y'], c= tsne_df['clust_color'], s = 5)
# legend
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
for i, label in enumerate(clust_color):
    plt.annotate(label,
                tsne_df.loc[tsne_df['cluster']==label, ['tsne_x','tsne_y']].mean(),
                color = 'k', size = 20)

plt.xlabel('tSNE X')
plt.ylabel('tSNE Y')
plt.savefig("phenographcluster_all.pdf")


site_list = tsne_df.Tissue.unique()
for site in site_list:
    #subset data
    df = tsne_df[tsne_df['Tissue'] == site]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_facecolor("lightgrey")
    ax.grid(False)
    plt.scatter(df['tsne_x'], df['tsne_y'], c= df['clust_color'], s = 5)
    for i, label in enumerate(clust_color):
        plt.annotate(label,
                df.loc[df['cluster']==label, ['tsne_x','tsne_y']].mean(),
                color = 'k', size = 20)
   
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
    plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
    plt.title(site, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("phenographcluster_all_{}.pdf" .format(site), bbox_inches="tight")

donor_list = tsne_df.Donor.unique()
for donor in donor_list:
    #subset data
    df = tsne_df[tsne_df['Donor'] == donor]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_facecolor("lightgrey")
    ax.grid(False)
    plt.scatter(df['tsne_x'], df['tsne_y'], c= df['clust_color'], s = 5)
    for i, label in enumerate(clust_color):
        plt.annotate(label,
                df.loc[df['cluster']==label, ['tsne_x','tsne_y']].mean(),
                color = 'k', size = 20)
   
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
    plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
    plt.title(donor, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("phenographcluster_all_{}.pdf" .format(donor), bbox_inches="tight")


# generate heatmap
exp_data = trans_data_df
exp_data['Tissue'] = tsne_df['Tissue']
exp_data.head()
    #generate a table of average expression of markers based on cluster
avg_exp = pd.pivot_table(exp_data, index = 'cluster')
avg_exp.head()
avg_exp.columns
avg_exp['CD3']
submarkers = ['CD3','CD4','CD8','CCR7','CD45RA','CD69','CD103','CCR5','Tbet', 'CRTH2','Tcf1', 'CXCR4', 
              'CXCR5', 'ICOS', 'PD-1', 'CD57', 'LAG3', 'TIGIT', 'CD25', 'CD127','CD95', 'CD27', 'CD28', 
              'CD38', 'HLADR', 'GranzymeB','Perforin', 'CD56', 'gdTCR', 'CD14', 'CD19','CD16','CD123',
              'CD1c','CD66b', 'CD11b']
sns.set(font_scale=1)
g = sns.clustermap(avg_exp[submarkers], figsize= (20, 10), cmap = 'Greys', 
                   linewidths = 0.01, linecolor = 'k', row_cluster = True, 
                   col_cluster = False, 
                      cbar_kws={"label":"Mean Expression Data", 
                                "orientation" :"horizontal"}, metric= "correlation")
g.cax.set_position((0.29, -0.10, 0.60, 0.055))
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("heatmap_all.pdf", bbox_inches="tight")
    #dpi "dots per inch" to adjust resolution


exp_data.head()
exp_data['tsne_x']=tsne_df['tsne_x']
exp_data['tsne_y']=tsne_df['tsne_y']
for site in site_list:
#    #subset data
    df = exp_data[exp_data['Tissue'] == site]
#    # generate plot
    for m in submarkers:
        fig, ax = plt.subplots(figsize = (6,5))
        ax.grid(False)
        plt.scatter(df['tsne_x'], df['tsne_y'], c= df[m].values, s = 5, cmap='jet')
        plt.title(site + '_'+ m, size = 15)
        plt.colorbar()
        plt.xlabel('tSNE X')
        plt.ylabel('tSNE Y')
        plt.savefig("all_{}_{}.pdf" .format(site, m), bbox_inches="tight")   

for m in submarkers:
        fig, ax = plt.subplots(figsize = (6,5))
        ax.grid(False)
        plt.scatter(exp_data['tsne_x'], exp_data['tsne_y'], c= exp_data[m].values, s = 5, cmap='jet')
        plt.title(m, size = 15)
        plt.colorbar()
        plt.xlabel('tSNE X')
        plt.ylabel('tSNE Y')
        plt.savefig("all_{}.pdf" .format(m), bbox_inches="tight")   

clust_freq_table = pd.DataFrame()
for site in site_list:
    df = exp_data[exp_data["Tissue"] == site]
    cells_inclust = df.cluster.value_counts().sort_index()
    freq_clust = (cells_inclust/len(df))* 100
    clust_freq_table = pd.concat([clust_freq_table, freq_clust], axis = 1)
    clust_freq_table.fillna('0', inplace = True)

clust_freq_table.columns = site_list
clust_freq_table = clust_freq_table.astype('float64')
print(clust_freq_table)
g = sns.clustermap(clust_freq_table, cmap = "Greys")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("all_clusterfrequencyheatmap.pdf",bbox_inches='tight')



exp_data.cluster.unique()
exp_data.columns
exp_data.shape
exp_data['clust_color']=tsne_df['clust_color']
exp_data['Donor']=tsne_df['Donor']
exp_data['tsne_x']=tsne_df['tsne_x']
exp_data['tsne_y']=tsne_df['tsne_y']


embedding = umap.UMAP(n_neighbors=20,
                      min_dist=0.3,
                      metric='correlation').fit_transform(exp_data[submarkers])

umap_df = pd.DataFrame(embedding, columns = ['umap_x', 'umap_y'])
umap_df['cluster'] = exp_data['cluster'].values
umap_df['clust_color'] = exp_data['clust_color'].values
umap_df.head()

fig, ax = plt.subplots(figsize = (10 , 10))
ax.grid(False)
plt.scatter(umap_df['umap_x'], umap_df['umap_y'], s = 3, c = umap_df['clust_color'])
for i, label in enumerate(clust_color):
    plt.annotate(label,
                umap_df.loc[umap_df['cluster']==label, ['umap_x','umap_y']].mean(),
                color = 'k', size = 20)
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
plt.xlabel('UMAP X')
plt.ylabel('UMAP Y')
plt.savefig("UMAP_all.pdf",bbox_inches='tight')

exp_data['umap_x'] = umap_df['umap_x']
exp_data['umap_y'] = umap_df['umap_y']
exp_data.head()
exp_data.columns
exp_data.to_csv('exp_data_tsne_umap.csv', index = False)
exp_data = pd.read_csv('exp_data_tsne_umap.csv')
exp_data.head()    

Tcell_clust = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 
               11, 12, 13, 14, 17, 18, 20,
               21, 22, 23, 27, 29, 31]
#cluster 9, 19 excluded because low CD45 expression
#cluster 15 excluded for high CD16 expression (>>2)
#cluster 16, 24 excluded for high CD11b expression (>2)
#cluster 26 excluded for high expression of multiple myeloid markers
# cluster 28 excluded for high CD1c and CD11b expression
#cluster 30 excluded for high CD19 and CD123 expression

Tcell = exp_data[exp_data['cluster'].isin(Tcell_clust)]
Tcell.head()
Tcell.shape
Tcell['Tissue'].value_counts()
T455 = Tcell.loc[Tcell['Donor'] == 'D455']
T455['Tissue'].value_counts()
T457 = Tcell.loc[Tcell['Donor'] == 'D457']
T457['Tissue'].value_counts()
T461 = Tcell.loc[Tcell['Donor'] == 'D461']
T461['Tissue'].value_counts()

Tissues = T455['Tissue'].unique()
Tissues

T455_sub = pd.DataFrame()
for t in Tissues:
    df = T455[T455['Tissue'] == t]
    df = df.sample(n=3776)
    T455_sub = pd.concat([T455_sub, df], axis = 0)
T455_sub.shape

T457_sub = pd.DataFrame()
for t in Tissues:
    df = T457[T457['Tissue'] == t]
    df = df.sample(n=2284)
    T457_sub = pd.concat([T457_sub, df], axis = 0)
T457_sub.shape

T461_sub = pd.DataFrame()
for t in Tissues:
    df = T461[T461['Tissue'] == t]
    df = df.sample(n=2579)
    T461_sub = pd.concat([T461_sub, df], axis = 0)
T461_sub.shape

Tcell=pd.concat([T455_sub, T457_sub, T461_sub], 
                            ignore_index=True, sort=False)
Tcell['Tissue'].value_counts()

Tcell_cols = ['CD4','CD8','CCR7','CD45RA','CD69','CD103','CCR5','Tbet', 'CRTH2','Tcf1', 'CXCR4', 
              'CXCR5', 'ICOS', 'PD-1', 'CD57', 'LAG3', 'TIGIT', 'CD25', 'CD127','CD95', 'CD27', 'CD28', 
              'CD38', 'HLADR', 'GranzymeB','Perforin', 'CD56', 'gdTCR']

tsne = TSNE(perplexity = 200, random_state=6, n_jobs = 8, negative_gradient_method="bh")
embed = tsne.fit(Tcell[Tcell_cols])

Tcell_tsne_df = pd.DataFrame(embed, columns = ['new_tsne_x', 'new_tsne_y'])

fig, ax = plt.subplots(figsize = (10,10))
ax.grid(False)
plt.scatter(Tcell_tsne_df['new_tsne_x'], Tcell_tsne_df['new_tsne_y'], c= 'grey', s = 1)
plt.xlabel('tSNE X')
plt.ylabel('tSNE Y')
plt.savefig("T_tsne.pdf", bbox_inches="tight")
#
##plot tsne with myeloid subsets for each tisue
Tcell_tsne_df["Tissue"]=Tcell['Tissue'].values
Tcell_tsne_df["Donor"]=Tcell["Donor"].values
Tcell_tsne_df.shape
Tcell_tsne_df

site_list = Tcell_tsne_df.Tissue.unique()
site_list
for site in site_list:
    #subset data
    df = Tcell_tsne_df[Tcell_tsne_df['Tissue'] == site]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.grid(False)
    plt.scatter(df['new_tsne_x'], df['new_tsne_y'], c= 'grey', s = 2)
    plt.title(site, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("T_tsne_{}.pdf" .format(site), bbox_inches="tight")

donor_list = Tcell_tsne_df.Donor.unique()
print(donor_list)
for donor in donor_list:
    #subset data
    df = Tcell_tsne_df[Tcell_tsne_df['Donor'] == donor]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.grid(False)
    plt.scatter(df['new_tsne_x'], df['new_tsne_y'], c= 'grey', s = 2)
    plt.title(donor, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("T_tsne_{}.pdf" .format(donor))


Tcell_cols = ['CD3','CD4','CD8','CCR7','CD45RA','CD69','CD103','CCR5','Tbet', 'CRTH2','Tcf1', 'CXCR4', 
              'CXCR5', 'ICOS', 'PD-1', 'CD57', 'LAG3', 'TIGIT', 'CD25', 'CD127','CD95', 'CD27', 'CD28', 
              'CD38', 'HLADR', 'GranzymeB','Perforin', 'CD56', 'gdTCR']
marker = Tcell_cols
mms = MinMaxScaler()
data = mms.fit_transform(Tcell[marker])
scaled_data = pd.DataFrame(data, columns = marker)
scaled_data.head()

for m in marker:
    scaled_data[m]
    fig, ax = plt.subplots(figsize = (12,10))
    ax.grid(False)
    plt.scatter(Tcell_tsne_df['new_tsne_x'], Tcell_tsne_df['new_tsne_y'], c = scaled_data[m].values, s = 1, cmap = 'jet')
    plt.colorbar()
    plt.title(m, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("T_{}.pdf" .format(m), bbox_inches="tight")    

##consolidate data into the "scaled_data" dataframe
scaled_data["Tissue"]= Tcell['Tissue'].values
scaled_data['new_tsne_x'] = Tcell_tsne_df['new_tsne_x'].values
scaled_data['new_tsne_y'] = Tcell_tsne_df['new_tsne_y'].values
scaled_data['donor']=Tcell_tsne_df['Donor']
scaled_data.head() 
scaled_data.Tissue.value_counts()

submarkers = ['CD4','CD8','CCR7','CD45RA','CD69','CD103','CCR5','Tbet', 'CRTH2','Tcf1', 'CXCR4', 
              'CXCR5', 'ICOS', 'PD-1', 'CD57', 'LAG3', 'TIGIT', 'CD25', 'CD127','CD95', 'CD27', 'CD28', 
              'CD38', 'HLADR', 'GranzymeB','Perforin', 'CD56', 'gdTCR']

site_list = scaled_data.Tissue.unique()
for site in site_list:
    #subset data
   df = scaled_data[scaled_data['Tissue'] == site]
    # generate plot
   for m in submarkers:
        fig, ax = plt.subplots(figsize = (6,5))
        ax.grid(False)
        plt.scatter(df['new_tsne_x'], df['new_tsne_y'], c= df[m].values, s = 1, cmap='jet')
        plt.title(site + '_'+ m, size = 15)
        plt.colorbar()
        plt.xlabel('tSNE X')
        plt.ylabel('tSNE Y')
        plt.savefig("T_{}_{}.pdf" .format(site, m), bbox_inches="tight")   
        
#run phenograph   


communities, graph, Q = phenograph.cluster(Tcell[submarkers], k = 30)
print(set(communities))
Tcell_tsne_df['cluster_new'] = communities

#graph clusters with phenograph colors/labels

palette=['#e55958','#FFBB4D','#45B0E5','#FF884D','#FFA071',
         '#FA8072','#5E94FF','#F17171','#136EF8','#A2D471',
         '#71D0FF','#0000FF','#71E2D0','#D64545','#d97cac',
         '#DDA0DD','#838DFF','#9065E5','#D6BFD8','#c58639',
         '#FF7187','#FFB6C1','#efc2ef','#C71585','#FFA500',
         '#45B0E5','#CCCCFF','#B38Dff','#45C5B0','#FFC44D',
         '#A2DCE4']
        
        
#        
#        '#14','c1','c0','c16','c2',
#         'c11','c3','c25','c10','#c5',
#         'c17','c4','c13','c29','c7',#C71585
#         'c26','#c20','c9','c8','c19',
#         'c27','c12','c28','c23','c30',
#         'c18','#D6BFD8','#c58639','c15','c24',
#         'c6']
        
        
        
#        '#F17171','#FA8072','#C71585','#45B0E5','#B38Dff',
#         '#FFA071','#DDA0DD','#FFB6C1','#FF884D','#D64545',
#         '#FFC44D','#4DDBC4','#71D0FF','#71E2D0','#A2D471',
#         '#838DFF','#5E94FF','#0000FF','#9065E5','#FFA500',
#         '#FFBB4D','#FF7187','#B0E0E6','#FF4500','#45C5B0',
#         '#D6BFD8','#CCCCFF','#FF99CC','#8B0000','#CF0426',
#         '#92EB92','#EEB13B']

clust_color = dict(zip(Tcell['cluster_new'].unique(), palette))
sample_color = Tcell['cluster_new'].map(clust_color)
Tcell['clust_color_new'] = sample_color

plt.rc('axes', axisbelow=True)
fig, ax = plt.subplots(figsize = (10,10))
ax.grid(True, color='white')
ax.set_facecolor("#F5F5F5")
plt.scatter(Tcell['new_tsne_x'], Tcell['new_tsne_y'], c= Tcell['clust_color_new'], s = 2)
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
for i, label in enumerate(clust_color):
    plt.annotate(label,
                Tcell.loc[Tcell['cluster_new']==label, ['new_tsne_x','new_tsne_y']].mean(),
                color = 'k', size = 20)

plt.xlabel('tSNE X')
plt.ylabel('tSNE Y')
plt.savefig("T_phenograph.pdf")
#
for site in site_list:
    #subset data
    df = Tcell[Tcell['Tissue'] == site]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_facecolor("#f5f5f5")
    plt.scatter(df['new_tsne_x'], df['new_tsne_y'], c= df['clust_color_new'], s = 1)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
    plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
    for i, label in enumerate(clust_color):
        plt.annotate(label,
                df.loc[df['cluster_new']==label, ['new_tsne_x','new_tsne_y']].mean(),
                color = 'k', size = 20)
    plt.title(site, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("T_phenograph_{}.pdf" .format(site), bbox_inches="tight")

donor_list = Tcell.Donor.unique()

for donor in donor_list:
    #subset data
    df = Tcell[Tcell['Donor'] == donor]
    # generate plot
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_facecolor("#f5f5f5")

    plt.scatter(df['new_tsne_x'], df['new_tsne_y'], c= df['clust_color_new'], s = 2)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
    plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
    for i, label in enumerate(clust_color):
        plt.annotate(label,
                df.loc[df['cluster_new']==label, ['new_tsne_x','new_tsne_y']].mean(),
                color = 'k', size = 20)
    plt.title(donor, size = 15)
    plt.xlabel('tSNE X')
    plt.ylabel('tSNE Y')
    plt.savefig("T_phenograph_{}.pdf" .format(donor), bbox_inches="tight")




# generate heatmap
submarkers=['CD3','CD4','CD8','CCR7','CD45RA','CD69','CD103','CCR5','Tbet', 'CRTH2','Tcf1', 'CXCR4', 
              'CXCR5', 'ICOS', 'PD-1', 'CD57', 'LAG3', 'TIGIT', 'CD25', 'CD127','CD95', 'CD27', 'CD28', 
              'CD38', 'HLADR', 'GranzymeB','Perforin', 'CD56', 'gdTCR']
Tcell_exp_data = Tcell[submarkers]
Tcell_exp_data['cluster_new'] = communities
Tcell_exp_data['Tissue'] = Tcell_tsne_df['Tissue']
Tcell_exp_data.head()
Tcell_exp_data.shape
Tcell_exp_data.columns


Tcell_avg_exp = pd.pivot_table(Tcell_exp_data, index = 'cluster_new')
Tcell_avg_exp.to_csv("T_cell_avg_exp.csv")
Tcell_avg_exp = pd.read_csv('T_cell_avg_exp.csv')
Tcell_avg_exp
Tcell_avg_exp.isnull().values.any()

g = sns.clustermap(Tcell_avg_exp[submarkers], standard_scale=1, figsize= (20, 10), cmap = 'Blues', 
                   linewidths = 0.01, linecolor = 'k', row_cluster = True, col_cluster = False, 
             cbar_kws= {"orientation":"horizontal", "label" : "Mean scaled expression"}, 
             metric= "correlation")
g.cax.set_position((0.29, -0.1, 0.60, 0.055))
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("Tcellsubset_heatmap.pdf",bbox_inches='tight')



markers=['CD3','CD4','CD8','CCR7','CD45RA','CD69','CD103','CCR5','Tbet', 'CRTH2','Tcf1', 'CXCR4', 
              'CXCR5', 'ICOS', 'PD-1', 'CD57', 'LAG3', 'TIGIT', 'CD25', 'CD127','CD95', 'CD27', 'CD28', 
              'CD38', 'HLADR', 'GranzymeB','Perforin', 'CD56', 'gdTCR']
mms = MinMaxScaler()
data = mms.fit_transform(Tcell_avg_exp[markers])
scaled_avg_exp = pd.DataFrame(data, columns = markers)
scaled_avg_exp
scaled_avg_exp["cluster"]=[0,'CD8','CD8',3,4,5,6,'CD8',8,9,10,
              'CD8','CD8',13,'CD8',15,'CD8',17,18, 'CD8',20,
              'CD8',22,'CD8','CD8','CD8',26,27,'CD8','CD8','CD8']
CD4_clust=scaled_avg_exp.loc[scaled_avg_exp.cluster != 'CD8']
CD8_clust=scaled_avg_exp.loc[scaled_avg_exp.cluster == 'CD8']
CD4_clust
CD8_clust
CD4_clust.shape[0]+CD8_clust.shape[0]

submarkers=['CCR7','CD45RA','CD69','CD103','CCR5','Tbet', 'CRTH2','Tcf1', 'CXCR4', 
              'CXCR5', 'ICOS', 'PD-1', 'CD57', 'LAG3', 'TIGIT', 'CD25', 'CD127','CD95', 'CD27', 'CD28', 
              'CD38', 'HLADR', 'GranzymeB','Perforin', 'CD56', 'gdTCR']
del CD8_clust ['cluster']

g = sns.clustermap(CD8_clust[submarkers], figsize= (20, 6), cmap = 'Blues', 
                   linewidths = 0.01, linecolor = 'k', row_cluster = True, col_cluster = False, 
             cbar_kws= {"orientation":"horizontal", "label" : "Mean scaled expression"}, 
             metric= "correlation")
g.cax.set_position((0.29, -0.1, 0.60, 0.055))
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("Tcellsubset_heatmap_CD8.pdf",bbox_inches='tight')

del CD4_clust ['cluster']

g = sns.clustermap(CD4_clust[submarkers],figsize= (20, 6.3), cmap = 'Greens', 
                   linewidths = 0.01, linecolor = 'k', row_cluster = True, col_cluster = False, 
             cbar_kws= {"orientation":"horizontal", "label" : "Mean scaled expression"}, 
             metric= "correlation")
g.cax.set_position((0.29, -0.1, 0.60, 0.055))
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("Tcellsubset_heatmap_CD4.pdf",bbox_inches='tight')




#save data as csv so don't have to run tsne again and get a slightly different graph
Tcell.shape
Tcell.head()


scaled_data.head()
scaled_data['new_clust_color']=Tcell['clust_color_new']
scaled_data.to_csv('Tcell_scaled_exp_data.csv', index = False)
scaled_data = pd.read_csv('Tcell_scaled_exp_data.csv')

#trans_data_df.head()
#trans_data_df['cluster']= exp_data['cluster']
#trans_data_df['tsne_x'] = tsne_df['tsne_x']
#trans_data_df['tsne_y'] = tsne_df['tsne_y']
#trans_data_df['all_data_clust_color'] = tsne_df['clust_color'].values
#trans_data_df.to_csv('Tcell_alldata_transformed_exp_data.csv', index = False)

tissue_list = Tcell.Tissue.unique()
tissue_list

#calculating cluster frequencies
clust_freq_table = pd.DataFrame()
for tissue in tissue_list:
    df = Tcell[Tcell["Tissue"] == tissue]
    cells_inclust = df.cluster_new.value_counts().sort_index()
    freq_clust = (cells_inclust/len(df))* 100
    clust_freq_table = pd.concat([clust_freq_table, freq_clust], axis = 1)
    clust_freq_table.fillna('0', inplace = True)

clust_freq_table.columns = tissue_list
clust_freq_table = clust_freq_table.astype('float64')
print(clust_freq_table)
clust_freq_table.to_csv('T_cluster_freq.csv', index=False)

g = sns.clustermap(clust_freq_table, cmap = "Purples")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap.pdf",bbox_inches='tight')

g = sns.clustermap(clust_freq_table, cmap = "Purples", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap_colzscore.pdf",bbox_inches='tight')


##creating histograms
#scaled_data.head()
#c15 = scaled_data[scaled_data["cluster_new"] == 15]
#c14 = scaled_data[scaled_data["cluster_new"] == 14]
#sns.kdeplot(c15['CD4'], shade = True, label = 'C15')
#sns.kdeplot(c14['CD4'], shade = True, color = "r", label = 'C14')
#print(np.median(c15['CD4']))
#print(np.median(c14['CD4']))

#UMAP projection
scaled_data.cluster_new.unique()
embedding = umap.UMAP(n_neighbors=20,
                      min_dist=0.3,
                      metric='correlation').fit_transform(Tcell[Tcell_cols])
Tcell.head()
Tcell_tsne_df.head()
umap_df = pd.DataFrame(embedding, columns = ['new_umap_x', 'new_umap_y'])
umap_df['cluster_new'] = Tcell['cluster_new'].values
umap_df['clust_color_new'] = Tcell_tsne_df['clust_color_new'].values
umap_df.head()

fig, ax = plt.subplots(figsize = (10 , 10))
ax.grid(True, color='white')
ax.set_facecolor("#f5f5f5")
plt.scatter(Tcell['new_umap_x'], Tcell['new_umap_y'], s = 1, c = Tcell['clust_color_new'])
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in clust_color.values()]
plt.legend(markers, clust_color.keys(), numpoints=1,bbox_to_anchor=(1.1, 1.01))
for i, label in enumerate(clust_color):
    plt.annotate(label,
                Tcell.loc[Tcell['cluster_new']==label, ['new_umap_x','new_umap_y']].mean(),
                color = 'k', size = 20)

plt.xlabel('UMAP X')
plt.ylabel('UMAP Y')
plt.savefig("Tcellsubset_UMAP.pdf",bbox_inches='tight')

Tcell.columns
Tcell.to_csv('Tcell_transformed_exp_data.csv', index = False)

Tcell.shape

clust_freq_table = pd.read_csv('T_cluster_freq.csv')
clust_freq_table["cluster"]=[0,'CD8','CD8',3,4,5,6,'CD8',8,9,10,
              'CD8','CD8',13,'CD8',15,'CD8',17,18, 'CD8',20,
              'CD8',22,'CD8','CD8','CD8',26,27,'CD8','CD8','CD8']
CD4_clust=clust_freq_table.loc[clust_freq_table.cluster != 'CD8']
CD8_clust=clust_freq_table.loc[clust_freq_table.cluster == 'CD8']
CD4_clust
CD8_clust
CD4_clust.shape[0]+CD8_clust.shape[0]

del CD8_clust ['cluster']

g = sns.clustermap(CD8_clust, cmap = "Blues")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap.pdf",bbox_inches='tight')
g = sns.clustermap(CD8_clust, cmap = "Blues", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap_colzscore.pdf",bbox_inches='tight')

del CD4_clust ['cluster']
g = sns.clustermap(CD4_clust, cmap = "Greens")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap.pdf",bbox_inches='tight')
g = sns.clustermap(CD4_clust, cmap = "Greens", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap_colzscore.pdf",bbox_inches='tight')

Tcell = pd.read_csv('Tcell_transformed_exp_data.csv')
Tcell.Donor.value_counts()
T_D455=Tcell.loc[Tcell['Donor']== 'D455']
T_D457=Tcell.loc[Tcell['Donor']== 'D457']
T_D461=Tcell.loc[Tcell['Donor']== 'D461']


D455table = pd.DataFrame()
for tissue in tissue_list:
    df = T_D455[T_D455["Tissue"] == tissue]
    cells_inclust = df.cluster_new.value_counts().sort_index()
    freq_clust = (cells_inclust/len(df))* 100
    D455table = pd.concat([D455table, freq_clust], axis = 1)
    D455table.fillna('0', inplace = True)

D455table.columns = tissue_list
D455table = D455table.astype('float64')
print(D455table)
D455table.to_csv('T_cluster_freq_D455.csv', index=False)

g = sns.clustermap(D455table, cmap = "Purples")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap_D455.pdf",bbox_inches='tight')

g = sns.clustermap(D455table, cmap = "Purples", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap_colzscore_D455.pdf",bbox_inches='tight')



D455table["cluster"]=[0,'CD8','CD8',3,4,5,6,'CD8',8,9,10,
              'CD8','CD8',13,'CD8',15,'CD8',17,18, 'CD8',20,
              'CD8',22,'CD8','CD8','CD8',26,27,'CD8','CD8','CD8']
CD4_clust=D455table.loc[D455table.cluster != 'CD8']
CD8_clust=D455table.loc[D455table.cluster == 'CD8']
CD4_clust
CD8_clust
CD4_clust.shape[0]+CD8_clust.shape[0]

del CD8_clust ['cluster']

g = sns.clustermap(CD8_clust, cmap = "Blues")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap_D455.pdf",bbox_inches='tight')
g = sns.clustermap(CD8_clust, cmap = "Blues", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap_colzscore_D455.pdf",bbox_inches='tight')

del CD4_clust ['cluster']
g = sns.clustermap(CD4_clust, cmap = "Greens")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap_D455.pdf",bbox_inches='tight')
g = sns.clustermap(CD4_clust, cmap = "Greens", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap_colzscore_D455.pdf",bbox_inches='tight')


D457table = pd.DataFrame()
for tissue in tissue_list:
    df = T_D457[T_D457["Tissue"] == tissue]
    cells_inclust = df.cluster_new.value_counts().sort_index()
    freq_clust = (cells_inclust/len(df))* 100
    D457table = pd.concat([D457table, freq_clust], axis = 1)
    D457table.fillna('0', inplace = True)

D457table.columns = tissue_list
D457table = D457table.astype('float64')
print(D457table)
D457table.to_csv('T_cluster_freq_D457.csv', index=False)

g = sns.clustermap(D457table, cmap = "Purples")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap_D457.pdf",bbox_inches='tight')

g = sns.clustermap(D457table, cmap = "Purples", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap_colzscore_D457.pdf",bbox_inches='tight')



D457table["cluster"]=[0,'CD8','CD8',3,4,5,6,'CD8',8,9,10,
              'CD8','CD8',13,'CD8',15,'CD8',17,18, 'CD8',20,
              'CD8',22,'CD8','CD8','CD8',26,27,'CD8','CD8','CD8']
CD4_clust=D457table.loc[D457table.cluster != 'CD8']
CD8_clust=D457table.loc[D457table.cluster == 'CD8']
CD4_clust
CD8_clust
CD4_clust.shape[0]+CD8_clust.shape[0]

del CD8_clust ['cluster']

g = sns.clustermap(CD8_clust, cmap = "Blues")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap_D457.pdf",bbox_inches='tight')
g = sns.clustermap(CD8_clust, cmap = "Blues", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap_colzscore_D457.pdf",bbox_inches='tight')

del CD4_clust ['cluster']
g = sns.clustermap(CD4_clust, cmap = "Greens")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap_D457.pdf",bbox_inches='tight')
g = sns.clustermap(CD4_clust, cmap = "Greens", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap_colzscore_D457.pdf",bbox_inches='tight')


D461table = pd.DataFrame()
for tissue in tissue_list:
    df = T_D461[T_D461["Tissue"] == tissue]
    cells_inclust = df.cluster_new.value_counts().sort_index()
    freq_clust = (cells_inclust/len(df))* 100
    D461table = pd.concat([D461table, freq_clust], axis = 1)
    D461table.fillna('0', inplace = True)

D461table.columns = tissue_list
D461table = D461table.astype('float64')
new_row = {'blood':0, 'spleen':0,'LLN':0,'MLN':0,'ILN':0,'lung':0, 'jejunum':0, 'skin':0}
D461table = D461table.append(new_row,ignore_index=True)
print(D461table)
D461table.to_csv('T_cluster_freq_D461.csv', index=False)

g = sns.clustermap(D461table, cmap = "Purples")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap_D461.pdf",bbox_inches='tight')

g = sns.clustermap(D461table, cmap = "Purples", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("T_clusterfrequencyheatmap_colzscore_D461.pdf",bbox_inches='tight')



D461table["cluster"]=[0,'CD8','CD8',3,4,5,6,'CD8',8,9,10,
              'CD8','CD8',13,'CD8',15,'CD8',17,18, 'CD8',20,
              'CD8',22,'CD8','CD8','CD8',26,27,'CD8','CD8','CD8']
CD4_clust=D461table.loc[D461table.cluster != 'CD8']
CD8_clust=D461table.loc[D461table.cluster == 'CD8']
CD4_clust
CD8_clust
CD4_clust.shape[0]+CD8_clust.shape[0]

del CD8_clust ['cluster']

g = sns.clustermap(CD8_clust, cmap = "Blues")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap_D461.pdf",bbox_inches='tight')
g = sns.clustermap(CD8_clust, cmap = "Blues", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD8_clusterfrequencyheatmap_colzscore_D461.pdf",bbox_inches='tight')

del CD4_clust ['cluster']
g = sns.clustermap(CD4_clust, cmap = "Greens")
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap_D461.pdf",bbox_inches='tight')
g = sns.clustermap(CD4_clust, cmap = "Greens", z_score=1)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("CD4_clusterfrequencyheatmap_colzscore_D461.pdf",bbox_inches='tight')