from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
#tnse
from sklearn.manifold import TSNE
#for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# #exmale of tSNE
# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# y = np.array([1,0,1,0])
# tsne=TSNE(n_components=2)#, verbose=1, perplexity=40, n_iter=300)
# X_embedded = tsne.fit_transform(X)
# print(X.shape, X_embedded.shape)
#
# df_subset=pd.DataFrame(X)
# df_subset['y']=y
# df_subset['tsne-2d-one'] = X_embedded[:,0]
# df_subset['tsne-2d-two'] = X_embedded[:,1]
#
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one",
#     y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )
# plt.show()


### tSNE for TCC-benchmark
visual_ids=[10,12,14,15]
X_cnn, X_lstm=[],[]
for i in visual_ids:
    file_name='./results/tccbenchmark/test%d.npy'%i
    mat=np.load(file_name, allow_pickle=True)
    X_cnn.append(mat[0].cpu().numpy())
    X_lstm.append(mat[1].cpu().numpy())
X_cnn=np.squeeze(np.stack(X_cnn))
X_lstm=np.squeeze(np.stack(X_lstm))
#X_cnn (4, 512, 13, 13) X_lstm (4, 128, 13, 13)
X_cnn=X_cnn.reshape(X_cnn.shape[0], X_cnn.shape[1], -1)
X_lstm=X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], -1)
#X_cnn (4, 512, 169) X_lstm (4, 128, 169)
X_cnn=np.swapaxes(X_cnn,1,2)
X_lstm=np.swapaxes(X_lstm,1,2)
#X_cnn (4, 169, 512) X_lstm (4, 169, 128)
X_cnn=X_cnn.reshape(-1, 512)
X_lstm=X_lstm.reshape(-1,128)
print('X_cnn', X_cnn.shape, 'X_lstm', X_lstm.shape)
y=np.repeat(visual_ids, 169)


tsne=TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_embedded = tsne.fit_transform(X_cnn)#X_lstm
df=pd.DataFrame()
df['y']=y
df['tSNE-1'] = X_embedded[:,0]
df['tSNE-2'] = X_embedded[:,1]
plt.figure(figsize=(16,10))
sns.set_style("white", {#"axes.spines.left": "False",
                        #"axes.spines.bottom": "False",
                        "axes.spines.right": "False",
                        "axes.spines.top": "False",
                        #"xtick.bottom": "False",
                        #"ytick.left": "False",
                        })
sns.scatterplot(
    x="tSNE-1",
    y="tSNE-2",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df,
    legend="full",
    alpha=0.7,
)
plt.show()
