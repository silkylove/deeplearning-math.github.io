import pandas as pd
import numpy as np
from ggplot import *
import keras
from keras.datasets import fashion_mnist
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

train_feature = np.asarray(pd.read_csv('train_feature.csv'))[:,1:]
test_feature = np.asarray(pd.read_csv('test_feature.csv'))[:,1:]
#(train_image, train_label), (test_image, test_label) = fashion_mnist.load_data()
'''
train_image = train_image.reshape((60000, 28*28))
print(train_image.shape)
pd.DataFrame(train_image).to_csv('train_image.csv', index = False)
pd.DataFrame(train_label).to_csv('train_label.csv', index = False)
test_image = test_image.reshape((10000, 28*28))
print(test_image.shape)
pd.DataFrame(test_image).to_csv('test_image.csv', index = False)
pd.DataFrame(test_label).to_csv('test_label.csv', index = False)
'''
train_image = np.asarray(pd.read_csv('train_image.csv'))
train_label = np.asarray(pd.read_csv('train_label.csv')).reshape(60000)
test_image = np.asarray(pd.read_csv('test_image.csv'))
test_label = np.asarray(pd.read_csv('test_label.csv')).reshape(10000)

pca = PCA(n_components = 2)
pca_result = pca.fit_transform(train_feature)
pca_one = pca_result[:,0]
pca_two = pca_result[:,1]
print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

pca_data = {'pca_one': pca_one, 'pca_two': pca_two, 'label': train_label}
df = pd.DataFrame(data = pca_data)
df['label'] = df['label'].apply(lambda i: str(i))
rndperm = np.random.permutation(df.shape[0])

ggplot( df.loc[rndperm[:3500],:], aes(x='pca_one', y='pca_two', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("First and Second Principal Components colored by digit")

n_sne = 5000
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(train_feature[:n_sne])
print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

x_tsne = tsne_results[:,0]
y_tsne = tsne_results[:,1]

tsne_data = {"x_tsne": x_tsne, "y_tsne": y_tsne, "label": train_label[:n_sne]}
df_tsne = pd.DataFrame(data = tsne_data)
df_tsne['label'] = df_tsne['label'].apply(lambda i: str(i))

ggplot( df_tsne.loc[:n_sne,:], aes(x='x_tsne', y='y_tsne', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("tSNE dimensions colored by digit")

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(train_feature)
print ('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50.explained_variance_ratio_)))

#T-sne using 50 dimensionality ,only use 5000 samples
n_sne_50 = 5000
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50[:n_sne_50])
print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# visualization

df_tsne_50 = df_tsne.loc[:n_sne_50, :].copy()
df_tsne_50['x-tsne-pca'] = tsne_pca_results[:,0]
df_tsne_50['y-tsne-pca'] = tsne_pca_results[:,1]

ggplot( df_tsne_50, aes(x='x-tsne-pca', y='y-tsne-pca', color='label') ) \
		+ geom_point(size=80,alpha=0.8) \
		+ ggtitle("tSNE dimensions colored by Digit (PCA)")

#manifold learning
# MDS embedding of the dataset
n_clf = 10000
time_start = time.time()
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
clf_results = clf.fit_transform(train_feature[:n_clf])
print ('MDS embedding done! Time elapsed: {} seconds'.format(time.time()-time_start))

x_clf = clf_results[:,0]
y_clf = clf_results[:,1]

clf_data = {"x_clf": x_clf, "y_clf": y_clf, "label": train_label[:n_clf]}
df_clf = pd.DataFrame(data = clf_data)
df_clf['label'] = df_clf['label'].apply(lambda i: str(i))

g = ggplot( df_clf.loc[:n_clf,:], aes(x='x_clf', y='y_clf', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("MDS embedding dimensions colored by digit")
g.save('mds.png')

#ISOMAP projection
n_iso = 10000
time_start = time.time()
iso = manifold.Isomap(n_neighbors = 30, n_components=2)
iso_results = iso.fit_transform(train_feature[:n_iso])
print ('ISOMAP projection done! Time elapsed: {} seconds'.format(time.time()-time_start))

x_iso = iso_results[:,0]
y_iso = iso_results[:,1]

iso_data = {"x_iso": x_iso, "y_iso": y_iso, "label": train_label[:n_iso]}
df_iso = pd.DataFrame(data = iso_data)
df_iso['label'] = df_iso['label'].apply(lambda i: str(i))

g = ggplot( df_iso.loc[:n_iso,:], aes(x='x_iso', y='y_iso', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("ISOMAP projection dimensions colored by digit")
g.save('isomap.png')

#Locally linear embedding
n_lle = 10000
time_start = time.time()
lle = manifold.LocallyLinearEmbedding(n_neighbors = 30, n_components=2, method='standard')
lle_results = lle.fit_transform(train_feature[:n_lle])
print ('Locally linear embedding done! Time elapsed: {} seconds'.format(time.time()-time_start))

x_lle = lle_results[:,0]
y_lle = lle_results[:,1]

lle_data = {"x_lle": x_lle, "y_lle": y_lle, "label": train_label[:n_lle]}
df_lle = pd.DataFrame(data = lle_data)
df_lle['label'] = df_lle['label'].apply(lambda i: str(i))

g = ggplot( df_lle.loc[:n_lle,:], aes(x='x_lle', y='y_lle', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("Locally linear embedding dimensions colored by digit")
g.save('lle.png')

#HLLE embedding
n_hlle = 10000
time_start = time.time()
hlle = manifold.LocallyLinearEmbedding(n_neighbors = 30, n_components=2, method='hessian')
hlle_results = hlle.fit_transform(train_feature[:n_hlle])
print ('Locally linear embedding done! Time elapsed: {} seconds'.format(time.time()-time_start))

x_hlle = hlle_results[:,0]
y_hlle = hlle_results[:,1]

hlle_data = {"x_hlle": x_hlle, "y_hlle": y_hlle, "label": train_label[:n_hlle]}
df_hlle = pd.DataFrame(data = hlle_data)
df_hlle['label'] = df_hlle['label'].apply(lambda i: str(i))

g = ggplot( df_hlle.loc[:n_hlle,:], aes(x='x_hlle', y='y_hlle', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("Hessian LLE embedding dimensions colored by digit")
g.save('hlle.png')

#Spectral embedding
n_spem = 10000
time_start = time.time()
spem = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
spem_results = spem.fit_transform(train_feature[:n_spem])
print ('Spectral embedding done! Time elapsed: {} seconds'.format(time.time()-time_start))

x_spem = spem_results[:,0]
y_spem = spem_results[:,1]

spem_data = {"x_spem": x_spem, "y_spem": y_spem, "label": train_label[:n_spem]}
df_spem = pd.DataFrame(data = spem_data)
df_spem['label'] = df_spem['label'].apply(lambda i: str(i))

g = ggplot( df_spem.loc[:n_spem,:], aes(x='x_spem', y='y_spem', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("Spectral embedding dimensions colored by digit")
g.save('spem.png')

#LTSA embedding
n_ltsa = 5000
time_start = time.time()
ltsa = manifold.LocallyLinearEmbedding(n_neighbors = 50, n_components=2, method='ltsa')
ltsa_results = ltsa.fit_transform(train_feature[n_ltsa:2*n_ltsa])
print ('LTSA embedding done! Time elapsed: {} seconds'.format(time.time()-time_start))

x_ltsa = ltsa_results[:,0]
y_ltsa = ltsa_results[:,1]

ltsa_data = {"x_ltsa": x_ltsa, "y_ltsa": y_ltsa, "label": train_label[:n_ltsa]}
df_ltsa = pd.DataFrame(data = ltsa_data)
df_ltsa['label'] = df_ltsa['label'].apply(lambda i: str(i))

g = ggplot( df_ltsa.loc[:n_ltsa,:], aes(x='x_ltsa', y='y_ltsa', color='label') ) \
		+ geom_point(size=75,alpha=0.8) \
		+ ggtitle("LTSA embedding dimensions colored by digit")
g.save('lsta.png')


#svm
time_start = time.time()
clf = SVC(C = 10, gamma = 0.001, kernel = 'rbf')
clf.fit(train_feature, train_label)

print ('training done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
y_preds = clf.predict(test_feature)

print ('prediction done ! Time elapsed: {} seconds'.format(time.time()-time_start))

score = accuracy_score(test_label, y_preds)
print ("The accuracy score on test set is: ", score)

pca_100 = PCA(n_components=512)
total_features = np. concatenate((train_feature,test_feature), axis =0)
pca_result_100 = pca_100.fit_transform(total_features)

print ('Explained variation per principal component (PCA): {}'.format(np.sum(pca_100.explained_variance_ratio_)))

ime_start = time.time()
clf = SVC(C = 10, gamma = 0.001, kernel = 'rbf')
# training must be only on training set features
clf.fit(pca_result_100[:60000,:], train_label)

print ('training done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
y_preds = clf.predict(pca_result_100[60000:,:])

print ('prediction done ! Time elapsed: {} seconds'.format(time.time()-time_start))

score = accuracy_score(test_label, y_preds)
print ("The accuracy score on test set is: ", score)

#logistic regression
ime_start = time.time()
clf = LogisticRegression()
# training must be only on training set features
clf.fit(train_feature, train_label)

print ('training done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
y_preds = clf.predict(test_feature)

print ('prediction done ! Time elapsed: {} seconds'.format(time.time()-time_start))

score = accuracy_score(test_label, y_preds)
print ("The accuracy score on test set is: ", score)

#PCA
ime_start = time.time()
clf = LogisticRegression()
# training must be only on training set features
clf.fit(pca_result_100[:60000,:], train_label)

print ('training done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
y_preds = clf.predict(pca_result_100[60000:,:])

print ('prediction done ! Time elapsed: {} seconds'.format(time.time()-time_start))

score = accuracy_score(test_label, y_preds)
print ("The accuracy score on test set is: ", score)

#Random Forest
ime_start = time.time()
clf = RandomForestClassifier(n_estimators= 50, max_depth=15, min_samples_split=110)
# training must be only on training set features
clf.fit(train_feature, train_label)

print ('training done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
y_preds = clf.predict(test_feature)

print ('prediction done ! Time elapsed: {} seconds'.format(time.time()-time_start))

score = accuracy_score(test_label, y_preds)
print ("The accuracy score on test set is: ", score)

#PCA
ime_start = time.time()
clf = RandomForestClassifier(n_estimators= 50, max_depth=15, min_samples_split=110)
# training must be only on training set features
clf.fit(pca_result_100[:60000,:], train_label)

print ('training done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
y_preds = clf.predict(pca_result_100[60000:,:])

print ('prediction done ! Time elapsed: {} seconds'.format(time.time()-time_start))

score = accuracy_score(test_label, y_preds)
print ("The accuracy score on test set is: ", score)