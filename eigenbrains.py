
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn import svm
from matplotlib import image
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import pairwise_distances

code_file = open('./sample_class.txt', 'r')

def load_data(style='patches'):
    if style=='patches':
        data_dir = './brain/T1_patches_norm_small'
    elif style=='full':
        data_dir = './brain/T1/'
    elif style=='grid':
        data_dir = './brain/grid/grid_3/'
    labels = {}
    all_brains= []
    i=1
    count = 0
    brains = os.listdir(data_dir)
    titles = []
    print 'len patches', len(brains)-1 #.DS store
    for file in brains:
        if file.endswith('.png'):
            full_name = os.path.join(data_dir, file)
            titles.append(file[:7])
            img = Image.open(os.path.join(data_dir, file))
            data = list(img.getdata()) #39676
            all_brains.append(data)
        if style=='full':
            labels = json.load(code_file)
            #img = img.crop((16,19,166,199)) #if its a full brain pic, crop            
    all_brains = np.array(all_brains)
    
    print 'shape  labels', np.shape(labels)
    return all_brains, titles

def train_test_split(data, titles, labels=None):
    titles  =np.array(titles)
    data= np.array(data)
    indices = np.random.permutation(data.shape[0])
    training_idx, test_idx = indices[:int(0.8*data.shape[0])], \
                              indices[int(0.8*data.shape[0]):]
    training, test = data[training_idx,:], data[test_idx,:]
    titles_train, titles_test = titles[training_idx], titles[test_idx]
    if labels:
        labels=np.array(labels)
        train_labels, test_labels = labels[training_idx], labels[test_idx]
        print'Shape of rating train & test data & labels:', np.shape(training),',', \
                np.shape(test), np.shape(train_labels), np.shape(test_labels)
        return training, test, train_labels, test_labels, titles_train, titles_test
    else:
        return training, test, titles_train, titles_test

def classify_svm(data_train, data_test, labels_train, labels_test):
    clf = svm.SVC()
    clf.fit(data_train,labels_train[:,1].T)
    predictions = np.array(list(clf.predict(data_test)))
    score = clf.score(data_test, labels_test)
    print 'correct labels: ',round(100*score, 2),'%'
    return score

def plot_images(images, titles, h=96, w=96, n_row=3, n_col=4, rg=6):
    print 'shape imgs',np.shape(images)
    for i in range(rg):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap = plt.cm.gray)
        plt.title(titles[i], size=8)
    plt.show()
    return

def outlier_detect_forest(X_train, X_test):
    rng = np.random.RandomState(42)
    out_fraction =0.2
    clf = IsolationForest(max_samples=4500, contamination=out_fraction, random_state = rng)
    clf.fit(X_train)
    avg_X = np.average(X_train[:,:20], axis=0) 
    dists = pairwise_distances(X_train[:,:20], avg_X, metric='l2')
    max_dist = np.max(dists, axis=0)
    prob = np.divide(dists, max_dist)
    avg_prob = np.average(prob)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    anomalies_Xtrain=[]
    anomalies_Xtest=[]
    anomalies = []
    normal = []
    for i in range(len(X_train[:,0])):
        if prob[i]>=0.5: #change this to do with bigger than deviation
            anomalies_Xtrain.append(X_train[i,:])
            anomalies.append(i)
        else:
            normal.append(X_train[i,:])
    for j in range(len(X_test[:,0])):
            if prob[i]>=0.5:
                anomalies_Xtest.append(X_test[j,:])
    anomalies_Xtrain = np.array(anomalies_Xtrain)
    anomalies_Xtest = np.array(anomalies_Xtest)
    normal = np.array(normal)
    #print 'anomalies', anomalies
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=-prob)
    #plt.scatter(X_test[:,0], X_test[:,1], c='green')
    #ax.scatter(anomalies_Xtrain[:,0], anomalies_Xtrain[:,1],anomalies_Xtrain[:,2],
    #            c='red')
    #plt.scatter(anomalies_Xtest[:,0], anomalies_Xtest[:,1], c='red')
    plt.show()
    return anomalies

def outlier_detect_svm(X_train, X_test):
    rng = np.random.RandomState(42)
    out_fraction =0.2
    #clf = IsolationForest(max_samples=4500, contamination=out_fraction, random_state = rng)
    clf = svm.OneClassSVM(nu=0.95*out_fraction+0.05, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    avg_X = np.average(X_train[:,:20], axis=0) #this is an average
    dists = pairwise_distances(X_train[:,:20], avg_X, metric='l2')
    max_dist = np.max(dists, axis=0)
    prob = np.divide(dists, max_dist)
    avg_prob = np.average(prob)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    anomalies_Xtrain=[]
    anomalies_Xtest=[]
    anomalies = []
    normal = []
    for i in range(len(X_train[:,0])):
        if prob[i]>=0.5: #change this to do with bigger than deviation
            anomalies_Xtrain.append(X_train[i,:])
            anomalies.append(i)
        else:
            normal.append(X_train[i,:])
    for j in range(len(X_test[:,0])):
            if prob[i]>=0.5:
                anomalies_Xtest.append(X_test[j,:])
    anomalies_Xtrain = np.array(anomalies_Xtrain)
    anomalies_Xtest = np.array(anomalies_Xtest)
    normal = np.array(normal)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=-prob)
    #plt.scatter(X_test[:,0], X_test[:,1], c='green')
    #ax.scatter(anomalies_Xtrain[:,0], anomalies_Xtrain[:,1],anomalies_Xtrain[:,2],
    #            c='red')
    #plt.scatter(anomalies_Xtest[:,0], anomalies_Xtest[:,1], c='red')
    plt.show()
    return anomalies

style='grid'
labels_provided = False
'''labels = json.load(code_file)
label = labels[name]
labels[file] = label'''

brains, titles = load_data(style)
n_samp, n_feat = np.shape(brains)
data_train, data_test, titles_train, titles_test = train_test_split(brains, titles)
var_list = {}
n_components = 20
pca = PCA(n_components, svd_solver='randomized', whiten=True)
pca.fit(data_train)

if style=='patches':
    eigenbrains = pca.components_.reshape((n_components, 96,96))
elif style=='grid':
    eigenbrains = pca.components_.reshape((n_components, 109,91))

train_brains_transf = pca.transform(data_train)
test_brains_transf = pca.transform(data_test)
print 'shape of transformed test brains', np.shape(test_brains_transf)

avg_brain = np.average(test_brains_transf, axis=0) 
for k in range(np.shape(test_brains_transf)[0]):
    a_score = abs(test_brains_transf[k] - avg_brain)

var = pca.explained_variance_ratio_
var_list[n_components]= var
eigentitles = ["eigenbrain %d" % i for i in range(eigenbrains.shape[0])]

print 'explained var with {} components '.format(n_components), \
        100*sum(var_list[n_components]), '%'

if style=='patches':
    plot_images(eigenbrains, eigentitles, h=96, w=96)
elif style=='grid':
    plot_images(eigenbrains, eigentitles, h=109, w=91)
elif style=='full':
    plot_images(eigenbrains, eigentitles, h=218, w=182)

if labels_provided:
    score = classify_svm(data_train, data_test, labels_train, labels_test)

outliers = outlier_detect_forest(train_brains_transf, test_brains_transf)

k=0
print 'shape data_train in the end:',np.shape(data_train)
num_plots_h = int(np.ceil(len(outliers)/4.0))
for outlier in outliers:
    plt.subplot(num_plots_h, 4, k+1)
    plt.imshow(data_train[outlier].reshape((109,91)), cmap = plt.cm.gray)
    plt.title(titles_train[outlier], size=10)
    k+=1
plt.show() 

