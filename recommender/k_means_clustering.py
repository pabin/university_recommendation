import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd

colors = 10*["g", "r", "c", "b", "k", "m", "y", "w"]
markers = ['x', 'v', '+', '*', 's', '2', '8', 'D', 'p', 'h']

csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                   "database/real_data.csv"

names = ['University', 'acceptance_rate', 'average_gre', 'min_gre', 'max_gre', 'average_gre_verbal', 
        'min_gre_verbal', 'max_gre_verbal', 'average_gre_quant', 'min_gre_quant', 'max_gre_quant', 
        'average_gpa', 'min_gpa', 'max_gpa', 'average_writing', 'min_writing', 'max_writing']

df = pd.read_csv(csv_filepathname, names=names, header=None)
#df.fillna(0, inplace=True)
df1 = df.dropna()
#print df1

df1.convert_objects(convert_numeric=True)

def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]


        if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
            column_contents =df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df 

df1 = handle_non_numeric_data(df1)
df1 = df1.drop(['University'], 1)
#df1 = df1.head(5)
print df1

X = np.array(df1.astype(float))
plt.scatter(X[:, 1], X[:, 10], s=150)
plt.xlim(xmin=260, xmax=341)
plt.ylim(ymin=1.5, ymax=4.1)
plt.xlabel('Average GRE Score')
plt.ylabel('Average GPA')
plt.title('Cluster Analysis Before Clustering')
plt.show() 

class k_means:
    def __init__(self, k=10, tol=0.001, max_iter=300):
        self.k = k
        self.tol =tol
        self.max_iter = max_iter

    def fit(self, data):
        # pick starting centers
        #self.centroids = np.array(data[i] for i in range (self.k)])

        self.centroids = {}
        prev_centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]
            prev_centroids[i] = data[i]

        #print self.centroids
        #print "---------"
        #print prev_centroids
        # if true
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            # cycle through known data and assign to class it is closest to
            for featureset in data:
                #compare distance to either centroid
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in
                               self.centroids]
                #print distances
                classification = distances.index(min(distances))
                #print classification
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)

            #print prev_centroids

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        #compare distance to either centroid
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))  
        return classification

clf = k_means()
clf.fit(X)
#print ('-------')

for centroid in clf.centroids:
    #print clf.centroids
    plt.scatter(clf.centroids[centroid][1], clf.centroids[centroid][10],
                marker='o', color='k', s=150, linewidths=5)
    #print "-------------"
for classification in clf.classifications:
    #print clf.classifications
    color = colors[classification]
    marker = markers[classification]
    #print "-------------"
    for featureset in clf.classifications[classification]:
        #print featureset
        plt.scatter(featureset[1], featureset[10], marker=marker, color=color, s=150, linewidths=5) 

plt.xlim(xmin=260, xmax=341)
plt.ylim(ymin=1.5, ymax=4.1)
plt.xlabel('Average GRE Score')
plt.ylabel('Average GPA')
plt.title('Clusters Based on Average GRE and Average GPA for Different Universities')
#plt.show()    
