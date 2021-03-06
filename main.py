import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import numpy

img = plt.imread("input.jpg")

print(img.shape)

width = img.shape[0]
heigh = img.shape[1]

img = img.reshape(width*heigh,3)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters).fit(img) 

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_
print(labels)

img2 = numpy.zeros_like(img)

for i in range(len(img2)):
    img2[i]= clusters[labels[i]]

img2 = img2.reshape(width, heigh, 3)
result = Image.fromarray(img2)
result.save("result.jpg")
