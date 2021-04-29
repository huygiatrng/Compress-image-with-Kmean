import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

img = plt.imread("apic.png")

print(img.shape)

width = img.shape[0]
heigh = img.shape[1]

img = img.reshape(width*heigh,3)
kmeans = KMeans(n_clusters=3).fit(img) #change n_clusters to the number of colors your want output picture to have.

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_
print(labels)

img2 = numpy.zeros_like(img)

for i in range(len(img2)):
    img2[i]= clusters[labels[i]]

img2 = img2.reshape(width, heigh, 3)

plt.imshow(img2)
plt.show()