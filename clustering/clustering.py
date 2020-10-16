import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', names=['id', 'image_id', 'center_x', 'center_y', 'area', 'pose_id'])
df.head()

x = df.drop(['id', 'image_id'], axis=1)
y = df['id'].values

x = StandardScaler().fit_transform(x)

features = ['center_x', 'center_y', 'area', 'pose_id']
pd.DataFrame(x, columns=features).head()

pca = PCA(n_components=3)
principalComponent = pca.fit_transform(x)
principalDF = pd.DataFrame(data=principalComponent, columns=['component1', 'component2', 'component3'])

print(principalDF.head())
print(sum(pca.explained_variance_ratio_))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_zlabel('Component 3', fontsize=15)
ax.set_title('3 component PCA', fontsize=20)

# targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets, colors):
#     indicesToKeep = principalDF['target'] == target
#     ax.scatter(principalDF.loc[indicesToKeep, 'principal component1']
#                , principalDF.loc[indicesToKeep, 'principal component2']
#                , c=color
#                , s=50)

# ax.legend(targets)


ax.scatter(principalDF['component1'], principalDF['component2'], principalDF['component3'], marker='o')

ax.grid()
plt.show()