class Centroid:
    def __init__(self, x, y, name =  None):
        self.x = x
        self.y = y 
        self.name = name
        
    def calculate_distance(self, point):
        from sklearn.metrics.pairwise import euclidean_distances

        return euclidean_distances(X = point, Y = [[self.x, self.y]])

class ClusterDemonstrator: 
    def __init__(self, centers, centroids, n_samples = 20, cluster_std = 0.6, df = None):
        from sklearn.datasets.samples_generator import make_blobs
        
        if df == None:
            self.df = pd.DataFrame(make_blobs(n_samples = n_samples, 
                                              centers = centers,
                                              cluster_std = cluster_std)[0], 
                                   columns = ['x', 'y'])
        else:
            self.df = df
        
        xmin, xmax = min(self.df['x']), max(self.df['x'])
        ymin, ymax = min(self.df['y']), max(self.df['y'])
        
        self.centroids = []
        for i in range(centroids):
            self.centroids.append(Centroid(x = np.random.randint(xmin, xmax), 
                                           y = np.random.randint(ymin, ymax), 
                                           name = 'c_{}'.format(i)))
                                  
        self.df['cluster'] = np.nan
        self.df['class'] = 'point'
        self.history = None
    def df_centroids(self):
        return(pd.DataFrame({'name':[c.name for c in self.centroids],
                             'x':[c.x for c in self.centroids],
                             'y':[c.y for c in self.centroids]}))
        
        
    def iterate(self, n):
        points =[[x,y] for x,y in zip(self.df['x'], self.df['y'])]
        
        for centroid in self.centroids:
            self.df[centroid.name] = centroid.calculate_distance(points)
        
        self.df['cluster'] = self.df[[c.name for c in self.centroids]].idxmin(axis=1)
        
        for centroid in self.centroids:
            self.df['center_{}'.format(centroid.name)] = str([centroid.x, centroid.y])
            mx = self.df[self.df['cluster'] == centroid.name]['x'].mean()
            my = self.df[self.df['cluster'] == centroid.name]['y'].mean()
            
            if not pd.isnull(mx):
                centroid.x = mx
                centroid.y = my     
            else:
                print('cluster {} has no points!'.format(centroid.name))
                self.centroids.remove(centroid)
                                  
        self.df['iteration'] = n
        
        if n == 0:
            self.history = self.df.copy()
        else:
            self.history = self.history.append(self.df.copy())
        
        for centroid in self.centroids:
            self.history = self.history.append(pd.DataFrame({'x':[centroid.x], 
                                                             'y':[centroid.y], 
                                                             'cluster':[centroid.name], 
                                                             'class':['centroid'],
                                                             'iteration':[n]}))
                                  
        
    def generate(self, max_iterations = 10000, vis = False):
        n = 0
        print('iterating...')
        while n < max_iterations:
            if vis == True:
                plt.scatter(gen1.df['x'], gen1.df['y'])
                plt.scatter([c.x for c in self.centroids], 
                            [c.y for c in self.centroids])
                plt.title(str(n))
                plt.show()
                
            prev = set([(round(c.x), round(c.y)) for c in self.centroids])
            if vis == False:
                print('...{}'.format(n))
            
            self.iterate(n)
            current = set([(round(c.x), round(c.y)) for c in self.centroids])
            
            
            if current == prev:
                break
                
            n += 1
            
        print('done')
        
