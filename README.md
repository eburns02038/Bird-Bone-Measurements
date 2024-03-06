# Bird-Bone-Measurements


# Background info
According to their environments and living habits, birds are classified into different ecological groups. There are 8 ecological groups of birds:
Swimming Birds
Wading Birds
Terrestrial Birds
Raptors
Scansorial Birds
Singing Birds
Cursorial Birds (not included in dataset)
Marine Birds (not included in dataset)

Birds belonging to different ecological groups have different appearances based on their ecological niche  (ex: swimming birds have different physical morphology to soaring raptors). The living habits of birds in different ecological groups are somewhat reflected by differences in  their bone sizes. This dataset allows us to examine the relationship between bone measurements and the ecological groups of different birds.

# Data Analyzed
This dataset consists of 10 measurements from 420 birds. Measurements include:
Length and Diameter of Humerus (upper wing bone)
Length and Diameter of Ulna (lower wing bone)
Length and Diameter of Femur (upper leg bone)
Length and Diameter of Tibiotarsus (middle leg bone)
Length and Diameter of Tarsometatarsus (lower leg bone)

# Project Goals
What questions are we addressing?
Can birds be classified into ecological groups solely based on bone measurements?
Why is this important? What is the benefit?
Direct observation of birds is often very time/money consuming, difficult, and depending on the location potentially dangerous. Additionally, for extinct species direct observation is simply impossible and information gleaned from skeletons and fossils is essential for understanding the species. 
The benefit to using a machine learning model to classify birds into ecological groups is that it allows us to be able to take bone measurement information from any bird and get insight into its living habits without having to observe it directly. 
Being able to use machine learning to gain information on a bird species’ living habits allows researchers to make initial estimates about habitat use, behavior, and evolution when direct observation is challenging or impossible.

# Dataset Exploration
In my initial dataset exploration I noticed there were a few null values in the measurements, and some ‘object’ type data that would need to be addressed in data preprocessing.
Looking at histograms of the data, I noticed that all of the measurement variables had similar distributions, with higher numbers of small measurements and increasingly fewer numbers of larger measurements. This makes sense as the majority of birds are smaller and there are fewer very large birds.
I also noticed that the distribution of the different ecological groups was fairly unbalanced, with some groups having 100 specimens and others having only about 20. While ideally we would want to work with a balanced dataset, this is more reflective of what we would observe in nature where some groups have many more species than others. 
Looking at the correlations between the variables, I saw that all of the variables are consistently very highly correlated with each other, which makes sense, as typically bigger birds have bigger bones overall and smaller birds have smaller bones overall. I also noticed that all of the measurement features are moderately to highly negatively correlated with the ‘type’ variable.
Looking at the scatter plots, I noticed that there looks to be a lot of overlap between measurements for birds of different ecological groups, which may make clustering difficult.

# Data Preprocessing
What did we do to ensure clean data?
To ensure that my data would be as clean as possible before starting any analysis, I evaluated any noticeable outliers in the data, and examined the data for potential errors, and did not find any.
Was there missing or incomplete data?
Next, I checked to see if any data was missing or null. There were 1-3 missing/null values per variable, so we dropped rows with nulls and ended up with 413 of the original 420 birds in the dataset with all of the necessary measurements. 
How did we prepare the data for analysis
Lastly, in preparing the data for analysis, I removed the column with the ID numbers (since the ecological groups had been measured in order, so numbers within a certain range would belong to the same ecological group).
Also, the variable for ‘type’ was a string of abbreviations for the group type name, so I changed the values to numeric (0-5) 

# Dimensionality reduction process overview
The three dimensionality reduction techniques I used were:
PCA
t-SNE
UMAP
Method
The method I followed by testing each of these techniques was to tune the hyperparameters for the model, each time examining a labeled graph to see how the low-dimensional representation looked, and evaluated whether each iteration was showing distinct clusters or not.
Then I took the best version I had gotten for the labeled representation of the data, and make an unlabeled graph of the same data to see if different clusters could be differentiated from each other

# Dimensionality Reduction: PCA
The first technique I used for dimensionality reduction was PCA (Principle Component Analysis. 
PCA transforms the data so that the first component tries to explain the maximum variance from the original data. While this simplifies the data and reduces noise, it does cause some loss of information or oversimplification of the data. 
When using PCA for dimensionality reduction, we see that the colors indicating the true assignments for the groups are laid out in lines, but there is a decent amount of overlap so the groups are sort of visible, but not very well defined when looking at an unlabeled scatter plot of the data

# Dimensionality Reduction: t-SNE
The next technique I looked at for dimensionality reduction was t-SNE. In contrast to PCA, t-SNE preserves local similarities in the data, so we would expect observations belonging to the same class to be similar to each other and therefore grouped together. 
Unfortunately this seemed to work against us for this dataset, as we had a lot of similar measurements for birds of different classes, so the groupings of the similar data points are mostly pretty mixed. 
Looking at the unlabeled graph, we can see that there looks to be some distinct groups, but this is misleading because in reality these groups are the mixed groupings of similar data from different classes. 

# Dimensionality Reduction: UMAP
The third dimensionality reduction technique I tried was UMAP. 
UMAP preserves local similarities like t-SNE, but captures the global structure of the data better than t-SNE. 
Looking at the labeled graph we can see that there are some defined groupings of the same class, but there is still a fair amount of overlap between the groups, and we can see a few instances of multiple separate small groups of the same class. On the unlabeled graph the data looks like two somewhat distinct larger groups, with no clear divisions into smaller groups. 
After looking at the three different methods of dimensionality reduction, it makes some sense that we are not able to get very well distinct clusters, as revisiting the initial scatter plots shos that for almost all of the variables there was a good deal of overlap between the various classes, indicating that there are likely birds of the same general size measurements in each of the six different ecological groups, so we would expect to see overlap like this. 

# Clustering Process overview
The four algorithms that I used for this data were:
K-means
DBSCAN
Hierarchical Clustering
Gaussian Mixture Models (GMM)
For each model I used the PCA transformed data to fit the model, then I would tune the hyperparameters individually until I arrived at a set that would give the best results
Evaluation methods
The evaluation criteria for the models were silhouette score and ARI
Silhouette score
The Silhouette score is the measure of how similar a data point is to its own cluster compared to other clusters. A higher Silhouette score indicates that the data point is more similar to its own cluster and less similar to other clusters. The best score value is 1 and -1 is the worst.
Adjusted Rand Index (ARI)
The Rand Index is the measure of similarity between two clusters. It computes the similarity measure by considering all pairs of samples and counting the number of pairs assigned to the same or different clusters in the true and predicted clusterings. The Adjusted Rand Index (ARI) does the same, but is adjusted for chance. The value returned is between 1, indicating identical clusters, and -1, indicating random labeling. 


# Algorithm 1: k-means
In k-means clustering, the algorithm works to partition the dataset into k distinct, non-overlapping clusters. It does this by determining k centroids in the dataset and assigning each data point to the nearest centroid. 
In the dataset, this algorithm had pretty good ARI and silhouette scores, but visually we can see a good number of places where clusters of points were assigned to the same group when they span multiple true classes. 
While it does have decent scores, one weakness of this model is that k-means assumes that clusters are isotropic (mostly circular) so these long drawn out clusters don’t fit the prerequisite 

# Algorithm 2: DBSCAN
The second algorithm I tried was Density-Based Spatial Clustering of Applications with Noise (DBSCAN). This algorithm uses inputs for the minimum number of points for a region to be considered “dense” and a measurement for determining which other points are close enough to form density-based clusters. If the minimum number of points for a cluster is met for a given point with the distance radius provided, then we consider all of these points to be in the same cluster. 
When this algorithm was applied to the dataset, it had ok ARI and silhouette scores, but we do see that a lot of the points are labeled as noise, especially in the less dense areas of the scatter. While we do see that the algorithm was able to identify the yellow and pink points as clusters, it did label them as being the same class. The high density and lots of overlap in the left side of the scatter likely contributed to this algorithm not performing as well. 

# Algorithm 3: Hierarchical Clustering (complete)
Next I tried hierarchical clustering, in which each point is initially considered to be in its own cluster, and the clusters are merged stepwise until all of the points are in a single cluster. The resulting dendrograms are used to group the points based on which points are in the same “branch” of the hierarchical tree together. 
The first version of the hierarchical clustering I tried used the ‘complete’ linkage, which determines the distance between two clusters as the longest distance between any two points in the two clusters.
From this method I got the highest ARI score so far, but also a low silhouette score compared to the previous models. Looking at the scatter, we see that the yellow and pink classes are mostly distinct from the other assignments, but like in the k-means and DBSCAN models they are still both assigned to the same class. The rest of the assignments seem to be grouped to rough areas, but many assignments for the same group are still spanning across multiple true classes. 

# Algorithm 3: Hierarchical Clustering (ward)
The second version of the hierarchical clustering method that I tried was the ‘ward’ linkage. This linkage determines the distance between two clusters by minimizing the increase in variance when the two clusters are merged, which usually produces clusters of similar shape and size. This doesn’t seem as true for this dataset, as we can see that most of the assigned groups are in oblong shapes.
The scores for this method were decent, but we see slightly lower ARI and silhouette scores compared to the k-means model. This model, like many of the previous models has assigned groups across multiple true classes and multiple instances of more distinct clusters being assigned to the same group when they should be assigned to separate groups. 

# Algorithm 3: Hierarchical Clustering (average)
The third version of the hierarchical clustering method that I tried was the ‘average’ linkage. This linkage determines the distance between two clusters as the average distance between all pairs of points in the two clusters. 
This method had pretty poor ARI and silhouette scores, and looking at the scatter we see that most of the points are assigned to just two groups, so this wouldn’t be good for our uses. 

# Algorithm 3: Hierarchical Clustering (single)
The last hierarchical clustering model I tried used the ‘single’ linkage, which determines the distance between two clusters as the shortest distance between any two points in the clusters. This method is sensitive to outliers and noise in the data, and with our data being very close together and overlapping in some areas, this method determined that all but a few points where in the same cluster, which gave us poor ARI and silhouette scores. The ARI score close to zero tells us that the assignments are no closer to perfect than they are to random, and the negative silhouette score indicates very poor clusters.

# Algorithm 4: Gaussian Mixture Models
The last algorithm that I tried for clustering the data was Gaussian Mixture Models (GMM). This method is a soft clustering technique that determines the probability of a point being in a particular cluster. GMM assumes that there are a certain number of Gaussian distributions and each distribution represents a cluster, so datapoints that belong to a single distribution will be grouped together. 
Looking at the results from our data, the ARI score is similar to the score for k-means and hierarchical clustering ‘complete’ methods, and the silhouette score is also not bad. Some weaknesses to this version are that we can’t really make out distinct shaped clusters within the data like we could on some of the other models, as the groups have a decent amount of overlap.


# Comparing Methods
Looking at the scores for each algorithm, we do not have a single model that had both the highest ARI and silhouette scores. The model with the highest ARI score was Hierarchical Clustering using the 'complete' linkage and the model with the highest Silhouette score was k-means.

Since there was no single algorithm that had both higher ARI and silhouette scores, the "best" algorithm would depend on what we would prioritize.
If we wanted to prioritize correct cluster assignments then the algorithm that would work best would be Hierarchical 'complete' clustering, because that had the highest ARI score.
If we wanted to prioritize tighter and more defined clusters then the algorithm that would work best would be one with a higher silhouette score, in this instance, the K-means.
The GMM model also scored well, as it had a higher ARI score than k-means, and only a slightly lower ARI score compared to the hierarchical clustering but it had a notably higher silhouette score than the hierarchical ‘complete’ clustering.

One point to note is that k-means makes assumptions about the geometry of the clusters, with one of the prerequisites for the data being that clusters are isotropic, which we can see is not true from looking at the clusters we generated via the different dimensionality reduction techniques.

Contrastingly, one of the advantages of GMM is that it does not assume anything about the geometry of the clusters. Additionally, GMM is a soft clustering algorithm. So, we can assess the confidence of the cluster assignments by looking at the probabilities. 

Personally I think that for this data I would want to prioritize the ARI score, which would make the Hierarchical 'complete' Clustering the preferred model. I would choose this because I didn't really expect to get a very high silhouette score, therefore not expecting very distinct clusters, as the exploratory data analysis showed a lot of overlap in measurements, indicating that many birds had similar measurements even when in different ecological groups.



# Conclusions
Revisiting the research question, “Can birds be classified into ecological groups solely based on bone measurements?”, I would say that based on the information I learned using the various machine learning models that using just bone measurements is not enough to generate distinct clusters of ecological groups. 

While the Hierarchical ‘complete’ clustering model that I believe performed the best is able to group the birds somewhat, I think that this is not a strong enough model to confidently group birds into ecological groups based on just the bone measurement data.
While many of the models were ok at grouping the data based on the bone measurements, none of them scored very well in terms of ARI (indicating that the percentage of correct assignments was kind of low) or silhouette score (indicating that there was some clustering but the clusters were poor and not well distinct),. This tells us that there is likely more data required to be able to correctly group birds into the six ecological groups than just individual limb bone measurements. 

One of the reasons why I believe the models did not perform extremely well is that there was a lot of similar data that was in different classes (indicating birds of similar bone measurements in different ecological groups). Revisiting the initial data exploration, we can see that there was a good amount of overlap for measurements for birds of different types, indicating, as mentioned previously, that there are similarly sized birds in different ecological groups. 

While this model would ideally be used for gaining information about a bird’s living habits based on their assignment to a particular ecological group without having to do any direct observation, it is not capable of consistently achieving this. The current version of the model is a good starting point, but could definitely use additional data and more fine tuning before it will be able to accomplish this goal. 

Some future directions I would be interested in pursuing include:
Adding additional variables to the dataset representing the full wingspan and length of leg, and the ratio between the two to see if that can provide additional useful information for the model. 
Finding additional data (for example, diet or geographical distribution) that can be added to the dataset to potentially offset the significant overlaps that I was seeing in just the bone measurements. 


