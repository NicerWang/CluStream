# CluStream
A Full Implementation of the [CluStream](http://www.vldb.org/conf/2003/papers/S04P02.pdf) Algorithm (online&offline) based on scikit-learn.

## Usage

* Install

  ```
  pip install -r requirements.txt
  ```

* **Online Part (With Pyramidal Snapshots)**

  ```python
  from online import CluStream
  
  dimension = 5
  # create algorithm
  clustream = CluStream(dimension, time_window=1000, timestamp=0, micro_cluster_cnt=100)
  # create snapshot handler(optional)
  manager = SnapshotManager(clustream=clustream,alpha=2, l=2, path="./clustream", log=True)
  
  # prepare input
  inputs = np.random.random((200, dimension))
  
  # initial fit
  clustream.fit(inputs)
  
  # add data
  for i in range(10000):
      new = np.random.random((dimension,))
      clustream.insert_new(new)
  # stop save snapshot(optional)    
  manager.stop()
  ```

* **Offline Part**

  ```python
  from offline import query
  
  # Query
  result = query(h=1000, n_cluster=10, path="./clustream", cur=int(time.time()));
  ```

  

