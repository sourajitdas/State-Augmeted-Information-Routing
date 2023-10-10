# State-Augmeted-Information-Routing
The repository contains the source code for policy routing in packet based networks. We leverage the use of unsupervised learning methods to improve the efficiency of packet based routing which is superior to conventional non-learning methods. We use Graph Neural Network parameterization for the routing optimization which provide the classical properties of stability and transferability.

# Abstract
This paper examines the problem of information routing in a large-scale communication network, which can be formulated as a constrained statistical learning problem having access to only local information. We delineate a novel State Augmentation (SA) strategy to maximize the aggregate information at source nodes using graph neural network (GNN) architectures, by deploying graph convolutions over the topological links of the communication network. The proposed technique leverages only the local information available at each node and efficiently routes desired information to the destination nodes. We leverage an unsupervised learning procedure to convert the output of the GNN architecture to optimal information routing strategies. In the experiments, we perform the evaluation on real-time network topologies to validate our algorithms. Numerical simulations depict the improved performance of the proposed method in training a GNN parameterization as compared to baseline algorithms.

# Citation
If you use the repository for your work please use the following BibTex citation to cite the ArXiv paper

@ARTICLE{2023arXiv231000248D,
       author = {{Das}, Sourajit and {NaderiAlizadeh}, Navid and {Ribeiro}, Alejandro},
        title = "{Learning State-Augmented Policies for Information Routing in Communication Networks}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Networking and Internet Architecture, Computer Science - Machine Learning, Electrical Engineering and Systems Science - Signal Processing},
         year = 2023,
        month = sep,
          eid = {arXiv:2310.00248},
        pages = {arXiv:2310.00248},
          doi = {10.48550/arXiv.2310.00248},
archivePrefix = {arXiv},
       eprint = {2310.00248},
 primaryClass = {cs.NI},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231000248D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
