# AMGDTI
![release](https://img.shields.io/badge/release-v1.0-blue)
![open issues](https://img.shields.io/badge/open%20issues-2-brightgreen)
![pull requests](https://img.shields.io/badge/pull%20requests-2%20open-brightgreen)

This repository contains the code for our BIB 2023 Research Track paper: [AMGDTI: drug–target interaction prediction based on adaptive meta-graph learning in heterogeneous network](https://academic.oup.com/bib/article-pdf/25/1/bbad474/54823473/bbad474.pdf)
![Alt](https://github.com/ahu-bioinf-lab/AMGDTI/blob/main/AMGDTI.png)
**Figure 1.** Overviewofthe AMGDTI algorithm,which is divided into three steps. (A) Constructing the heterogeneous networkwithmulti-source biomedical data and employing the Node2Vec algorithm to encode the node representation. (B) Searching for the adaptive meta-graph for the information aggregation of drugs (b1) and protein targets (b2) based on GCN in the heterogeneous network, respectively. (C) Utilizing the inner product of the aggregated feature representation of drugs and proteins to predict potential DTI.

# How to run
## Step 1: Data Preprocessing
Run the `preprocess.py` script to prepare the input heterogeneous network. This step ensures your data is correctly processed for subsequent training and prediction.
## Step 2: Meta-Graph Optimization
Execute the `train_search.py` script to identify the optimal adaptive meta-graph for DTI. This stage involves a search process to determine the meta-graph structure that best suits DTI prediction.
## Step 3: DTI Prediction
Use the `train.py` script to apply the adaptive meta-graph to DTI prediction. This step employs the best adaptive meta-graph from the previous step to make predictions and generate results.
Following these steps in order will help ensure successful replication of the results presented in our manuscript. If you encounter any challenges during execution or need more detailed information, please consult our code documentation and program instructions for guidance on parameter settings and data preparation.

# Environment
```python
python = 3.8 
pytorch = 1.12 
pandas = 1.4.2 
scipy = 1.9.1
```
# Citation
If you find our work helpful in your own research, please consider citing our paper:
```tex
@article{su2024amgdti,
  title={AMGDTI: drug--target interaction prediction based on adaptive meta-graph learning in heterogeneous network},
  author={Su, Yansen and Hu, Zhiyang and Wang, Fei and Bin, Yannan and Zheng, Chunhou and Li, Haitao and Chen, Haowen and Zeng, Xiangxiang},
  journal={Briefings in Bioinformatics},
  volume={25},
  number={1},
  pages={bbad474},
  year={2024},
  publisher={Oxford University Press}
}
```

## Contact

For questions or suggestions, please contact [huzylife@163.com] or open an issue on GitHub.
