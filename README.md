# AMGDTI
# Step 1: Data Preprocessing
Run the preprocess.py script to prepare the input heterogeneous network. This step ensures your data is correctly processed for subsequent training and prediction.
# Step 2: Meta-Graph Optimization
Execute the train_search.py script to identify the optimal adaptive meta-graph for DTI. This stage involves a search process to determine the meta-graph structure that best suits DTI prediction.
# Step 3: DTI Prediction
Use the train.py script to apply the adaptive meta-graph to DTI prediction. This step employs the best adaptive meta-graph from the previous step to make predictions and generate results.
Following these steps in order will help ensure successful replication of the results presented in our manuscript. If you encounter any challenges during execution or need more detailed information, please consult our code documentation and program instructions for guidance on parameter settings and data preparation.
