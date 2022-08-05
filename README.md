# DeepHyper-CORDS Project

This repository holds the python script, requirements, and the results from investigating the utility of subset selection in DeepHyper.
As the machine learning field continues to develop, the demand for more computational power and time has risen. The use of subset selection strategies in the context of hyperparameter searches is one potential approach in addressing the computational issue when defining an optimal model. We explored this approach by setting up four search scenarios with slightly differing dataset conditions, and then comparing the accuracy of each scenario’s top performers. We found that subset selections can offer an up to 85.7% reduction in average search time with equivalent accuracy performance to full datasets.

## Computation setup

The project uses the Python programming language with a PyTorch library to construct and run neural network model training. To perform hyperparameter search (HPS) and data subset selection (DSS), the DeepHyper software package and CORDS library \cite{Killamsetty_CORDS_COResets_and_2022} are used respectively. To compare against the Grad Match results from the Decile Team, the project uses the CIFAR10 image database with a Residual Neural Network (ResNet), particularly ResNet-18, to serve as the model. The baseline uses the same default configuration from CORDS, and has hyperparameter variables for the learning rate in logarithmic distribution with the parameters of [0.00001, 0.5], regularization coefficient with a logarithmic distribution of [0.1, 10], momentum between [0.05, 0.95], a learning rate scheduler length between [1, 50] and weight decay between [0.00001, 0.005] with RMSprop as the optimizer. 

There were four different scenarios that were evaluated. The first trained on a full dataset for the full number of epochs, and the second trained on a full dataset with half of the epochs. The third trained on a random subset of the full dataset at for the full number of epochs, and final scenario used Grad-Match to choose a smart subset of the training data for the full number of epochs. The total amount of configurations for each scenario was 50 and the batch size was 32 for all described scenarios. The Random and Grad-Match DataLoaders used 12.5\% of the training data batch with no warm starting. More details about the dataloaders can be found by viewing the script itself. Each search scenario was ran with a full node from Theta-GPU, which comprises of 8 NVIDIA A100 GPUs for up to 2 hours using MPI to coordinate the search. After which, the top 8 configurations were ran for 100 epochs with a batch size of 128 on the full dataset for a fair comparison of each scenario’s configurations. The top performing from each scenario was chosen as the best performing test accuracy. 

## Results

The scenarios generated the following results:

    • Full Data, 50 epochs: 1465.27 avg. config run time, 92.5% highest comparison accuracy
    • Full Data, 25 epochs: 734.45 sec., 93.3%
    • Random:  209.86 sec.,  91.1% 
    • Grad-Match: 262.18 sec., 92.5% 
    
We see that the accuracy after a more complete training is very similar with a range of 2.2%, with the full dataset with the half the number of total epochs performing the best. This was somewhat surprising as we hypothesized that Grad-Match would be the highest performing due to its optimized subsets. Nonetheless, Grad-Match and Random clearly outperformed the full dataset scenarios given the average time spent on an evaluation, with the 50 epoch and full dataset taking 24:25 per configuration run to Random’s 3:29 which provides an 85.7% reduction for similar accuracy. 

More information can be seen in the report.pdf.
