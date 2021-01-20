## GCKB: A novel Gated Convolutional Embedding Method for Knowledge Graphs



Source code for paper, GCKB: A novel Gated Convolutional Embedding Method for Knowledge Graphs.

#### Requirements

torch >= 1.6.0



#### Datasets

We used two commonly benchmark datasets for evaluating GCKB.

FB15k-237,

WN18RR.

#### Training and Evaluation

###### Parameters

`--data`: Specify the folder name of the dataset.

`--epochs`: Number of epochs.

`--lr`: Initial learning rate.

`--weight_decay_conv`: L2 regularization.

`--output_folder`: Path of output folder for saving models.

`--batch_size`: Batch size

`--valid_invalid_ratio_conv`: Ratio of valid to invalid triples for training.

`--out_channels`: Number of output channels in convolution layer.

#### Reproducing results

To reproduce the results published in the paper, run preparation script with:

+ FB15k-237

  ```shell
  bash run_fb15k237.sh
  ```

+ WN18RR

  ```shell
  bash run_wn18rr.sh
  ```

  



