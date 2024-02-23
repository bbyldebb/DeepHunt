# Getting Started

## Environment
Python 3.9.13, PyTorch 1.12.1, scikit-learn 1.1.2, and DGL 0.9.0 are suggested.

## Dataset
1. Dataset D1 is collected from a simulated e-commerce system based on a microservice architecture. The system under study is deployed on a real cloud environment, and its traffic is consistent with the real business traffic. The system comprises 46 system instances, including 40 microservice instances and 6 virtual machines. Each microservice has its corresponding container monitor, and the deployed virtual machines have associated monitoring data. The failure scenarios in this dataset are derived from real system failures and are replayed in batches. To collect the failure records, operators conducted a failure replay in the system for several days in May 2022. The recorded failures were then labeled with their respective real root cause instances. We open source the raw data and root cause labels of failures for D1 https://anonymous.4open.science/r/Aiops-Dataset-1E0E.
2. Dataset D2 is collected from the management system of a top-tier commercial bank. The system under study comprises 18 system instances, including microservices, web servers, application servers, databases, and dockers. Due to the non-disclosure agreement, we cannot make this dataset publicly available. Two experienced operators examined the failure records from January 2021 to June 2021 and labeled the root cause instances of each failure. The labeling process was conducted separately by each operator, and they cross-checked their labels with each other to ensure consensus. This dataset has been used in the International AIOps Challenge 2022 https://aiops-challenge.com/.


We have preprocessed two raw datasets and placed them in the following folder.

```
D1: DeepHunt/data/D1.zip

D2: DeepHunt/data/D2.zip
```

In the compressed file:

- `graphs_info`: Node_hash.pkl holds a dictionary that records the index of the microservice instances.

- `samples`: There are three files in this directory, all samples (samples.pkl), samples for pre-training GAE (train_samples.pkl), and samples for evaluation (test_samples.pkl). 

  PS. Each sample is a tuple: (timestamp, graphs, features of each node). Graphs indicate the topology of the microservice system generated from call relationships and deployment information; Features of each node are composed of pod metric feats, pod trace feats, pod log feats and node metric feats.

- `cases.csv`: The four items in the table header indicate the failure injection time, failure level, root cause of the failure, and failure type respectively.

## Demo
We provide a demo. 

Firstly, please extract the DeepHunt/data/D1.zip before running the demo:

```
unzip data/D1.zip -d data/
```

Then run:

```
python main.py
```

## Parameter Description in the Demo

Take dataset D1 as an example.

### GAE pre-training

* `num_layers`: The number of layers of the encoder/decoder. (default: 2)
* `in_dim`: The input dimension of GAE. (default: 130)
* `hidden_dim`: The hidden dimension of GAE. (default: 64)
* `out_dim`: The z dimension of GAE. (default: 32)
* `noise_rate`: The ratio of masked features in the noise engineering module. (default: 0.3)
* `epochs`: The training epochs. (default: 1000)
* `batch_size`: The batch size. (default: 64)
* `learning_rate`: The learning rate. (default: 0.001)
* `aug_multiple`: The augmentation multiple. (default: 3)
* `feat_span`: Dimension initiation for different types of features. Only required when selecting ModalLoss function. (default: [(0, 52), (73, 129), (53, 61), (61, 72)] )

### Feedback

* `frozen`: Whether to freeze the parameters of pre-trained GAE when feedback. (default: True)
* `epochs`: The training epochs. (default: 5)
* `batch_size`: The batch size. (default: 16)
* `window_size`: The selected timewindow size for fault analysis. (default: 10)
* `learning_rate`: The learning rate. (default: 0.01)
* `sample_num`: The integer type indicates the number of samples for feedback, and the float type indicates the sample proportion. (default: 0.3)
* `mask_rate`: The probability of randomly masking node features when feedback. (default: 0)

More details can be found in the configuration file: DeepHunt/config/D1.yaml.

