# llmBased_vulnPrediction
Prediction of vulnerable software components by fine-tuning Transformer-based pre-trained Large Language Models.

### Replication Package of our research work entitled "Vulnerability prediction using pre-trained models: An empirical evaluation"

To replicate the analysis and reproduce the results:

~~~
git clone https://github.com/certh-ai-and-softeng-group/llmBased_vulnPrediction.git
~~~
and navigate to the cloned repository.

The "data" directory contains the data required for training and evaluating the models.

The dataset.csv file in the repository is the pre-processed "sequences of tokens" format of the dataset.

The py_corpus.txt file contains the python corpus utilized to train the word2vec embedding vectors.

The jupyter notebook files (.ipynb) in the "ml" folder are python files, which perform the whole analysis. Specifically:

• pytorch/ptBert performs fine-tuning of the BERT and CodeBERT models to the downstream task of vulnerability prediction

• pytorch/ptGpt performs fine-tuning of the GPT-2 and CodeGPT-2 models to the downstream task of vulnerability prediction

• pytorch/ptT5 performs fine-tuning of the T5 and CodeT5 models to the downstream task of vulnerability prediction

• bow/bow_simple trains ML models based on the Bag of Words representation format

• bow/bow_tf_idf trains ML models based on the term frequency–inverse document frequency (tf-idf) representation format

• gensim/train_embeddings trains custom word embedding vectors using word2vec

• gensim/embs_to_dl trains an LSTM model on the word2vec encoded sequences of tokens


### Acknowledgements

Special thanks to HuggingFace for providing the transformers libary.

Special thanks to Gensim for providing word embedding models.

Special thanks to the paper entitled "A Comparison of Different Source Code Representation Methods for Vulnerability Prediction in Python" for providing the original version of the dataset, which we enriched. For the original dataset cite:

~~~
@inproceedings{bagheri2021comparison,
  title={A comparison of different source code representation methods for vulnerability prediction in python},
  author={Bagheri, Amirreza and Heged{\H{u}}s, P{\'e}ter},
  booktitle={Quality of Information and Communications Technology: 14th International Conference, QUATIC 2021, Algarve, Portugal, September 8--11, 2021, Proceedings 14},
  pages={267--281},
  year={2021},
  organization={Springer}
}
~~~

### Appendix

Evaluation results of the overall analysis
| **Model**   | **Accuracy (%)** | **Precision (%)** | **Recall (%)** | **F1-score (%)** | **F2-score (%)** |
|-------------|------------------|-------------------|----------------|------------------|------------------|
| BoW         | 89.7             | 96.7              | 59.0           | 73.2             | 63.9             |
| TF-IDF      | 90.2             | 98.3              | 60.0           | 74.5             | 65.0             |
| Word2vec    | 85.2             | 69.3              | 68.0           | 68.7             | 68.2             |
| **BERT**    | **93.0**         | **90.8**          | **79.0**       | **84.5**         | **81.1**         |
| GPT-2       | 87.3             | 77.0              | 67.0           | 71.6             | 68.8             |
| T5          | 91.1             | 88.9              | 72.0           | 79.5             | 74.8             |
| CodeBERT    | 90.9             | 78.1              | 86.1           | 81.9             | **84.3**         |
| CodeGPT-2   | 88.7             | 81.1              | 68.9           | 74.5             | 71.1             |
| CodeT5      | 90.9             | 86.0              | 74.0           | 79.5             | 76.1             |


### Licence

[MIT License](https://github.com/certh-ai-and-softeng-group/llmBased_vulnPrediction/blob/main/LICENSE)

### Citation

I. Kalouptsoglou, M. Siavvas, A. Ampatzoglou, D. Kehagias, A. Chatzigeorgiou, Vulnerability prediction using pre-trained models: An empirical evaluation, in: 32nd International Symposium on the Modeling, Analysis, and Simulation of Computer and Telecommunication Systems (EuroCyberSec2024 Workshop), Poland, 2024
