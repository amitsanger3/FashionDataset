# Fashion Tweets & News Classification

This project focuses on the classification of fashion-related tweets and news articles using a semi-supervised learning approach. The main goal is to iteratively improve the model by leveraging both labeled and unlabeled datasets. The novelty of this project lies in the methodology to use Semi-supervised learning in fashion tweets & news classification, also this project provide novel labelled dataset in fashion related tweets and news.

## Project Overview
The fashion dataset contains three types of datasets:

1. **Tweets Dataset**: Labeled fashion-related tweets.
2. **Labeled News Dataset**: News headlines with labels indicating whether they are fashion-related.
3. **Unlabeled News Dataset**: A set of news headlines without any labels, used for semi-supervised learning.

The machine learning model trains on the labeled datasets, and the semi-supervised approach allows the model to iteratively predict and label the unlabeled data, improving its accuracy over time.


## Requirements

- Python 3.10
- All required dependencies are listed in the `requirements.txt` file.

To install the dependencies, run:


```
pip install -r requirements.txt
```

## Code Structure

All the source code for the project is placed inside the `.src` directory, with the main file being `training.py`.

### Directory Structure

```bash
  .src/
    |-- __init__.py
    |-- training.py      # Main file for training and semi-supervised learning
```

### Running the Project

To run the training process and the semi-supervised learning loop, execute the following command:

```bash
python .src/training.py
```

This script will:

1. Load the labeled fashion tweets and news datasets.
2. Perform initial training on the labeled data.
3. Iteratively load unlabeled data from the unlabeled directory, predict their labels, and retrain the model.
4. Plot how the accuracy of the model improves over the iterations.

### Dataset

Download the dataset from [google drive](https://drive.google.com/drive/folders/1OzCyJevV8j8eJyILU0b4wvIMQPlIYtcI?usp=sharing) and pass its path in `dataset_dir` variable in `training.py` file.

----------------

# Semi-Supervised Tweets & News Dataset

Fashion dataset have 3 types of dataset. First is the tweets dataset, second labelled news dataset and third is the un-labeled news dataset.



## License

[MIT](https://choosealicense.com/licenses/mit/)



## Authors

- [@amitsanger3](https://www.github.com/amitsanger3)


## Cite 

If you are using this dataset, please Cite us.

```bash
@article{sangercontrastive,
  title={CONTRASTIVE TF-IDF VECTORIZATION FOR IMPROVED CLASSIFICATION OF FASHION-RELATED TWEETS: A CROSS-DOMAIN TRANSFER LEARNING SEMI-SUPERVISED LEARNING APPROACH},
  author={Sanger, Amit Kumar}
}

```


## Structure of the dataset

Fashion dataset have 3 types of dataset. First is the tweets dataset, second labelled news dataset and third is the un-labeled news dataset. You can downlad this data from the [google drive](https://drive.google.com/drive/folders/1OzCyJevV8j8eJyILU0b4wvIMQPlIYtcI?usp=sharing).

#### Structure of the FashionDataset directory

```bash
  FashionDataset 
    |-- fashion_tweets_for_classification.csv
    |-- fashion_news_labeled_data
    |-- un_labeled_news_data
        |-- 2022-11-03_news.csv
        |-- 2022-11-04_news.csv
                ---
        |-- 2023-08-08_news.csv
```


## Acknowledgements

 - [News labelled data is borrowed from Rishabh Misra](https://www.kaggle.com/datasets/rmisra/news-category-dataset)



## Appendix

The news-labelled data is borrowed from 
1. Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).

#
And, I am very thankful to [Rishabh Mishra](https://rishabhmisra.github.io/) for his contribution to the research community. I converted the JSON file of the dataset to CSV and then used “STYLE & BEAUTY” as the target label.



## Contributing

Contributions are always welcome!

Add labels on unlabbeled data or add information for extraction purposes.

Please adhere to this project's `code of conduct`.

