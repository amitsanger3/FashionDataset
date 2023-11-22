# Semi-Supervised Tweets & News Dataset

Fashion dataset have 3 types of dataset. First is the tweets dataset, second labelled news dataset and third is the un-labeled news dataset.



## License

[MIT](https://choosealicense.com/licenses/mit/)



## Authors

- [@amitsanger3](https://www.github.com/amitsanger3)


## Cite 

If you are using this dataset, please Cite us.

```bash
@article{amitsanger2023contrastiveTfIdf,
    author = {Amit Kumar Sanger},
    title = {CONTRASTIVE TF-IDF VECTORIZATION FOR IMPROVED CLASSIFICATION OF FASHION-RELATED TWEETS: A CROSS-DOMAIN TRANSFER LEARNING SEMI-SUPERVISED LEARNING APPROACH},
    journal = {IRJMETS},
    volume = {5},
    year = {2023},
    number = {152},
    pages = {2582--5208}
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

