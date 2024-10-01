import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, tweets_file, news_file, unlabeled_dir):
        self.tweets_file = tweets_file
        self.news_file = news_file
        self.unlabeled_dir = unlabeled_dir
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def load_labeled_data(self):
        # Load the labeled datasets
        print("Loading labelled data.")
        tweets_df = pd.read_csv(self.tweets_file)
        news_df = pd.read_csv(self.news_file)

        X_tweets = tweets_df['text'].astype(str)
        y_tweets = tweets_df['useful_for_fashion_industry'].apply(lambda x: 1 if x == 'yes' else 0)

        X_news = news_df['headline'].astype(str)
        y_news = news_df['target']
        print(f"Fashion tweets: {len(X_tweets)}")
        print(f"Fashion News: {len(X_news)}")
        return X_tweets, y_tweets, X_news, y_news

    def load_unlabeled_data(self):
        # Load all unlabeled data files from the specified directory
        all_unlabeled_data = []
        for file in os.listdir(self.unlabeled_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.unlabeled_dir, file))
                all_unlabeled_data.append(df['Extras'].astype(str))

        return all_unlabeled_data

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X):
        return self.vectorizer.transform(X)


class ModelTrainer:
    def __init__(self):
        self.svm_model = SVC(kernel='linear', probability=True, verbose=True)
        self.nb_model = MultinomialNB()
        self.vectorizer = None

    def train_initial_models(self, X_tweets, y_tweets):
        # Train initial models on the labeled tweets
        print("Initial training on SVM of fashion tweets started.")
        self.svm_model.fit(X_tweets, y_tweets)
        print("Initial training on SVM of fashion tweets finished.")

        print("Initial training on Naive bayes of fashion tweets started.")
        self.nb_model.fit(X_tweets, y_tweets)
        print("Initial training on Naive bayes of fashion tweets finished.")

    def train_on_news(self, X_news, y_news):
        # Train models on fashion news (transfer learning)
        print("Transfer learning on SVM with news data started.")
        self.svm_model.fit(X_news, y_news)
        print("Transfer learning on SVM with news data finished.")

        print("Transfer learning on Naive Bayes with news data started.")
        self.nb_model.fit(X_news, y_news)
        print("Transfer learning on Naive Bayes with news data finished.")

    def evaluate_model(self, X, y, model_name="SVM"):
        if model_name == "SVM":
            y_pred = self.svm_model.predict(X)
        else:
            y_pred = self.nb_model.predict(X)

        return accuracy_score(y, y_pred), classification_report(y, y_pred)

    def predict_on_unlabeled(self, X_unlabeled):
        # Predict on the unlabeled data using the SVM model
        return self.svm_model.predict_proba(X_unlabeled)

    def retrain_on_augmented_data(self, X_augmented, y_augmented):
        # Retrain the SVM model on the augmented dataset
        self.svm_model.fit(X_augmented, y_augmented)


class SemiSupervisedTrainer:
    def __init__(self, unlabeled_dir, dataloader, model_trainer, confidence_threshold=0.9):
        self.unlabeled_dir = unlabeled_dir
        self.dataloader = dataloader
        self.model_trainer = model_trainer
        self.confidence_threshold = confidence_threshold
        self.accuracy_progress = []

    def run(self):
        # Load the labeled data
        X_tweets, y_tweets, X_news, y_news = self.dataloader.load_labeled_data()

        # Fit and transform the data using TF-IDF
        X_tweets_tfidf = self.dataloader.fit_transform(X_tweets)
        X_news_tfidf = self.dataloader.transform(X_news)

        # Train initial models on labeled tweets
        self.model_trainer.train_initial_models(X_tweets_tfidf, y_tweets)

        # Transfer learning on news dataset
        self.model_trainer.train_on_news(X_news_tfidf, y_news)

        # Evaluate and store the initial accuracy on tweets
        initial_acc, report = self.model_trainer.evaluate_model(X_tweets_tfidf, y_tweets, model_name="SVM")
        print("Initial Evaluation Report on Fashion Tweets:\n", report)
        self.accuracy_progress.append(initial_acc)

        # Load unlabeled data for semi-supervised learning
        # all_unlabeled_data = self.dataloader.load_unlabeled_data()

        print("Semi-Supervised learning started.")
        # Iteratively perform semi-supervised learning
        X_augmented = pd.DataFrame(X_news_tfidf.toarray())
        y_augmented = pd.Series(y_news)

        # for unlabeled_data in all_unlabeled_data:
        for file in os.listdir(self.unlabeled_dir)[:3]:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.unlabeled_dir, file))
                unlabeled_data = df['Extras'].astype(str)
                X_unlabeled_tfidf = self.dataloader.transform(unlabeled_data[:100])
                unlabeled_predictions = self.model_trainer.predict_on_unlabeled(X_unlabeled_tfidf)

                # Select high-confidence predictions
                high_confidence_indices = (unlabeled_predictions.max(axis=1) >= self.confidence_threshold)
                high_confidence_preds = self.model_trainer.svm_model.predict(X_unlabeled_tfidf[high_confidence_indices])

                # Augment the training data
                X_augmented = pd.concat([X_augmented, pd.DataFrame(X_unlabeled_tfidf[high_confidence_indices].toarray())],
                                        ignore_index=True)
                y_augmented = pd.concat([y_augmented, pd.Series(high_confidence_preds)], ignore_index=True)

                # Retrain the model
                self.model_trainer.retrain_on_augmented_data(X_augmented, y_augmented)

                # Re-evaluate on fashion tweets and log the improvement
                acc, report = self.model_trainer.evaluate_model(X_tweets_tfidf, y_tweets, model_name="SVM")
                print(f"Accuracy after semi-supervised iteration: {acc}")
                self.accuracy_progress.append(acc)

        # Plot the accuracy progress over iterations
        self.plot_accuracy_progress()

    def plot_accuracy_progress(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.accuracy_progress, marker='o', linestyle='-', color='b')
        plt.title('Accuracy Progression during Semi-Supervised Learning')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    dataset_dir = "../Dataset"
    # Usage
    tweets_file = os.path.join(dataset_dir, "fashion_tweets_for_classification.csv")
    news_file = os.path.join(dataset_dir, "fashion_news_labeled_data.csv")
    unlabeled_dir = os.path.join(dataset_dir, "un_labeled_news_data")  # Directory containing unlabeled CSV files

    # Create instances of the classes
    dataloader = DataLoader(tweets_file, news_file, unlabeled_dir)
    model_trainer = ModelTrainer()
    semi_supervised_trainer = SemiSupervisedTrainer(unlabeled_dir, dataloader, model_trainer)

    # Run the semi-supervised training process
    semi_supervised_trainer.run()
