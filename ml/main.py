from ml.data.download import download_data
from ml.data.preprocessing import preprocessing
from ml.data.split import split
from ml.training.train import train
from ml.analysis.analysis import analysis 
from ml.evaluation.evaluation import evalutation


def main():
    download_data()
    preprocessing()
    split()
    train()
    analysis()
    evalutation()
