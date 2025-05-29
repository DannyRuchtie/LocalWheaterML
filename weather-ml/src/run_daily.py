from fetch import fetch_and_store
from train import train
from predict import make_prediction

if __name__ == "__main__":
    fetch_and_store()
    train()
    make_prediction() 