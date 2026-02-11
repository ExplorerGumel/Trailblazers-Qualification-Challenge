from types import SimpleNamespace
from train_improved import main

args = SimpleNamespace(train=r"C:\Users\Administrator\Downloads\Data\Train.csv", test=None)
results = main(args)
print('RESULTS:', results)
