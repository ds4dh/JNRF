import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
bashCommand = "python scr/evaluation_script.py data/test data/predictions"
os.system(bashCommand)