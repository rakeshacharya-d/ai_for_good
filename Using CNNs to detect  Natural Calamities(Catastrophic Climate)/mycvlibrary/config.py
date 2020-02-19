import os

#initializing path

DATASET_PATH = "Cyclone_Wildfire_Flood_Earthquake_Database"

# class labels

CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]

#sizing the training

TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1
TEST_SPLIT = 0.25

# min_lr,max_lr,batch_size,step_size,cyclic_learning_rate,number of epoches

MIN_LR = 1e-6
MAX_LR = 1e-4
BATCH_SIZE = 32
STEP_SIZE =8
CLR_METHOD = "traingular"
NUM_EPOCHES = 48

#seriaized model after training
MODEL_PATH = os.path.sep.join(["output", "natural_disaster.model"])

LRFIND_PLOT_PATH = os.path.sep.join(["output","lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output","training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])
