import sys
from controller import controllerService

# hard-coded variables
test = 3

modelTraining = controllerService(test)

def run_modelTraining():
    finalOutput = modelTraining.runTrainingClassificationModel()
    print(finalOutput)

def run_modelResults():
    finalOutput = modelTraining.runValidatingClassificationModel()
    print(finalOutput)

arg = sys.argv[1].lower()
if arg == 'train' or arg == 'training':
    run_modelTraining()
elif arg == 'result' or arg == 'results':
    run_modelResults()