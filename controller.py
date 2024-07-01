from models.classificationModel.trainingClassificationModel import trainingClassificationModel
from models.classificationModel.validatingClassificationModel import validatingClassificationModel

class controllerService:

    def __init__(self, test):
        self.test = test

    def runTrainingClassificationModel(self):
        output = trainingClassificationModel(self)
        return output

    def runValidatingClassificationModel(self):
        output = validatingClassificationModel(self)
        return output