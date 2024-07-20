def validatingClassificationModel(self):
    testResult = self.test*20

    # take model from huggingFace
    # read csv from "C:\..\GitHub\ssicsync\models\summaryModel\modelOutputFiles\pdfModelSummaryOutputs.csv"
    # output csv file name as 'pdfModelFinalOutputs.csv' (not xlsx!)
    # Store csv in "C:\..\GitHub\ssicsync\models\classificationModel\modelOutputFiles\pdfModelFinalOutputs.csv"

    # Wee Yang's codes on other model evaluation metrices should be inserted here too!
    # Then combine WY's output and Roy's parsed model output results into a final Excel file:
    # 'C:\..\GitHub\ssicsync\results.xlsx'

    # streamlit's visualisation should be the based on the CSV files, after the model results has been parsed (pdfModelFinalOutputs.csv)!

    return testResult