import MachineLearningMethods.LinearRegression as LR
import MachineLearningMethods.Data as Data
import MachineLearningMethods.LogisticRegression as logreg

training_data, testing_data = Data.import_data("breast_cancer_dataset.csv", 0.8)
training_data_gt = training_data.iloc[:, 1:2]
testing_data_gt = testing_data.iloc[:,1:2]
del training_data["diagnosis"]
del training_data["id"]
del testing_data["diagnosis"]
del testing_data["id"]
model = LR.iterative_gradient_decent(0.00000001, 1000, (training_data, testing_data, training_data_gt, testing_data_gt))
correct = 0
wrong = 0
for row, gt in zip(testing_data.values, testing_data_gt.values):
    prediction = round(logreg.sigmoid(model[0] + sum(w * x for w, x in zip(model[1:], row))),1)
    print(prediction, gt, True if round(prediction) == round(gt[0]) else False)
    if round(prediction) == round(gt[0]):
        correct += 1
    else:
        wrong += 1
print("Correct: ", correct, "Wrong: ", wrong)