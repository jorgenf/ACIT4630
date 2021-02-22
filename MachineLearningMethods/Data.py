import pandas

def import_data(file, train_p):
    global training_data, testing_data,training_data_gt, testing_data_gt
    data = pandas.read_csv(file, delimiter=",")
    training_data = data[:round(train_p * len(data))]
    testing_data = data[round(train_p * len(data)):]
    return training_data, testing_data
