import pandas as pd
import numpy as np
from sklearn import linear_model
import random
import math

def loadFile(dataFile):
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
    ds = pd.read_csv(dataFile, dtype=dtype_dict)
    return ds

def get_numpy_data(df, features, output):
    df['constant'] = 1 # add a constant column 

    # prepend variable 'constant' to the features list
    features = ['constant'] + features

    # Filter by features
    fm = df[features]
    y = df[output]
   
    # convert to numpy matrix/vector whatever...
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
    features_matrix = fm.as_matrix()
    output_array = y.as_matrix()

    return(features_matrix, output_array)

def predict_output(feature_matrix, weights):
    result = feature_matrix.dot(weights.T)
    return result

def normalize_features(feature_matrix):
    norms = []
    (rows, columns) = feature_matrix.shape
    nf_matrix = np.copy(feature_matrix)
    
    for i in range(0, columns):
        norm = np.linalg.norm(feature_matrix[:, i], axis=0)
        nf_matrix[:, i] /= norm
        norms.append(norm)
        
    return (nf_matrix, norms)

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = feature_matrix.dot(weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = feature_matrix[:, i].dot(output - prediction + feature_matrix[:, i] * weights[i])
    
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2.0
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2.0
    else:
        new_weight_i = 0.
    
    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = np.copy(initial_weights)
    converged = False
    
    while not converged:
        old_weights = np.copy(weights)
        for i in range(0, len(initial_weights)):
            weights[i] =  lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)

        # stop when every single delta is less than the tolerance
        max = np.amax(np.absolute(weights - old_weights))
        converged = max < tolerance
    return weights

# Tester
# print(lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)], [2./math.sqrt(13),3./math.sqrt(10)]]), np.array([1., 1.]), np.array([1., 4.]), 0.1))

# Use the house data
print("\n\nUsing all data and 2 features; sqft_living & bedrooms")
house_data = loadFile('kc_house_data.csv')

features  = ['sqft_living', 'bedrooms']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(house_data, features, my_output)
(nf_matrix, norms) = normalize_features(feature_matrix)


initial_weights = np.array([1.0, 4.0, 1.0])

# Predict prices
yhat = nf_matrix.dot(initial_weights)

ro = [None, None, None]
for i in range(0, len(initial_weights)):
    if i==0:
        ro[i] = initial_weights[i]
    else:
        ro[i] = nf_matrix[:, i].dot(output - yhat + nf_matrix[:, i] * initial_weights[i])


# Cyclical coordinate descent
initial_weights = np.array([0.0, 0.0, 0.0])
l1_penalty = 1e7
tolerance = 1.0

weights = lasso_cyclical_coordinate_descent(nf_matrix, output, initial_weights, l1_penalty, tolerance)

# Predict prices
yhat = nf_matrix.dot(weights)

rss = np.sum((yhat - output) ** 2)
print("Rss: ", rss)
print("Weights: ", weights)

print("\n\nEvluating lasso with more features")
train_data = loadFile('kc_house_train_data.csv')
features  = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, features, my_output)
(nf_matrix, norms) = normalize_features(feature_matrix)

initial_weights = np.zeros(len(features) + 1)
l1_penalty = 1e7
tolerance = 1.0

weights1e7 = lasso_cyclical_coordinate_descent(nf_matrix, output, initial_weights, l1_penalty, tolerance)
print("weights1e7: ", weights1e7)
weights1e7_normalized = weights1e7 / norms
print(weights1e7_normalized[3])

l1_penalty = 1e8
weights1e8 = lasso_cyclical_coordinate_descent(nf_matrix, output, initial_weights, l1_penalty, tolerance)
print("weights1e8: ", weights1e8)
weights1e8_normalized = weights1e8 / norms

l1_penalty = 1e4
tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(nf_matrix, output, initial_weights, l1_penalty, tolerance)
print("weights1e4: ", weights1e4)
weights1e4_normalized = weights1e4 / norms

# Using test data
print("\n\nUsing test data...")
test_data = loadFile('kc_house_test_data.csv')
(feature_matrix, output) = get_numpy_data(test_data, features, my_output)

print("Using weights1e7_normalized")
yhat = feature_matrix.dot(weights1e7_normalized)
rss = np.sum((yhat - output) ** 2)
print("Rss: ", rss)
 
print("Using weights1e8_normalized")
yhat = feature_matrix.dot(weights1e8_normalized)
rss = np.sum((yhat - output) ** 2)
print("Rss: ", rss)

print("Using weights1e4_normalized")
yhat = feature_matrix.dot(weights1e4_normalized)
rss = np.sum((yhat - output) ** 2)
print("Rss: ", rss)