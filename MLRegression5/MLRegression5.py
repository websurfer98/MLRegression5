import pandas as pd
import numpy as np
from sklearn import linear_model
import random
from math import log, sqrt

dataFile = "kc_house_data.csv"

def loadFile(dataFile):
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
    ds = pd.read_csv(dataFile, dtype=dtype_dict)

    ds['sqft_living_sqrt'] = ds['sqft_living'].apply(sqrt)
    ds['sqft_lot_sqrt'] = ds['sqft_lot'].apply(sqrt)
    ds['bedrooms_square'] = ds['bedrooms']*ds['bedrooms']
    ds['floors_square'] = ds['floors']*ds['floors']

    return ds

def computeLasso(training, alphaVal, validation):
    print("Using an L1 penalty value of: ", alphaVal)

    lm = linear_model.Lasso(alpha=alphaVal, copy_X=True, normalize=True, fit_intercept=True) # set parameters
    all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']
    lm.fit(training[all_features], training['price']) # learn weights
    X = training[all_features]
    coeffs = pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['features', 'estimatedCoefficients'])
    #print(coeffs)

    vald_X = validation[all_features]
    rss = np.sum((validation['price'] - lm.predict(vald_X)) ** 2)
    #print("Rss: ", rss)
    
    return (rss, lm)

# Sales data
print("Using sales data...")
sales = loadFile("kc_house_data.csv")

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

alphaVal = 5e2
model_all = linear_model.Lasso(alpha=alphaVal, copy_X=True, normalize=True, fit_intercept=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights
X = sales[all_features]
coeffs = pd.DataFrame(list(zip(X.columns, model_all.coef_)), columns = ['features', 'estimatedCoefficients'])
print(coeffs)


# Training, validation and test data
print("\n\nUsing training and validation data...")
training = loadFile('wk3_kc_house_train_data.csv')
validation = loadFile('wk3_kc_house_valid_data.csv')

l1_penalty = np.logspace(1, 7, num=13)
print("L1 penalty list", l1_penalty)

bestRss = None
bestPenalty = None
bestLM = None

for i in range(0, len(l1_penalty)):
    (rss, lm) = computeLasso(training, l1_penalty[i], validation)
    if bestRss == None:
            bestRss = rss
            bestPenalty = l1_penalty[i]
            bestLM = lm
    else:
        if rss < bestRss:
            bestRss = rss   
            bestPenalty = l1_penalty[i]
            bestLM = lm

print("Best Rss = ", bestRss)
print("Best penalty = ", bestPenalty)

# Using test data
print("\n\nUsing test data...")
testing = loadFile('wk3_kc_house_test_data.csv')

test_X = testing[all_features]
rss = np.sum((testing['price'] - bestLM.predict(test_X)) ** 2)
print("Rss: ", rss)

# Using the best model
nonZero = np.count_nonzero(bestLM.coef_) + np.count_nonzero(bestLM.intercept_)
print("Number of non-zero coeffs for the best LM: ", nonZero)

# Limiting the number of non-zeroes
max_nonzeros = 7
l1_penalty = np.logspace(1, 4, num=20)
print("L1 penalty list", l1_penalty)

l1_penalty_min = None
l1_penalty_max = None
for i in range(0, len(l1_penalty)):
    (rss, lm) = computeLasso(training, l1_penalty[i], validation)
    nonZero = np.count_nonzero(lm.coef_) + np.count_nonzero(lm.intercept_)
    print("Penalty={0}, non-zero={1}".format(l1_penalty[i], nonZero))

    if l1_penalty_min == None:
        l1_penalty_min = l1_penalty[i]
    else:
        if nonZero > max_nonzeros:
            l1_penalty_min = l1_penalty[i]

    if nonZero < max_nonzeros:
        if l1_penalty_max == None:
            l1_penalty_max = l1_penalty[i]
        elif l1_penalty_max > l1_penalty[i]:
            l1_penalty_max = l1_penalty[i]

print("The penalty limts are {0}, {1}".format(l1_penalty_min, l1_penalty_max))

l1_penalty = np.linspace(l1_penalty_min,l1_penalty_max,20)
print("L1 penalty list", l1_penalty)

bestRss = None
bestPenalty = None
bestLM = None
bestNZ = None

for i in range(0, len(l1_penalty)):
    (rss, lm) = computeLasso(training, l1_penalty[i], validation)
    nonZero = np.count_nonzero(lm.coef_) + np.count_nonzero(lm.intercept_)

    if nonZero == max_nonzeros:
        if bestRss == None:
            bestRss = rss
            bestPenalty = l1_penalty[i]
            bestLM = lm
            bestNZ = nonZero
        elif rss < bestRss:
            bestRss = rss   
            bestPenalty = l1_penalty[i]
            bestLM = lm
            bestNZ = nonZero

print("Best Rss = ", bestRss)
print("Best penalty = ", bestPenalty)
print("Best NZ =", bestNZ)
print("Coeffs for bestLM = ", bestLM.coef_)
