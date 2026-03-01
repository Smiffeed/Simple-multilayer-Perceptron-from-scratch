import pandas as pd
import numpy as np
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(x1, x2, x3, weight_input, weight_hidden, bias_hidden, bias_output):
    y1 = sigmoid(x1 * weight_input[0][0] + x2 * weight_input[0][1] + x3 * weight_input[0][2] + bias_hidden[0])
    y2 = sigmoid(x1 * weight_input[1][0] + x2 * weight_input[1][1] + x3 * weight_input[1][2] + bias_hidden[1])
    y3 = sigmoid(x1 * weight_input[2][0] + x2 * weight_input[2][1] + x3 * weight_input[2][2] + bias_hidden[2])
    y_output = sigmoid(y1 * weight_hidden[0] + y2 * weight_hidden[1] + y3 * weight_hidden[2] + bias_output)
    return y1,y2,y3,y_output

def fit(x1, x2, x3, y, lr, weight_input, weight_hidden, bias_hidden, bias_output):
    list_error = list()
    square_error = float("inf")
    w1_delta, w2_delta, w3_delta = [0,0,0], [0,0,0], [0,0,0]

    # Literation
    for epoch in range(1000):
        print(f"epoch {epoch},\n w_input={weight_input},\n w_hidden={weight_hidden}.")
        list_error = list()
        for yi, xi1, xi2, xi3 in zip(y, x1, x2, x3):
            old_wi, old_w_output = weight_input, weight_hidden
            y_pred = forward_propagation(xi1, xi2, xi3, weight_input, weight_hidden, bias_hidden, bias_output)

            error = yi - y_pred[3] # get error differentiation
            if error == 0:
                continue

            g_output = sigmoid_derivative(y_pred[3]) * error # Calculate the error gradient

            # Back propagation
            # Weights correction of hidden weights and bias
            w1_hidden_delta = lr * y_pred[0] * g_output
            w2_hidden_delta = lr * y_pred[1] * g_output
            w3_hidden_delta = lr * y_pred[2] * g_output
            bias_hidden_delta = lr * (-1) * g_output

            # CAlculate error gradient of hidden layer
            g1 = y_pred[0] * sigmoid_derivative(y_pred[0]) * g_output * weight_hidden[0]
            g2 = y_pred[1] * sigmoid_derivative(y_pred[1]) * g_output * weight_hidden[1]
            g3 = y_pred[2] * sigmoid_derivative(y_pred[2]) * g_output * weight_hidden[2]

            # Weight corrections others
            w1_delta[0] = lr * xi1 * g1
            w1_delta[1] = lr * xi1 * g2
            w1_delta[2] = lr * xi1 * g3
            w2_delta[0] = lr * xi2 * g1
            w2_delta[1] = lr * xi2 * g2
            w2_delta[2] = lr * xi2 * g3
            w3_delta[0] = lr * xi3 * g1
            w3_delta[1] = lr * xi3 * g2
            w3_delta[2] = lr * xi3 * g3

            # correct bias of hidden layer
            bias1_delta = lr * (-1) * g1
            bias2_delta = lr * (-1) * g2
            bias3_delta = lr * (-1) * g3

            # Update weights
            weight_input[0][0] += w1_delta[0]
            weight_input[0][1] += w1_delta[1]
            weight_input[0][2] += w1_delta[2]
            weight_input[1][0] += w2_delta[0]
            weight_input[1][1] += w2_delta[1]
            weight_input[1][2] += w2_delta[2]
            weight_input[2][0] += w3_delta[0]
            weight_input[2][1] += w3_delta[1]
            weight_input[2][2] += w3_delta[2]

            # Update biases
            bias_hidden[0] +=  bias1_delta
            bias_hidden[1] +=  bias2_delta
            bias_hidden[2] +=  bias3_delta

            list_error.append(error)

        oldsquare_error = square_error
        square_error =np.sum(np.square(list_error))
        if square_error >= oldsquare_error:
            return old_wi, old_w_output

    return weight_input, weight_hidden

# Load dataset
dataset = pd.read_csv("bmi.csv")
x1 = np.array(dataset['Height'].values.tolist())
x2 = np.array(dataset['Weight'].values.tolist())
x3 = np.array(dataset['Gender'].values.tolist())
y = np.array(dataset['Index'].values.tolist())

# Normalize Data
x1, x2, y = x1 / 1000, x2 / 100, y/10
x2[np.floor(x2) > 0] /= 10
x3 = np.where(x3 == 'Male', 0.1, x3)
x3 = np.where(x3 == 'Female', 0.2, x3)
x3 = x3.astype(float)

# Initialize weights and bias
weight_input = np.random.uniform(-0.5,0.5,size=(3,3))
weight_hidden = np.random.uniform(-0.5,0.5,size=(3))
bias_hidden = np.random.uniform(-0.5,0.5, size=(3))
bias_output = np.random.uniform(-0.5,0.5)
lr = 0.01

result = fit(x1, x2, x3, y, lr, weight_input, weight_hidden, bias_hidden, bias_output)
gender = 0.1 # 0.1 male and 0.2 female
height = 0.175  # in centimeter
weight = 0.96  # in kg
index = forward_propagation(gender, height, weight, result[0],result[1],bias_hidden, bias_output)

print(f"For a person with height {height*1000} m and weight {weight*100} kg, the predicted BMI category is {index[3]*10:.2f}.")
