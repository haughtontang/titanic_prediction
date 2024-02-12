import numpy as np
import math


def sigmoid(z):
    """
    Compute the sigmoid of z
    - This function serves as the model, g will be a number between 0 and 1
    """
    g = 1 / (1 + np.exp(-z))
    return g


def compute_cost(X, y, w, b):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
    Returns:
      total_cost : (scalar) cost
    """
    m, n = X.shape
    total_cost = 0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        fwb_i = sigmoid(z_i)
        total_cost += -y[i]*np.log(fwb_i) - (1-y[i])*np.log(1-fwb_i)
    return total_cost / m


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        # Calculate f_wb (exactly as you did in the compute_cost function above)
        z_i = np.dot(X[i], w) + b
        f_wb = sigmoid(z_i)

        # Calculate the  gradient for b from this example
        dj_db_i = f_wb - y[i]

        # add that to dj_db
        dj_db += dj_db_i
        for j in range(n):
            dj_dw_ij = dj_db_i * X[i, j]
            dj_dw[j] += dj_dw_ij

    # divide dj_db and dj_dw by total number of examples
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant

    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    # number of training examples
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient(X, y, w_in, b_in)
        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = compute_cost(X, y, w_in, b_in)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing


def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb = sigmoid(z_i)
        # TODO These thresholds are placeholders for now - will update in the later version if needed
        pred = 1 if f_wb >= 0.5 else 0
        p[i] = pred
    return p
