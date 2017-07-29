from numpy import *

def computer_error(b, m, points):
    y = m * points[:, 0] + b
    error = (points[:, 1] - y) ** 2
    total_error = sum(error) / len(points)
    return total_error

    # total_error = 0
    # for i in range(len(points)):
    #     x = points[i, 0]
    #     y = points[i, 1]
    #     total_error += (y - (m * x + b)) ** 2
    # total_error = total_error / len(points)
    #
    # return total_error

def step_gradient(current_b, current_m, points, learning_rate):
    x = points[:, 0]
    y = points[:, 1]
    y_hat = current_m * x + current_b
    N = float(len(points))
    b_gradient = sum(-(2/N) * (y - y_hat))
    m_gradient = sum(-(2/N) * x * (y - y_hat))
    new_b = current_b - learning_rate * b_gradient
    new_m = current_m - learning_rate * m_gradient

    # b_gradient = 0
    # m_gradient = 0
    # N = float(len(points))
    # for i in range(len(points)):
    #     x = points[i, 0]
    #     y = points[i, 1]
    #     y_hat = current_m * x + current_b
    #     b_gradient += -(2/N) * (y - y_hat)
    #     m_gradient += -(2/N) * x * (y - y_hat)
    # new_m = current_m - learning_rate * m_gradient
    # new_b = current_b - learning_rate * b_gradient
    return new_b, new_m

def gradient_descent_runner(
        points,
        starting_b,
        starting_m,
        learning_rate,
        num_iterations
):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return b, m

def main():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001 # hyper-parameter
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    b, m = gradient_descent_runner(
        points,
        initial_b,
        initial_m,
        learning_rate,
        num_iterations
    )
    print(b)
    print(m)

if __name__ == "__main__":
    main()
