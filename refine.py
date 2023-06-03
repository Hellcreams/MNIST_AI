from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random


def xavier_init(size):
    """Xavier 초기화를 수행하는 함수"""
    in_dim = size[0]
    xavier_stddev = 1.0 / np.sqrt(in_dim / 2.0)
    return np.random.randn(*size) * xavier_stddev


class MLP:
    def __init__(self, hidden_units, input_dim, output_dim, learning_rate, bias1, bias2):
        # Argument
        self.hidden_units = hidden_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.W1 = xavier_init((self.input_dim, self.hidden_units))
        self.b1 = bias1
        self.W2 = xavier_init((self.hidden_units, self.output_dim))
        self.b2 = bias2

        # Layer

        # 입력층 -> 은닉층 순전파
        # maximum -> 0보다 작으면 0, 0보다 크면 값이 그대로 (ReLu 함수 반영)
        # Sigmoid : 미분계수에 의해 ReLu보다 반영이 느림 -> 훈련 속도가 느려짐
        hidden_layer = np.maximum(0, np.dot(X, self.W1) + self.b1)

    def for_prop(self, X):
        # 입력층 -> 은닉층 순전파
        # maximum -> 0보다 작으면 0, 0보다 크면 값이 그대로 (ReLu 함수 반영)
        # Sigmoid : 미분계수에 의해 ReLu보다 반영이 느림 -> 훈련 속도가 느려짐
        hidden_layer = np.maximum(0, np.dot(X, self.W1) + self.b1)

        # 은닉층 -> 출력층 순전파
        # softmax의 exponent 한 값을 구하고, 그 다음에 전체 값으로 나눔.
        output_layer = np.exp(np.dot(hidden_layer, self.W2) + self.b2)
        output_probs = output_layer / np.sum(output_layer, axis=1, keepdims=True)

        return output_probs

    def back_prop(self, Y):
        y_one_hot = np.eye(self.output_dim)[y_train]

        delta_output = Y - y_one_hot
        grad_W2 = np.dot(hidden_layer.T, delta_output) / len(X)
        grad_b2 = np.mean(delta_output, axis=0)