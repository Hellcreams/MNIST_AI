from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random


def xavier_init(size):
    """Xavier 초기화를 수행하는 함수"""
    in_dim = size[0]
    xavier_stddev = 1.0 / np.sqrt(in_dim / 2.0)
    return np.random.randn(*size) * xavier_stddev


# 데이터셋 준비
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 정규화
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

# train / vaild 분리
X_vaild_normalized = X_train_normalized[len(X_train_normalized)*9//10:]
X_train_normalized = X_train_normalized[:len(X_train_normalized)*9//10]
y_vaild = y_train[len(y_train)*9//10:]
y_train = y_train[:len(y_train)*9//10]
# 가중치 초기화
hidden_units = 100
input_dim = 28 * 28
output_dim = 10
learning_rate = 0.1
bias = 0
epoch = 100

# 은닉층 가중치 초기화
W1 = xavier_init((input_dim, hidden_units))
b1 = np.zeros(hidden_units)

# 출력층 가중치 초기화
W2 = xavier_init((hidden_units, output_dim))
b2 = np.zeros(output_dim)


# 입력 데이터를 1차원으로 변환
X = X_train_normalized.reshape(X_train_normalized.shape[0], -1)
error_list = []

for i in range(epoch):
    # 입력층에서 은닉층으로의 순전파
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1 + bias)

    # 은닉층에서 출력층으로의 순전파
    output_layer = np.exp(np.dot(hidden_layer, W2) + b2 + bias)
    output_probs = output_layer / np.sum(output_layer, axis=1, keepdims=True)


    # 실제값을 one-hot 인코딩으로 변환
    y_one_hot = np.eye(output_dim)[y_train]

    # CEE 오차 계산
    loss = -np.mean(np.sum(y_one_hot * np.log(output_probs), axis=1))
    print(i+1, "번째 training의 loss값 :", loss)
    error_list.append(loss)

    # 검증
    X_test = X_vaild_normalized.reshape(X_vaild_normalized.shape[0], -1)
    hidden_layer_test = np.maximum(0, np.dot(X_test, W1) + b1)
    output_layer_test = np.exp(np.dot(hidden_layer_test, W2) + b2)
    output_probs_test = output_layer_test / np.sum(output_layer_test, axis=1, keepdims=True)
    predictions = np.argmax(output_probs_test, axis=1)
    accuracy = np.mean(predictions == y_vaild) * 100
    print("테스트 정확도: {:.2f}%".format(accuracy))

    # 역전파
    delta_output = output_probs - y_one_hot
    grad_W2 = np.dot(hidden_layer.T, delta_output) / len(X)
    grad_b2 = np.mean(delta_output, axis=0)

    delta_hidden = np.dot(delta_output, W2.T) * (hidden_layer > 0)
    grad_W1 = np.dot(X.T, delta_hidden) / len(X)
    grad_b1 = np.mean(delta_hidden, axis=0)

    # 가중치 업데이트
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1


# 오차 그래프 그리기
plt.plot(range(len(error_list)), error_list)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('CEE Error')
plt.show()


# 테스트 데이터 준비
X_test = X_test_normalized.reshape(X_test_normalized.shape[0], -1)

# 순전파
hidden_layer_test = np.maximum(0, np.dot(X_test, W1) + b1)
output_layer_test = np.exp(np.dot(hidden_layer_test, W2) + b2)
output_probs_test = output_layer_test / np.sum(output_layer_test, axis=1, keepdims=True)

# 예측값 계산
predictions = np.argmax(output_probs_test, axis=1)

# 정확도 계산
accuracy = np.mean(predictions == y_test) * 100
print("테스트 정확도: {:.2f}%".format(accuracy))

# 테스트 데이터의 총 개수 출력
num_test_samples = len(X_test)
print("테스트 데이터 개수:", num_test_samples)


while True:
    input("엔터를 누르면 무작위 데이터가 출력됩니다...")
    # 무작위로 테스트 데이터 선택
    random_index = random.randint(0, num_test_samples - 1)
    # 예측값과 실제값 출력
    print("예측값:", predictions[random_index])
    print("실제값:", y_test[random_index])

    # 이미지 시각화
    image = X_test[random_index].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
