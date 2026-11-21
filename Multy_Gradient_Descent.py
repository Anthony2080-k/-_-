import numpy as np

# 재현성을 위한 시드 고정
np.random.seed(42)

# 300명의 데이터 생성
N = 300

# X1: 공부 시간 (1 ~ 10 사이의 값 300개)
# X2: 수면 시간 (3 ~ 9 사이의 값 300개)
X1 = np.random.uniform(1.0, 10.0, N)
X2 = np.random.uniform(3.0, 9.0, N)

# 다차원 입력 데이터 X 합치기 (Shape: 300행 2열)
X = np.column_stack((X1, X2))

# 실제 정답 규칙: Y = 3.5*X1 + 2.0*X2 + 10.0 + (약간의 노이즈)
# 진짜 정답 W는 [3.5, 2.0], B는 10.0 입니다.
Y = 3.5 * X1 + 2.0 * X2 + 10.0 + np.random.normal(0, 1.0, N)

"""
predict_Y = W1 * X1 + W2 * X2 + B = np.dot([W1, W2],X) + B
Y = w1 * X1 + w2 * X2 + b
Loss = predict_Y - Y = (W1-w1) * X1 + (W2-w2) * X2 + B-b

W1- w1 = loss_W1
W2- w2 = loss_W2
B-b = loss_B
loss**2 = (predict_Y - Y)**2 = (W1-w1)**2 * X1**2 + (W2-w2)**2 * X2**2 + (B-b)**2
                               + 2 * (W1-w1) * (W2-w2) * X1 * X2
                               + 2 * (W1-w1) * (B-b) * X1 + 2 * (W2-w2) * (B-b) * X2
                             = loss_W1 ** 2 * X1 ** 2 + loss_W2 ** 2 * X2 ** 2 + loss_B**2
                               + 2 *loss_W1 * loss_W2 * X1 * X2
                               + 2*loss_W1 * loss_B * X1 + 2*loss_W2 * loss_B * X2

∂(Loss**2)/∂(loss_W1) = 2 * loss_W1 * X1**2 + 2 * loss_W2 * X1 * X2 + 2 * loss_B * X1
                      = 2 * X1 * (loss_W1 * X1 + loss_W2 * X2 +loss_B)
                      = 2 * X1 * Loss

∂(Loss**2)/∂(loss_W2) = 2 * loss_W2 * X2**2 + 2 * loss_W1 * X1 * X2 + 2 * loss_B * X2
                      = 2 * X2 * (loss_W2* X2 + 2* loss_W1 * X1 + loss_B * X1)
                      = 2 * X2 * Loss

∂(Loss**2)/∂(loss_B) = 2 * Loss_B + 2 * loss_W1 * X1 + 2 * loss_W2 * X2
                     = 2 * Loss

 => [∂(Loss**2)/∂(loss_W1), ∂(Loss**2)/∂(loss_W2), ∂(Loss**2)/∂(loss_B)]
    = 2 * loss * [X1, X2, 1]
"""
W = np.array([1.0, 3.0])
b = np.random.uniform(1.0, 10.0)


Multy_predict_Y = np.dot(W,X.T) + b
Loss = Multy_predict_Y - Y

learning_rate = 0.01

# print("W.shape",W.shape)
# print("X.shape",X.shape)
# print("Multy_predict", Multy_predict_Y.shape)
# print("Loss.shape", Loss.shape)

while np.mean(Loss**2) > 1.00059:

    gradient_W = 2 * np.dot(Loss, X) / N
    gradient_b = 2 * np.mean(Loss)

    print("Loss = ",np.mean(Loss ** 2))
    print("W = ", W , "W_N = ", W - learning_rate * gradient_W)
    print("b = ", b,  "b_N = ", b - learning_rate * gradient_b)

    W = W - learning_rate * gradient_W
    b = b - learning_rate * gradient_b


    Multy_predict_Y = np.dot(W,X.T) + b
    Loss = Multy_predict_Y - Y


