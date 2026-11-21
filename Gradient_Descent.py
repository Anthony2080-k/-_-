# import numpy as np
# import matplotlib.pyplot as plt
#
# # 입력 데이터 (X: 공부 시간)
# X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
#
# # 실제 정답 데이터 (Y: 시험 점수, 규칙: y = 3x + 2)
# Y = np.array([5.0, 8.0, 11.0, 14.0, 17.0])
#
# # 처음 가정 weight 와 bias
# w, b = 5, 0
#
# # 처음 그래프 생성
# Gredient_predict_Y = w * X + b
# plt.scatter(X, Y, color='red')
# plt.plot(X, Gredient_predict_Y, color='blue')
# plt.show()
#
# #학습
# Learning_Rate = 0.01
# Loss = np.array(Gredient_predict_Y - Y)
# """
# predict_Y = w * X + b
# Y = W * X + B
# Loss = predict_Y - Y = (w-W)*X + b-B
# Loss**2 = (predict_Y - Y)**2 = ((w-W)**2)*(X**2) + 2(b-B)*(w-W)*X + (b-B)**2
# w-W = loss_W
# b-B = loss_B
# Loss**2 = (loss_W**2)*(X**2) + 2*loss_B*loss_W*X + loss_B**2
# ∂(Loss**2)/∂(loss_W) = 2*(X**2)*loss_W + 2*loss_B*X = 2*X*Loss
# ∂(Loss**2)/∂(loss_B) = 2*(loss_W)*X + 2*loss_B = 2*Loss
#
# Next_W = w - Learning_rate * 2*X*Loss
# Next_B = b - Learning_rate * 2*Loss
# """
# train_time = 1
# while np.mean(Loss**2)>0.01:
#     print("train_time:", train_time)
#     w = w - Learning_Rate * np.mean(2 * Loss * X)
#     b = b - Learning_Rate * np.mean(2 * Loss)
#     predict_Y = w * X + b
#     if train_time % 100 == 0:
#         print(train_time, w, b, np.mean(Loss**2))
#         plt.scatter(X, Y, color='red')
#         plt.plot(X, predict_Y, color='blue')
#         plt.show()
#     train_time += 1
#     Loss = np.array(predict_Y - Y)
