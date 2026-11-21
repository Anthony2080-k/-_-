import torch

# 재현성을 위한 시드 고정
torch.manual_seed(42)

# 300명의 데이터 생성
N = 300

# X1: 공부 시간 (1 ~ 10 사이의 값 300개)
# X2: 수면 시간 (3 ~ 9 사이의 값 300개)
X1 = torch.rand(N) * 9.0 + 1.0  # 1.0 ~ 10.0
X2 = torch.rand(N) * 6.0 + 3.0  # 3.0 ~ 9.0

# 입력 데이터 X 합치기 (Shape: [300, 2])
X = torch.stack([X1, X2], dim=1)

# 실제 정답 규칙: Y = 3.5*X1 + 2.0*X2 + 10.0 + (노이즈)
# (Shape: [300])
noise = torch.randn(N) * 1.0
Y = 3.5 * X1 + 2.0 * X2 + 10.0 + noise

print("X shape:", X.shape)  # torch.Size([300, 2])
print("Y shape:", Y.shape)  # torch.Size([300])


w = torch.tensor([10, 10], dtype = torch.float32, requires_grad=True)
b = torch.tensor([50], dtype = torch.float32, requires_grad=True)
print("w shape:", w.shape)
print("b shape:", b.shape)

predict_Y = X @ w + b

criterion = torch.nn.MSELoss()
loss = criterion(predict_Y, Y)
optimizer = torch.optim.SGD([w, b], lr=0.001)

while loss.item() > 1.05:

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predict_Y = X @ w + b
    loss = criterion(predict_Y, Y)


    print("w: ", w)
    print("w_grad: ", w.grad)
    print("b: ", b)
    print("b_grad: ", b.grad)
    print("loss:", loss.item(),"\n")








