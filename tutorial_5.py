# 5. 시험용 데이터로 신경망 검사하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 학습용 데이터셋을 2회 반복하여 신경망을 학습시켰는데요, 신경망이 전혀 배운게
# 없을지도 모르니 확인해보겠습니다.
#
# 신경망이 예측한 정답과 진짜 정답(Ground-truth)을 비교하는 방식으로 확인할텐데요,
# 예측이 맞다면 샘플을 '맞은 예측값(Correct predictions)'에 넣겠습니다.
#
# 먼저 시험용 데이터를 좀 보겠습니다.

dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지 출력
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 좋습니다, 이제 신경망이 어떻게 예측했는지를 보죠:

outputs = net(Variable(images))

########################################################################
# 출력은 10개 분류 각각에 대한 값으로 나타납니다. 어떤 분류에 대해서 더 높은 값이
# 나타난다는 것은, 신경망이 그 이미지가 더 해당 분류에 가깝다고 생각한다는 것입니다.
# 따라서, 가장 높은 값을 갖는 인덱스(index)를 뽑아보겠습니다:
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

########################################################################
# 결과가 괜찮아보이네요.
#
# 그럼 전체 데이터셋에 대해서는 어떻게 동작하는지 보겠습니다.

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

########################################################################
# (10가지 분류에서 무작위로) 찍었을 때의 정확도인 10% 보다는 나아보입니다.
# 신경망이 뭔가 배우긴 한 것 같네요.
#
# 그럼 어떤 것들을 더 잘 분류하고, 어떤 것들을 더 못했는지 알아보겠습니다:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))