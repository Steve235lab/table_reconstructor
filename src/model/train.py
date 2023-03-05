import torch


def valid(epoch, net, test_loarder, writer, device='cuda:0'):
    print("epoch %d 开始验证..." % epoch)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loarder:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net.forward(images, device)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('correct number: ', correct)
        print('totol number:', total)
        acc = 100 * correct / total
        print('第%d个epoch的识别准确率为：%d%%' % (epoch, acc))
        writer.add_scalar('valid_acc', acc, global_step=epoch)

    return correct


def train(epoch, net, criterion, optimizer, train_loader, writer, save_iter=100, device='cuda:0'):
    print("epoch %d 开始训练..." % epoch)
    net.train()
    sum_loss = 0.0
    total = 0
    correct = 0
    # 数据读取
    for i, (inputs, labels) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)
        # print(next(net.parameters()).device, inputs.device)   # 打印模型和输入数据分别位于哪个设备
        outputs = net.forward(inputs, device)
        loss = criterion(outputs, labels)
        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        # 每训练100个batch打印一次平均loss与acc
        sum_loss += loss.item()
        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            # 每跑完一次epoch测试一下准确率
            acc = 100 * correct / total
            print('epoch: %d, batch: %d loss: %.03f, acc: %.04f'
                  % (epoch, i + 1, batch_loss, acc))
            writer.add_scalar('train_loss', batch_loss, global_step=i + len(train_loader) * epoch)
            writer.add_scalar('train_acc', acc, global_step=i + len(train_loader) * epoch)
            for name, layer in net.named_parameters():
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)
            total = 0
            correct = 0
            sum_loss = 0.0
