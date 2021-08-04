import torch
import torch.nn as tnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from cnn import cnn
from utils import export_sample_images
from pytorchtools import EarlyStopping

N_CLASSES = 2   # male & female
BATCH_SIZE = 20
LEARNING_RATE = 0.001
EPOCH = 120
PATIENCE = 20

# train
if __name__ == "__main__":
    # use gpu0
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    transform_train = transforms.Compose([
        transforms.Resize((200, 200)),
        # data enhancement
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees   = 360,            # rotating
                                translate = (0.35, 0.35),   # translating
                                scale     = (0.8, 1.7)),    # scaling
        #
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.556, 0.445, 0.396 ],
                             std  = [ 0.234, 0.205, 0.185 ])
        ])

    transform_test = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.556, 0.445, 0.396 ],
                             std  = [ 0.234, 0.205, 0.185 ])
        ])
    
    trainData = datasets.ImageFolder('~/data/gender/train/', transform_train)
    testData = datasets.ImageFolder('~/data/gender/test', transform_test)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)
    
    export_sample_images(10, './samples/train/', trainData)
    export_sample_images(10, './samples/test/', testData)

    model = cnn(N_CLASSES)
    model.cuda()

    # Cost function, optimizer & scheduler
    cost = tnn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    '''
    Training
    '''
    train_losses = []
    test_losses  = []
    acc_list     = []
    # early stop
    early_stop = EarlyStopping(patience=PATIENCE, verbose=True)
    for epoch in range(EPOCH):
        # training
        model.train()
        avg_loss = 0
        cnt = 0
        for images, labels in trainLoader:
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = cost(outputs, labels)
            avg_loss += loss.data
            cnt += 1
            print("[E: %d] \tloss: %f  \tavg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
            loss.backward()
            optimizer.step()
        scheduler.step(avg_loss)
        train_losses += [avg_loss / cnt]

        # testing
        model.eval()
        valid_losses = []
        # acc
        correct = 0
        total = 0
        # loss
        avg_loss = 0
        cnt = 0
        for images, labels in testLoader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            # acc
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum()
            # loss
            loss = cost(outputs, labels)
            valid_losses += [loss.item()]
            avg_loss += loss.data
            cnt += 1
            print("Test losses: %f" % (loss.data))

        # acc
        acc_list += [correct / total * 100]
        # loss
        test_losses += [avg_loss / cnt]
        valid_loss = avg_loss / cnt
        # early stop
        early_stop(valid_loss, model)
        # save the model
        if 0 == epoch % 10:
            torch.save(model, 'model.pt')
            torch.save(model.state_dict(), 'model.state_dict.pt')
        # early stop
        if early_stop.early_stop:
            print("Early stopping at epoch: " + str(epoch))
            break

    # save output data... (loss values...)
    with open('train-loss', 'w') as fp:
        for i in train_losses:
            fp.write(str(i) + '\n')
    with open('test-loss', 'w') as fp:
        for i in test_losses:
            fp.write(str(i) + '\n')

    # Final testing
    model.eval()
    correct = 0
    total = 0

    for images, labels in testLoader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100 * correct/total))

    # Save the Trained Model
    torch.save(model.state_dict(), 'model.state_dict.pt')
    torch.save(model, 'model.pt')
