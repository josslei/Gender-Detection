import torch
import torch.nn as tnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
#
from train import BATCH_SIZE
from train import LEARNING_RATE

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("[1] Please specify the path of model!")
        print("[2] Please specify how many epochs to train!")
        exit(-1)

    transform_train = transforms.Compose([
        transforms.Resize(200),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std  = [ 0.229, 0.224, 0.225 ])
        ])
    transform_test = transforms.Compose([
        transforms.Resize(200),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std  = [ 0.229, 0.224, 0.225 ])
        ])
    
    trainData = datasets.ImageFolder('~/data/gender/train/', transform_train)
    testData = datasets.ImageFolder('~/data/gender/test', transform_test)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

    EPOCH = int(sys.argv[2])
    model = torch.load(sys.argv[1])
    model.cuda()

    # Cost function, optimizer & scheduler
    cost = tnn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=(LEARNING_RATE / 100))  # avoid a large lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    '''
    Training
    '''
    for epoch in range(EPOCH):
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
            print("[E: %d] \tloss: %f  \tavg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
            loss.backward()
            optimizer.step()
        scheduler.step(avg_loss)
        # save module
        torch.save(model.state_dict(), 'model.state_dict.pt')
        torch.save(model, 'model.pt')

    # Test the model
    model.eval()
    correct = 0
    total = 0

    for images, labels in testLoader:
        images = images.cuda()
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100* correct/total))

    # Save the Trained Model
    torch.save(model.state_dict(), 'model.state_dict.pt')
    torch.save(model, 'model.pt')
