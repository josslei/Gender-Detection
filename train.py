import torch
import torch.nn as tnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from cnn import cnn

N_CLASSES = 2   # male & female
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 100

# train
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std  = [ 0.229, 0.224, 0.225 ])
        ])
    
    trainData = datasets.ImageFolder('~/data/gender/train/', transform)
    testData = datasets.ImageFolder('~/data/gender/test', transform)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

    model = cnn(N_CLASSES)
    model.cuda()

    # Cost function, optimizer & scheduler
    cost = tnn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
