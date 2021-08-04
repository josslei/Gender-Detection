import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import sys

def export_sample_images(amount:int, export_dir:str, dataset, shuffle=True):
    os.makedirs(export_dir, exist_ok=True)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=amount, shuffle=shuffle)
    for images, _ in loader:
        for i, img in enumerate(images):
            img = img.squeeze(0)
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(export_dir, str(i)) + '.png')
        break

def getStat(train_data):
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    if input('Are you sure to start calculating mean and std? [y/n] ') != y:
        exit()
    if len(sys.argv) != 2:
        print('Please specify the path of the dataset')
        exit(-1)
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root=r'/home/user/data/gender/train', transform=transform)
    mean, std = getStat(train_dataset)
    print('mean = ', mean)
    print('std  = ', std)

