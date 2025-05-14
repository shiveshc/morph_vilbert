import torchvision
from torch.utils.data import Dataset, DataLoader

image_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

class MNISTtrain(Dataset):
    def __init__(self):
        super().__init__()
        
        self.train_mnist = torchvision.datasets.MNIST(
            root='/Users/shiveshchaudhary/Documents/projects/mnist',
            train=True,
            download=True,
            transform=image_transform
            )
    
    def __len__(self):
        return len(self.train_mnist)

    def __getitem__(self, index):
        img, label = self.train_mnist[index]
        batch = {
            'img':img,
            'label':label
        }
        return batch

class MNISTtest(Dataset):
    def __init__(self):
        super().__init__()
        
        self.test_mnist = torchvision.datasets.MNIST(
            root='/Users/shiveshchaudhary/Documents/projects/mnist',
            train=False,
            download=True,
            transform=image_transform
            )
    
    def __len__(self):
        return len(self.test_mnist)

    def __getitem__(self, index):
        img, label = self.test_mnist[index]
        batch = {
            'img':img,
            'label':label
        }
        return batch

def get_mnist_dataset():
    train_mnist = MNISTtrain()
    test_mnist = MNISTtest()
    return train_mnist, test_mnist
