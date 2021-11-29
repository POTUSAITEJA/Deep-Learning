from torchvision import transforms, datasets

root_dir = ':/data/'

train_transforms= transforms.Compose([transforms.resize(255),
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])


test_transforms = transforms.Compose([transforms.resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])


train_dataset = datasets.ImageFolder(root=root_dir+"train/",transform=train_transforms)

test_dataset = datasets.ImageFolder(root=root_dir+"test/",transform=test_transforms)