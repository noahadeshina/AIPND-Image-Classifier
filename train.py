import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
import argparse


def build_model(arch, hidden_units, lr):
    layers = hidden_units
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, layers)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(layers, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
        
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, layers)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('fc2', nn.Linear(layers, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        print('Invalid Architecture!')
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    return model, optimizer, criterion

def train(arch, hidden_units, lr, trainloader, validloader):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = build_model(arch, hidden_units, lr)
    model.to(device)
    
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    
    print("Training Started")
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f'Epoch {epoch+1}/{epochs}.. '
                      f'Train loss: {running_loss/print_every:.3f}.. '
                      f'Validation loss: {valid_loss/len(valid_loader):.3f}.. '
                      f'Validation accuracy: {accuracy/len(valid_loader):.3f}')
                running_loss = 0

                model.train()
                
    print('Model has been trained successfully!')
    if args.save_dir:
        model.class_to_idx = train_datasets.class_to_idx
        
        checkpoint = {'arch': arch,
                      'classifier': model.classifier,
                      'class_to_idx': model.class_to_idx,
                      'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict()}
        
        path = args.save_dir + 'model_chkpt.pth'
        torch.save(checkpoint, path)
        print('Saving Checkpoint...')
        print(f'Checkpoint saved to ./{path}')
    return model

def main():     
    if args.data_dir:
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        
        valid_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224,0.225])])
        
        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224,0.225])])
        
        data_dir = args.data_dir
        
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
        test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
        
        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
        test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
        
        train(arch=args.arch, hidden_units=args.hidden_units, lr=args.learning_rate,
              trainloader=train_loader, validloader=valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help='path to images folder')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('--arch', type=str,
                        default='densenet121',
                        help='choose model architecture [densenet121 or vgg16]')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int,
                        default=512, help='number of hidden layers')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')

    args = parser.parse_args()
    
    main()