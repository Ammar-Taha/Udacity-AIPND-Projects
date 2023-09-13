import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim
from collections import OrderedDict
from PIL import Image

# 1. Function to Load and Prepare Datasets
def DataLoaders(data_dir='flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Compose Transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_valid_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Datasets
    trainset = ImageFolder(root=train_dir, transform=train_transforms)
    validset = ImageFolder(root=valid_dir, transform=test_valid_trans)
    testset  = ImageFolder(root=test_dir , transform=test_valid_trans)
    # DataLoaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = DataLoader(validset, batch_size=64)
    testloader  = DataLoader(testset,  batch_size=64)

    return trainset, trainloader, validloader, testloader

# 2. Function to Build a Network from a Pretrained Model
## Choices of Pretrained Networks with in_features as Values
imagenets = {"vgg16":25088, "densenet121":1024, "alexnet":9216}
def CNNetwork(pretrained='vgg16', learn_rate=0.001, hidden1=512, drop=0.2, worker='gpu'):
    # Determine the Pretrained Choice | I used vgg16 , and (alexnet, resnet) chosen from First Project
    if pretrained == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif pretrained == 'densenet121':
        model = models.densenet121(pretrained = True)
    else:
        model = models.alexnet(pretrained=True)
        print("The AlexNet is used as a Feature Extraction Architecture.\n Other Choices are (vgg16,densenet121)")
    # Freeze Pretrained Network Parameters
    for param in model.parameters():
        param.requires_grad = False
    # Building the Flower Classifier
    out_features = 102   # Number of classes for flower species, in_features left to match Pretrained (=25088) for vgg16
    hidden2 = hidden1 / 2 # Cutting the hidden layer
    FlowersClassifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(imagenets[pretrained], hidden1)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(drop)),   
        ('fc2', nn.Linear(hidden1, hidden2)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(drop)), 
        ('fc3', nn.Linear(hidden2, out_features)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = FlowersClassifier
    if torch.cuda.is_available() and worker == 'gpu':
        model.cuda()
    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    return model, criterion, optimizer


# 3. Function to Train the Netwrok and perform Validation
def NetworkTrainer(model, criterion, optimizer, trainloader, validloader, epochs = 5, worker='gpu'):
    print("########### The Training and Validation Process Started ###########")
    # The Training Loop
    for epoch in range(epochs):
        #### Training ####
        model.train()  
        train_loss = 0.0
        for images, labels in trainloader:
            # Sending variables to device and Empty Gradients 
            if torch.cuda.is_available() and worker =='gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            # Forward Pass 
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
            # Training Loss
            train_loss += loss.item()
        #### Validation ####
        model.eval()
        valid_loss = 0.0
        accuracy = 0
        with torch.no_grad():
            for images, labels in validloader:
                if torch.cuda.is_available() and worker =='gpu':
                    images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # Calculate average training and validation loss
        avg_train_loss = train_loss / len(trainloader)
        avg_valid_loss = valid_loss / len(validloader)
        # Calculate average accuracy
        avg_accuracy = accuracy / len(validloader)         
        # Print Stats     
        print(f"Epoch {epoch+1}/{epochs}, "                                                         
            f"Train Loss: {avg_train_loss:.4f}, "       
            f"Validation Loss: {avg_valid_loss:.4f}, "                               
            f"Validation Accuracy: {avg_accuracy*100:.2f}%")   
        
# 4. Function to Test the Network and Print Test Accuracy
def NetworkTester(model, criterion, testloader, worker='gpu'):
    print("########### The Testing Process Started ###########")
    model.eval()
    test_loss = 0.0
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            if torch.cuda.is_available() and worker =='gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # Calculate accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    # Calculate average test loss and accuracy
    avg_test_loss = test_loss / len(testloader)
    avg_test_accuracy = accuracy / len(testloader)

    print(f"Test Loss: {avg_test_loss:.4f}, "
        f"Test Accuracy: {avg_test_accuracy*100:.2f}%")

# 5. Function to Save a Checkpoint for the Network
def SaveCheckpoint(model, trainset, path='checkpoint.pth', pretrained='vgg16', epochs=5, n_classes=102, hidden1=512, drop=0.2, learn_rate=0.001):
    FlowerClassifier = model.classifier
    model.class_to_idx = trainset.class_to_idx
    checkpoint = {
        'base_net': pretrained,
        'epochs': epochs, 
        'hidden_layer1': hidden1,
        'dropout': drop, 
        'learning_rate':learn_rate,
        'classifier': FlowerClassifier,
        'in_features':FlowerClassifier[0].in_features, 
        'n_classes':n_classes,     # out_features
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, path)

# 6. Function to  Load the Checkpoint
def LoadCheckpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    # Creating a New Model with the CNNetwork()
    pretrained  = checkpoint['base_net']
    in_features = checkpoint['in_features']
    learn_rate, drop = checkpoint['learning_rate'], checkpoint['dropout']
    hidden1 = ['hidden_layer1']
    model,_,_ = CNNetwork(pretrained, in_features, learn_rate, hidden1, drop)
    # Assigning the Classifier
    model.classifier = checkpoint['classifier']
    # Getting Model StateDict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# 7. Function to Process PIL Image
def PILImageProcessor(img_path):
    pil_image = Image.open(img_path)
    # Making a Compose Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_torch = transform(pil_image)
    return image_torch

# 8. Function to Make a Prediction with the Network - Model Inference
def Predictor(image_path, model, cat_to_name, topk=5, worker='gpu'):
    # cat_to_name has to be loaded in the main script before using this function
    if torch.cuda.is_available() and worker == 'gpu':
        model.to('cuda')
    # Process Image Input
    image_torch = PILImageProcessor(image_path)
    image = image_torch.unsqueeze_(0).float()
    # Make a Forward Step
    with torch.no_grad():
        if worker == 'gpu' and torch.cuda.is_available():
            prediction = model.forward(image.cuda())
        else:
            prediction = model.forward(image)
    # Retrieve Top-K Probabilities with Class Indices
    class_probs, class_indices = torch.topk(nn.functional.softmax(prediction, dim=1), topk)
    # Obtain Class Names from Indices
    class_names = [cat_to_name[str(class_idx)] for class_idx in class_indices.squeeze().cpu().numpy()]
    pred_class_name = class_names[class_probs.argmax().item()]
    
    return class_probs, class_names, pred_class_name