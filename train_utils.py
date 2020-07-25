import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim 
from torchvision import datasets, transforms, models
import seaborn as sns
import PIL
sns.set()

from torch.autograd import Variable


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
	
def save_checkpoint_location(state, save_dir, is_best=False, filename='my_checkpoint.pth.tar'):
    path = save_dir + filename
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, 'my_checkpoint.pth.tar')

def load_data(data_dir = 'flowers'):
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'

    
	
	data_transforms = {
    'train_transform' : transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    'validation_transform' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),
    'test_transform' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
                            
	}
	train_data = datasets.ImageFolder(train_dir,transform = data_transforms['train_transform'])
	validation_data = datasets.ImageFolder(valid_dir,transform = data_transforms['validation_transform'])
	test_data = datasets.ImageFolder(test_dir,transform = data_transforms['test_transform'])
	
	trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32,shuffle=True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)
	validloader = torch.utils.data.DataLoader(validation_data, batch_size = 32)

    return trainloader, validloader, testloader

	
def train_model(trainloader, validloader, arch, hidden_units, learning_rate, cuda, epochs, save_dir, save_every):
    # Initial parameters
    print_every = 1
    save_every = 50

    # Get model
    model = eval("models.{}(pretrained=True)".format(arch))

    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False

    classifier =  nn.Sequential(
            nn.Linear(25088,4096,bias = True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,102,bias = True),
            nn.LogSoftmax(dim=1)
            )

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if cuda:
        model.cuda()
    else:
        model.cpu()
        
    epochs = epochs
    steps = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        accuracy_train = 0
        
        for images, labels in iter(trainloader):
            steps += 1
            


            inputs, labels = Variable(images), Variable(labels)
            
            optimizer.zero_grad()
            
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
      
            output = model.forward(inputs)
  
            loss = criterion(output, labels)

            loss.backward()
    
            optimizer.step()
            
            running_loss += loss.item()
            ps_train = torch.exp(output).data
            equality_train = (labels.data == ps_train.max(1)[1])
            accuracy_train += equality_train.type_as(torch.FloatTensor()).mean()
            
            
            
            if steps % print_every == 0:
                model.eval()
                
                accuracy = 0
                valid_loss = 0
                
                for images, labels in validloader:
                    with torch.no_grad():
                        inputs = Variable(images)
                        labels = Variable(labels)

                        if cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = model.forward(inputs)

                        valid_loss += criterion(output, labels).item()

                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])

                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs), 
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}..".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                model.train()
                
            if steps % save_every == 0:
                print("Saving step number {}...".format(steps))
                state = {'state_dict': model.classifier.state_dict(),
                         'optimizer' : optimizer.state_dict(),
                         'class_to_idx':train_dataset.class_to_idx}
                
                save_checkpoint(state, save_dir)
                print("Done")



    return model
	
	
def predict_from_checkpoint(image_path, checkpoint, topk=5, category_names=None, cuda=False):


	model = load_checkpoint(checkpoint, cuda=cuda)
	image_data = process_image(image_path)
	if cuda:
		model.cuda()
	else:
		model.cpu()
		
	model_p = model.eval()
	
	inputs = Variable(image_data.unsqueeze(0))

	if cuda:
		inputs = inputs.cuda()
	
	output = model_p(inputs)
	ps = torch.exp(output).data
	
	ps_top = ps.topk(topk)
	idx2class = model.idx_to_class
	probs = ps_top[0].tolist()[0]
	classes = [idx2class[i] for i in ps_top[1].tolist()[0]]



	class_names = ""
	if category_names is not None:
		with open(category_names, 'r') as f:
			cat_to_name = json.load(f)

		class_names = [cat_to_name[i] for i in classes]



	return probs, classes, class_names
	

def process_image(image):


    pil_image = PIL.Image.open(image)


    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    

    transform_pil = transform(pil_image)
    
    processed_image = np.array(transform_pil)
    
    return processed_image
	
	
