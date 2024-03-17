import torch
import torchvision.transforms as tvt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import os

# class to organize the training data into a dataset
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self,root):
        super()
        self.root = root
        self.filenames = os.listdir(root)
        self.transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.gt_images = []
        self.noisy_images = []
        for i in range(len(self.filenames)):
            if i%2 == 0:
                self.gt_images.append(Image.open(root + "/" + self.filenames[i]))
            else:
                self.noisy_images.append(Image.open(root + "/" + self.filenames[i]))

    def __len__(self):
        return 100

    def __getitem__(self,index):
        gt_image = self.transform(self.gt_images[index])
        noisy_image = self.transform(self.noisy_images[index])
        return gt_image, noisy_image
    
# class to organize the testing data into a dataset
class TestingDataset(torch.utils.data.Dataset):
    def __init__(self,root):
        super()
        self.root = root
        self.filenames = os.listdir(root)
        self.transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.gt_images = []
        self.noisy_images = []
        for i in range(len(self.filenames)):
            if i%2 == 0:
                self.gt_images.append(Image.open(root + "/" + self.filenames[i]).resize(512,512))
            else:
                self.noisy_images.append(Image.open(root + "/" + self.filenames[i]).resize(512,512))

    def __len__(self):
        return 100

    def __getitem__(self,index):
        gt_image = self.transform(self.gt_images[index])
        noisy_image = self.transform(self.noisy_images[index])
        return gt_image, noisy_image
    
# dCNN model class used by the paper
class dCNN(torch.nn.Module):
  def __init__(self, channels, layers=17):
    super(dCNN, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1, bias=False)
    self.convList = torch.nn.ModuleList()
    for i in range(layers-2):
      self.convList.append(torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
      self.convList.append(torch.nn.BatchNorm2d(64))
      self.convList.append(torch.nn.ReLU(inplace=True))
    self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, padding=1, bias=False)
  def forward(self, x):
    x = self.conv1(x)
    x = torch.nn.functional.relu(x, inplace=True)
    for conv in self.convList:
      x = conv(x)
    x = self.conv2(x)
    return x

#function to check PSNR(dB)/SSIM values of the resulted images
def calculate_metrics(image, original, data_range):
    image = image.data.cpu().numpy().astype(np.float32)
    original = original.data.cpu().numpy().astype(np.float32)
    psnr_value = 0
    ssim_value = 0
    for i in range(image.shape[0]):
        psnr_value += psnr(original[i,:,:,:], image[i, :, :, :], data_range=data_range)
        ssim_value += ssim(original[i, :, :, :], image[i, :, :, :], channel_axis=0, data_range=data_range)
    psnr_value /= image.shape[0]
    ssim_value /= image.shape[0]

    return psnr_value, ssim_value

#training function
def train(model, dataloader, device, epochs=80, sigma=25):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    alpha = 20
    for epoch in range(epochs):
      running_loss = 0.0
      psnr_value =0.0
      for idx, data in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        noisy_image, clean_image = data
        noisy_image = noisy_image.to(device)
        clean_image = clean_image.to(device)
        D = (sigma/255) * torch.FloatTensor(noisy_image.size()).normal_(mean=0, std=1.).cuda()
        input = noisy_image + alpha*D
        target = noisy_image - D/alpha
        output = model(input)
        loss = criterion(output, target) / target.size()[0]
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        #evaluating the performance per batch
        model.eval()
        output = torch.clamp(model(noisy_image), 0, 1)
        psnr_val, ssim_value = calculate_metrics(output, clean_image, 1.0)
        psnr_value += psnr_val
        if idx % 50 == 49:
          print('Epoch:', epoch + 1, '[', idx+1, '/', len(dataloader), ']', '=>', 'PSNR:', psnr_value/50, 'Loss:', running_loss/50)
          running_loss = 0.0
          psnr_value = 0.0
    file_name = 'model_' + str(sigma) + '.pth'
    torch.save(model.state_dict(), file_name)

#testing function
def test(dataset, model, model_path, sigma, device):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    psnr_list = []
    ssim_list = []
    alpha = 0.5
    test_output = None
    for i in range(len(dataset)):
        noisy_image, clean_image = dataset[i]
        noisy_image = torch.unsqueeze(noisy_image,0).to(device)
        clean_image = torch.unsqueeze(clean_image,0).to(device)
        input = noisy_image + alpha * (sigma/255) * torch.FloatTensor(noisy_image.size()).normal_(mean=0, std=1.).cuda()
        with torch.no_grad():
            output = model(input)
        test_output = output.detach() if test_output == None else test_output + output.detach()
        del output
        out = torch.clamp(test_output, 0, 1)
        psnr_val, ssim_value = calculate_metrics(out, clean_image, data_range=1.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_value)
    print( 'PSNR of testing dataset:', sum(psnr_list)/len(psnr_list), 'SSIM of testing dataset:', sum(ssim_list)/len(ssim_list))

def sample_testing(dataset, model, model_path, sigma, device):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    idx = random.randint(0, 128)
    noisy_image, clean_image = dataset[idx]
    test_output = None
    noisy_image = torch.unsqueeze(noisy_image,0).to(device)
    clean_image = torch.unsqueeze(clean_image,0).to(device)
    alpha = 0.5
    input = noisy_image + alpha * (sigma/255) * torch.FloatTensor(noisy_image.size()).normal_(mean=0, std=1.).cuda()
    with torch.no_grad():
      output = model(input)
    test_output = output.detach() if test_output == None else test_output + output.detach()
    del output
    out = torch.clamp(test_output, 0, 1)
    psnr_val, ssim_value = calculate_metrics(out, clean_image, data_range=1.0)
    fig,axes = plt.subplots(1,3)
    noisy_image = noisy_image.squeeze().cpu() * torch.tensor([0.247, 0.243, 0.261]).view(3,1,1)
    noisy_image = noisy_image + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    noisy_image = tvt.ToPILImage()(noisy_image)
    axes[1].imshow(noisy_image)
    axes[1].set_title("Noisy Image")
    clean_image = clean_image.squeeze().cpu() * torch.tensor([0.247, 0.243, 0.261]).view(3,1,1)
    clean_image = clean_image + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    clean_image = tvt.ToPILImage()(clean_image)
    axes[0].imshow(clean_image)
    axes[0].set_title("Clean Image")
    out_image = out.squeeze().cpu() * torch.tensor([0.247, 0.243, 0.261]).view(3,1,1)
    out_image = out_image + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    out_image = tvt.ToPILImage()(out_image)
    axes[2].imshow(out_image)
    axes[2].set_title("Denoised Image")

    fig.suptitle("PSNR: " + str(round(psnr_val,2)) + " SSIM: " +  str(round(ssim_value,2)) + ' when σ=' + str(sigma))
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_dataset = TrainingDataset("./CroppedImages")
    testing_dataset = TestingDataset("./OriginalImages")    