from torch.utils.data import DataLoader 
from dataset import ICLEVRLoader  
from model import Generator, Discriminator 
import os 
import tqdm 
from util import get_test_conditions, save_image
import torch 
from evaluator import evaluation_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, NetG, NetD, root_folder, epochs, lr, z_dim, batch_size):
    Loss = nn.BCELoss()
    NetG_optimizer = torch.optim.Adam(NetG.parameters(), lr, betas=(0.5, 0.99))
    NetD_optimizer = torch.optim.Adam(NetD.parameters(), lr, betas=(0.5, 0.99))

    best_acc = 0.
    best_model = {}
    
    for epoch in tqdm(range(1, 1+epochs)):
        print_interval = 0
        total_Gloss = 0.
        total_DLoss = 0.
        for idx, (img, cond) in enumerate(train_loader):
            # train G:D = 10:1

            NetG.train()
            
            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            cond = cond.to(device)
            
            # train G
            for i in range(10):
                NetG_optimizer.zero_grad()
                z = torch.randn(batch_size, z_dim).to(device)
                imgG = NetG(z, cond)
                pred = NetD(imgG, cond)
                loss_g = Loss(pred, real)
                loss_g.backward()
                NetG_optimizer.step()
            
            # train D 
            NetD.train()
            NetD_optimizer.zero_grad()

            #真照片 (dataloader拿)
            img = img.to(device)
            pred = NetD(img, cond)   
            loss_r = Loss(pred, real)
            #假照片 (latent sample)
            z = torch.randn(batch_size, z_dim).to(device)
            generated_img = NetG(z, cond)
            pred = NetD(generated_img, cond)
            loss_f = Loss(pred, fake)
            loss_sum_discriminator = loss_r + loss_f
            loss_sum_discriminator.backward()
            NetD_optimizer.step()

            if print_interval % 50 == 0:
                print(f"Epoch {epoch}: {idx}|{len(train_loader)} G Loss: {loss_g.item():.4f} D Loss: {loss_sum_discriminator.item():.4f}")
            print_interval += 1

            total_DLoss += loss_sum_discriminator.item()
            total_Gloss += loss_g.item()

        #test
        NetD.eval()
        NetG.eval()
        test_cond = get_test_conditions(os.path.join(root_folder, "test.json")).to(device)
        test_latent = torch.randn(len(test_cond), z_dim).to(device)
        with torch.no_grad():
            generated_img = NetG(test_latent, test_cond)
        acc = evaluation_model.eval(generated_img, test_cond)
        if acc > best_acc:
            best_acc = acc 
            best_model.load_state_dict(NetG.state_dict())
            path_name = "epoch"+epoch+"_acc"+best_acc+".weights"
            best_model_pth = os.path.join("weights", path_name)
            torch.save(best_model, best_model_pth)
        print(f"Testing Accuracy: {acc:.4f}")
        print(f"Average Loss| G: {total_Gloss/len(train_loader):.4f} D:{loss_sum_discriminator/len(train_loader):.4f}")
        save_image(generated_img, os.path.join("result_images", f"epoch{epoch}.png"), nrow=8, normalize=True)

        

def main():
    # param init 
    epochs= 200
    lr = 0.001
    # lr = 0.0002
    batch_size = 64
    z_dim = 100
    c_dim = 200
    img_shape = (64, 64, 3)
    root_folder = "/store3/DL_dataset/data/task_1/"
    data_json_path = "/store3/DL_dataset/data/task_1/train.json"
    # load data
    trainset = ICLEVRLoader(root_folder=root_folder, data_json_path=data_json_path, mode='train')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # model
    NetG = Generator(z_dim,c_dim).to(device)
    NetD = Discriminator(img_shape,c_dim).to(device)
    NetG.weight_init(mean=0,std=0.02)
    NetD.weight_init(mean=0,std=0.02)


    # train
    train(train_loader,NetG,NetD,root_folder,z_dim,epochs,lr, batch_size)


if __name__ == '__main__':
    main()