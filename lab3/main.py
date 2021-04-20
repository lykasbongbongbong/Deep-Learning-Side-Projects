import dataloader
import train
def main():
    #處理data
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    #train
    model = train.EEGNet()
    #test
    print(model)

if __name__ == '__main__':
    main()