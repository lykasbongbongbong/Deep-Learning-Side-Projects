def resnet_18_with_pretrained():
    #hyper param:
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    Loss = nn.CrossEntropyLoss()
    epochs_feature_extraction = 5
    epochs_fine_tuning = 5
    weight_path = "weights/resnet18_with_pretrained.weight"

    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ResNet18(classes=5 ,pretrained=True)
    model.to(device)
    # model.load_state_dict(torch.load("weights/resnet18_with_pretrained.weight"))
    # print(f"---Model loads pretrained weight from {weight_path}---")
    
    #feature extraction
    print(f"---Start feature extraction for {epochs_feature_extraction} epochs.---")
    params_to_update = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=5e-4)
    dataframe_feature_extraction = train_eval(model, train_loader, test_loader, epochs_feature_extraction, optimizer)
    dataframe_feature_extraction.to_csv("r18_with_pretrained_accuracy_feature_extraction.csv", index=False)
    print(dataframe_feature_extraction)

    #fine tune
    print(f"---Start fine tuning for {epochs_fine_tuning} epochs.---")
    for param in model.parameters():
        param.requires_grad=True
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    dataframe_fine_tuning = train_eval(model, train_loader, test_loader, epochs_fine_tuning, optimizer)
    dataframe_fine_tuning.to_csv("r18_with_pretrained_accuracy_fine_tuning.csv", index=False)
    print(dataframe_fine_tuning)

    
    dataframe_ff = pd.concat([dataframe_feature_extraction, dataframe_fine_tuning], axis=0, ignore_index=True)
    print(dataframe_ff)
    dataframe_ff.to_csv("r18_with_pretrained_accuracy.csv", index=False)