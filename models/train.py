from models.layers import *
from tqdm import tqdm
from models.noise_engineering import data_aug

# Model Training
def train(samples, config):
    # hyperparameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    in_dim = config['in_dim']
    hidden_dim = config['hidden_dim']
    out_dim = config['out_dim']
    dropout = mask_rate = config['noise_rate']
    num_layers = config['num_layers']
    norm = config['norm']
    aug_multiple = config['aug_multiple']
    
    PATIENCE = 5
    early_stop_threshold = 1e-3
    prev_loss = np.inf
    stop_count = 0
    
    # model init
    model = GraphSAGE(in_dim, hidden_dim, out_dim, dropout, mask_rate, num_layers, norm)
    best_state_dict= model.state_dict()

    # Loss = nn.MSELoss()
    modal_loss = ModalLoss(config['feat_span'])
    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    dataloader = create_dataloader(samples, batch_size, shuffle=False)
    for epoch in tqdm(range(epochs)):
        running_loss = []
        for batch_samples in dataloader:
            for _ in range(aug_multiple):
                _, graphs, inputs = batch_samples
                opt.zero_grad()
                aug_gs, aug_inputs = data_aug(graphs, inputs, model.mask_rate)
                outputs = model(aug_gs, aug_inputs)
                # loss = Loss(outputs, aug_inputs)
                loss = modal_loss(outputs, aug_inputs)
                loss.backward()
                opt.step()
                running_loss.append(loss.item())
        epoch_loss = np.mean(running_loss)
        if prev_loss - epoch_loss < early_stop_threshold:
            stop_count += 1
            if stop_count == PATIENCE:
                print('Early stopping')
                model.load_state_dict(best_state_dict)
                break
        else:
            best_state_dict = model.state_dict()
            stop_count = 0
            prev_loss = epoch_loss
        if epoch % 50 == 0:
            print(f'epoch {epoch} loss: ', epoch_loss)
        scheduler.step(np.mean(running_loss))
    return model
