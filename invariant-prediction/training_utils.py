import torch

def update_stats(training_stats, epoch_stats):
    """Append epoch_stats to a list of dicts; init with None."""
    if training_stats is None:
        return [epoch_stats]
    return training_stats + [epoch_stats]

def train_pyg(loader, model, optimiser, epoch, loss_fct, device):
    """ Train model for one epoch
    """
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()

        y_hat = model(batch)
        loss = loss_fct(y_hat, batch.y)

        loss.backward()
        optimiser.step()
    return loss.item()

def evaluate_pyg(loader, model, loss_fct, device):
    """ Evaluate model on dataset
    """
    model.eval()
    loss_eval = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            y_hat = model(batch)
            loss = loss_fct(y_hat, batch.y)

            loss_eval += loss.item() * batch.num_graphs
    loss_eval /= len(loader.dataset)
    return loss_eval

def run_exp_pyg(model, train_loader, val_loader, test_loader, loss_fct,
                    lr=0.001, num_epochs=100, device=torch.device('cpu')):
    """ Train the model for NUM_EPOCHS epochs
    """
    print("\nModel architecture:")
    print(model)

    model = model.to(device)

    #Instantiatie our optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    training_stats = None

    #initial evaluation (before training)
    val_loss = evaluate_pyg(val_loader, model, loss_fct, device)
    train_loss = evaluate_pyg(train_loader, model, loss_fct, device)
    epoch_stats = {'train_loss': train_loss, 'val_loss': val_loss, 'epoch':0}
    training_stats = update_stats(training_stats, epoch_stats)

    print("\nStart training:")
    for epoch in range(num_epochs):
        train_loss = train_pyg(train_loader, model, optimiser, epoch,
                                        loss_fct, device)
        val_loss = evaluate_pyg(val_loader, model, loss_fct, device)
        print(f"[Epoch {epoch+1}]",
                    f"train loss: {train_loss:.3f} val loss: {val_loss:.3f}",
              )
        # store the loss and the computed metric for the final plot
        epoch_stats = {'train_loss': train_loss, 'val_loss': val_loss,
                      'epoch':epoch+1}
        training_stats = update_stats(training_stats, epoch_stats)

    test_loss = evaluate_pyg(test_loader, model,  loss_fct, device)
    print(f"Done! Test loss: {test_loss:.3f}")
    return training_stats