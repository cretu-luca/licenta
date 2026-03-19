import copy
import torch

def update_stats(training_stats, epoch_stats):
    """ Store metrics along the training
    Args:
      epoch_stats: dict containg metrics about one epoch
      training_stats: dict containing lists of metrics along training
    Returns:
      updated training_stats
    """
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats

def train_pyg(loader, model, optimiser, epoch, loss_fct, DEVICE="cpu"):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimiser.zero_grad()

        y_hat = model(batch)
        loss = loss_fct(y_hat, batch.y)

        loss.backward()
        optimiser.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_pyg(loader, model, loss_fct, DEVICE="cpu"):
    """ Evaluate model on dataset
    """
    model.eval()
    loss_eval = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.no_grad():
            y_hat = model(batch)
            loss = loss_fct(y_hat, batch.y)

            loss_eval += loss.item() * batch.num_graphs
    loss_eval /= len(loader.dataset)
    return loss_eval

def run_exp_pyg(model, train_loader, val_loader, test_loader, loss_fct,
                    lr=0.001, num_epochs=100, patience=20, DEVICE="cpu"):
    print("\nModel architecture:")
    print(model)

    model = model.to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    training_stats = None

    val_loss = evaluate_pyg(val_loader, model, loss_fct, DEVICE)
    train_loss = evaluate_pyg(train_loader, model, loss_fct, DEVICE)
    epoch_stats = {'train_loss': train_loss, 'val_loss': val_loss, 'epoch': 0}
    training_stats = update_stats(training_stats, epoch_stats)

    best_val_loss = val_loss
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    print("\nStart training:")
    for epoch in range(num_epochs):
        train_loss = train_pyg(train_loader, model, optimiser, epoch,
                                        loss_fct, DEVICE)
        val_loss = evaluate_pyg(val_loader, model, loss_fct, DEVICE)
        print(f"[Epoch {epoch+1}]",
                    f"train loss: {train_loss:.3f} val loss: {val_loss:.3f}",
              )
        epoch_stats = {'train_loss': train_loss, 'val_loss': val_loss,
                      'epoch': epoch+1}
        training_stats = update_stats(training_stats, epoch_stats)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_model_state)
    test_loss = evaluate_pyg(test_loader, model, loss_fct, DEVICE)
    print(f"Done! Best val loss: {best_val_loss:.3f} | Test loss: {test_loss:.3f}")
    return training_stats