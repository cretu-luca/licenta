import copy
import torch


def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key, val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


def train_pyg(loader, model, optimiser, loss_fct, DEVICE="cpu"):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimiser.zero_grad()

        y_hat = model(batch)
        loss = loss_fct(y_hat, batch.y)

        loss.backward()
        optimiser.step()

        total_loss += loss.item() * batch.num_graphs
        preds = y_hat.argmax(dim=-1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def evaluate_pyg(loader, model, loss_fct, DEVICE="cpu"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.no_grad():
            y_hat = model(batch)
            loss = loss_fct(y_hat, batch.y)

            total_loss += loss.item() * batch.num_graphs
            preds = y_hat.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

    return total_loss / total, correct / total


def run_exp_pyg(model, train_loader, val_loader, test_loader, loss_fct,
                lr=0.001, num_epochs=400, patience=100, DEVICE="cpu"):
    print("\nModel architecture:")
    print(model)

    model = model.to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    training_stats = None

    val_loss, val_acc = evaluate_pyg(val_loader, model, loss_fct, DEVICE)
    train_loss, train_acc = evaluate_pyg(train_loader, model, loss_fct, DEVICE)
    epoch_stats = {
        'train_loss': train_loss, 'train_acc': train_acc,
        'val_loss': val_loss, 'val_acc': val_acc, 'epoch': 0,
    }
    training_stats = update_stats(training_stats, epoch_stats)

    best_val_loss = val_loss
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    print("\nStart training:")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_pyg(train_loader, model, optimiser,
                                          loss_fct, DEVICE)
        val_loss, val_acc = evaluate_pyg(val_loader, model, loss_fct, DEVICE)
        print(f"[Epoch {epoch+1}]"
              f" train loss: {train_loss:.3f} acc: {train_acc:.3f}"
              f" | val loss: {val_loss:.3f} acc: {val_acc:.3f}")

        epoch_stats = {
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch + 1,
        }
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
    test_loss, test_acc = evaluate_pyg(test_loader, model, loss_fct, DEVICE)
    print(f"Done! Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")

    training_stats['test_loss'] = test_loss
    training_stats['test_acc'] = test_acc
    return training_stats
