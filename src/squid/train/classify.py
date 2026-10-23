import torch

def prepare_classify_model(model, num_classes, emb_dim):
    # 关闭model中所有参数的梯度更新
    for param in model.parameters():
        param.requires_grad = False

    # 替换输出头，由原本的(B, T, vocab_size) 替换为 (B, T, num_classes)
    # 原本是 (B, T, emb_dim)  @ (emb_dim, vocab_size)  替换成  (B, T, emb_dim) @ (emb_dim, num_classes)
    model.out_head = torch.nn.Linear(
        in_features=emb_dim,
        out_features=num_classes,
    )
    model.out_head.to(model.tok_emb.weight.device)

    for param in model.trf_block[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True



def train_classifier_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.size(0)
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_classifier_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_epochs)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_epochs)
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
    return total_loss / num_batches

def evaluate_classifier_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predictions_labels = torch.argmax(logits, dim=-1)
            num_examples += predictions_labels.shape[0]
            correct_predictions += (
                (predictions_labels == target_batch).sum().item()
            )
        else:
            break
    model.train()
    return correct_predictions / num_examples