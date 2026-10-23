from squid.train.common import calc_loss_batch
from squid.train.common import evaluate_model


def pretrain_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, track_tokens_seen = [], [], []
    global_step = -1
    # 记录训练过的token数量
    token_seen = 0

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Ep {epoch + 1} (Step {global_step: 06d}): "
                      f"Train loss {train_loss:.3f},"
                      f"Val loss {val_loss:.3f}"
                )

    return train_losses, val_losses, track_tokens_seen