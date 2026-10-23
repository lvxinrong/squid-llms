import torch

from squid.train.common import lr_at_step, calc_loss_batch, evaluate_model


def sft_train_model(
        model, train_loader, val_loader, optimizer,
        device, n_epochs, eval_freq, eval_iter,
        warmup_steps, initial_lr=3e-5, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    peak_lr = optimizer.param_groups[0]["lr"]
    total_steps = len(train_loader) * n_epochs

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1
            lr = lr_at_step(global_step, total_steps, warmup_steps, peak_lr, initial_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            track_lrs.append(lr)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            if global_step >= warmup_steps:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step: 06d}): "
                      f"Train loss {train_loss:.3f},"
                      f"Val loss {val_loss:.3f}"
                      )

    return train_losses, val_losses, track_tokens_seen, track_lrs

