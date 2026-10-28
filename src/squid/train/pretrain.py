from squid.train.common import calc_loss_batch
from squid.train.common import evaluate_model

"""
loss = calc_loss(...)   # ① 算出损失值（一个张量）
loss.backward()         # ② 把损失对每个参数的梯度算出来，写进 param.grad
optimizer.step()        # ③ 读取每个参数的 .grad，按规则更新参数（W -= lr × grad 等）
optimizer.zero_grad()   # ④ 把 .grad 清零，准备下一轮
"""
def pretrain_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter,
                   max_steps=None, accum_step: int = 1):
    train_losses, val_losses, track_tokens_seen = [], [], []
    global_step = -1
    # 记录训练过的token数量
    token_seen = 0

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            if max_steps is not None and global_step >= max_steps:
                break
            global_step += 1  # 连续 micro-batch 计数（跨 epoch 不重置）
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss = loss / accum_step
            loss.backward()

            token_seen += input_batch.numel()
            if (global_step + 1) % accum_step == 0:
                # before = model.out_head.weight.detach().clone()
                optimizer.step()
                # print("权重变化量:", (model.out_head.weight - before).abs().max().item())
                optimizer.zero_grad()

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