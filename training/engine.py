"""training.engine: extracted from the original training entry point."""

import gc
import os
import time
import torch
from data_pipeline.normalization import normalize_input
from utils.metrics import compute_all_metrics_torch
from .checkpoint import save_checkpoint


def train_epoch(model, dataloader, optimizer, criterion, device, scaler,
                clip_grad=None, model_name='PROPHET', input_dim=3, scaler_amp=None, epoch=0):
    """Train for one epoch with multi-channel support and optional mixed precision training.
    """
    model.train()

    epoch_loss = 0.0
    epoch_pred_loss = 0.0
    epoch_stssl_temporal_loss = 0.0
    epoch_stssl_spatial_loss = 0.0
    n_batches = 0

    use_stssl = model_name in ['STSSL', 'STSSDL']
    use_prophet = model_name.upper() == 'PROPHET'

    for batch in dataloader:
        X = batch['X'].to(device)
        y_raw = batch['y'].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            if use_stssl:
                # STSSL模型使用自己的calculate_loss方法
                loss, sep_losses = model.calculate_loss({'X': X, 'y': y_raw})
                epoch_pred_loss += sep_losses.get('pred', 0.0)
                epoch_stssl_temporal_loss += sep_losses.get('temporal', 0.0)
                epoch_stssl_spatial_loss += sep_losses.get('spatial', 0.0)
            else:
                # 标准模型使用标准流程
                X_norm = normalize_input(X, scaler, input_dim)
                output = model({'X': X_norm})  # (B, T, N, C_out)

                y_pred_norm = output.mean(dim=-1, keepdim=True)  # (B, T, N, 1)
                y_pred_raw = scaler.inverse_transform(y_pred_norm) # (B, T, N, D_out)

                y_target_raw = y_raw[..., 0:1] # (B, T, N, 1)

                pred_loss = criterion(y_pred_raw, y_target_raw)
                epoch_pred_loss += pred_loss.detach().item()
                loss = pred_loss

        # 混合精度训练的 backward
        if scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            if clip_grad:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        epoch_loss += loss.detach().item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
    avg_pred = epoch_pred_loss / n_batches if n_batches > 0 else 0.0
    avg_stssl_temporal = epoch_stssl_temporal_loss / n_batches if n_batches > 0 and use_stssl else 0.0
    avg_stssl_spatial = epoch_stssl_spatial_loss / n_batches if n_batches > 0 and use_stssl else 0.0
    return {'loss': avg_loss, 'pred_loss': avg_pred,
            'stssl_temporal_loss': avg_stssl_temporal, 'stssl_spatial_loss': avg_stssl_spatial}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, scaler,
             input_dim=3, output_dim=1, model_name='PROPHET'):
    """Evaluate model on validation/test set.

    For multi-channel output, predictions are averaged across channels.

    Returns:
        dict: 包含 loss, MAE, RMSE, MAPE, R2, SMAPE 指标
    """
    model.eval()
    all_preds_raw, all_targets_raw = [], []

    use_stssl = model_name in ['STSSL', 'STSSDL']

    for batch in dataloader:
        X = batch['X'].to(device)
        y_raw = batch['y'].to(device)  # (B, T, N, 1)

        if use_stssl:
            # STSSL: 使用predict方法获取预测
            output = model.predict({'X': X})  # (B, T, N, output_dim)
            y_pred_raw = output[..., 0:output_dim]  # (B, T, N, output_dim)
        else:
            X_norm = normalize_input(X, scaler, input_dim)
            output = model({'X': X_norm})  # (B, T, N, output_dim)

            y_pred_norm = output[..., 0:output_dim]  # (B, T, N, output_dim)
            y_pred_raw = scaler.inverse_transform(y_pred_norm) # (B, T, N, output_dim)

        all_preds_raw.append(y_pred_raw)
        all_targets_raw.append(y_raw)

    all_preds = torch.cat(all_preds_raw, dim=0)
    all_targets = torch.cat(all_targets_raw, dim=0)

    loss = criterion(all_preds, all_targets).item()

    mae, rmse, mape, r2, smape = compute_all_metrics_torch(all_preds, all_targets, null_value=-1.0)

    return {'loss': loss, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'SMAPE': smape}


def train_single_model(args, model, train_loader, val_loader, test_loader,
                       optimizer, scheduler, criterion, device, scaler, logger,
                       model_name, num_nodes, adj_mx=None):
    """Train a single model with early stopping and optional mixed precision training."""
    best_val_mae = float('inf')
    patience_counter = 0
    training_history = []
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

    # 判断是否为 HA 模型
    is_ha = model_name.upper() == 'HA'

    # 创建混合精度训练的 GradScaler
    scaler_amp = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not is_ha)

    # HA 模型使用过去历史平均，没有可训练参数
    # 直接评估一次，不需要多轮训练
    if is_ha:
        logger.info("HA model: using historical average (last 144 steps) for prediction")
        logger.info("Evaluating on val/test set...")
        # 评估验证集
        val_metrics = evaluate(model, val_loader, criterion, device, scaler,
                               input_dim=args.input_dim, output_dim=args.output_dim,
                               model_name=model_name)
        logger.info(f"Val MAE: {val_metrics['MAE']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f} | Val MAPE: {val_metrics['MAPE']:.2f}%")
        training_history = [{'epoch': 1, 'train_loss': 0.0, 'val_loss': val_metrics['loss'], 'val_mae': val_metrics['MAE']}]
        # 评估测试集
        test_metrics = evaluate(model, test_loader, criterion, device, scaler,
                                input_dim=args.input_dim, output_dim=args.output_dim,
                                model_name=model_name)
        logger.info(f"Test MAE: {test_metrics['MAE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f} | Test MAPE: {test_metrics['MAPE']:.2f}%")
        return best_model_path, training_history, test_metrics

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, args.clip_grad, model_name,
            input_dim=args.input_dim, scaler_amp=scaler_amp,
            epoch=epoch,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, scaler,
                              input_dim=args.input_dim, output_dim=args.output_dim,
                              model_name=model_name)

        args.current_epoch = epoch
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0
        epoch_time = time.time() - epoch_start

        logger.info("-" * 60)

        # 构建日志信息
        if model_name in ['STSSL', 'STSSDL']:
            # SSL模型日志
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.0e} | Model: {model_name.upper()}")
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Pred Loss: {train_metrics.get('pred_loss', 0.0):.4f} | "
                f"Temporal: {train_metrics.get('stssl_temporal_loss', 0.0):.4f} | "
                f"Spatial: {train_metrics.get('stssl_spatial_loss', 0.0):.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.4f}"
            )
        elif model_name.upper() in ['PROPHET', 'STID', 'STDN']:
            # 主模型日志
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.0e} | Model: {model_name.upper()}")
            cont_loss = train_metrics.get('contrastive_loss', 0.0)
            cont_str = f" | ContLoss: {cont_loss:.6f}" if model_name.upper() == 'PROPHET' and cont_loss > 0 else ""
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"{cont_str} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.4f}"
            )
        else:
            # 其他模型日志 (GRU, STGCN, AGCRN, etc.)
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.0e} | Model: {model_name.upper()}")
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.4f}"
            )

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['MAE'],
            'lr': current_lr,
            'epoch_time': epoch_time
        })

        if val_metrics['MAE'] < best_val_mae:
            best_val_mae = val_metrics['MAE']
            patience_counter = 0
            # HA 模型不需要保存 checkpoint
            if model_name.upper() != 'HA':
                save_checkpoint(model, optimizer, scheduler, val_metrics, args, scaler, best_model_path)
            logger.info(f"✅ Best model (Val MAE: {best_val_mae:.4f})")
        else:
            patience_counter += 1

        # HA 模型跳过 scheduler
        if scheduler is not None:
            scheduler.step()

        # HA 模型没有参数，只评估一次后退出
        if is_ha:
            logger.info(f"HA model: no training needed, completing after 1 epoch(s)")
            break

        if patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
            break

        if device.type == 'cuda' and epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Test evaluation
    logger.info("-" * 50)
    logger.info(f"Model: {model_name.upper()} - Test evaluation...")

    # HA 模型跳过 checkpoint 加载
    if model_name.upper() != 'HA':
        checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    test_metrics = evaluate(model, test_loader, criterion, device, scaler,
                            input_dim=args.input_dim, output_dim=args.output_dim, model_name=model_name)

    logger.info(f"Test Results ({model_name.upper()}):")
    logger.info(f"  Loss:  {test_metrics['loss']:.4f}")
    logger.info(f"  MAE:   {test_metrics['MAE']:.4f}")
    logger.info(f"  RMSE:  {test_metrics['RMSE']:.4f}")
    logger.info(f"  MAPE:  {test_metrics['MAPE']:.4f}%")
    logger.info(f"  R2:    {test_metrics['R2']:.4f}")
    logger.info(f"  SMAPE:  {test_metrics['SMAPE']:.4f}%")

    return best_model_path, training_history, test_metrics
