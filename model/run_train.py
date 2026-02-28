import torch
import tqdm
import wandb

from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model.parameters import Params


device = Params.DEVICE

def run_train(
        model,
        train_data,
        val_data,
        epochs
):

    wandb.init(
        project="extract-match-qa",
        name=f"first_train",
        tags=["bert", "russian", "qa"],
        config={
            "model": "MilyaShams/rubert-russian-qa-sberquad",
            "epochs": epochs,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "weight_decay": 1e-2,
            "warmup_ratio": 0.1,
        }
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)

    num_train_steps = epochs * len(train_data)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1*num_train_steps),
        num_training_steps=num_train_steps
    )

    loss_train = []
    loss_val = []
    exact_match_train = []
    exact_match_val = []
    f1_val = []

    for epoch in range(epochs):
        
        model.train()

        loss_t, accuracy_train = one_epoch(
            model,
            train_data,
            loss_function,
            epoch,
            epochs,
            optimizer=optimizer,
            scheduler=scheduler
        )

        loss_train.append(loss_t)
        exact_match_train.append(accuracy_train)

        with torch.no_grad():
            model.eval()
            loss_v, accuracy_val, f1 = one_epoch(
                model,
                val_data,
                loss_function,
                epoch,
                epochs,
                evaluate=True
            )

        loss_val.append(loss_v)
        exact_match_val.append(accuracy_val)
        f1_val.append(f1)

        print(f"F1 score: {f1}")
        print(f"Epoch: [{epoch+1}/{epochs}], train_loos: {loss_t:.4f}, train_exact_match: {accuracy_train:.4f}, val_loos: {loss_v:.4f},  val_exact_match: {accuracy_val:.4f}")

        wandb.log({
            'train_loss': loss_t,
            'train_exact_match': accuracy_train,
            'val_loss': loss_v,
            'val_exact_match': accuracy_val,
            'val_f1': f1,
            'epoch': epoch + 1
        })
    
        if epoch % 2 == 0 or epoch == epochs-1:
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch+1}.pt")
    
    wandb.finish()

    return [
        loss_train,
        loss_val,
        exact_match_train,
        exact_match_val,
        f1_val
    ]


def calculate_f1(pred_start, pred_end, target_start, target_end):

    pred_tokens = set(range(pred_start, pred_end+1))
    target_tokens = set(range(target_start, target_end+1))

    intersection = len(pred_tokens & target_tokens)

    if intersection == 0:
        return 0

    precision = intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = intersection / len(target_tokens) if len(target_tokens) > 0 else 0

    if precision + recall == 0:
        return 0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def one_epoch(
        model, 
        data,
        loss_function,
        epoch,
        epochs,
        evaluate=False,
        optimizer=None,
        scheduler=None):
    
    running_loss = []
    running_accuracy = []
    running_f1 = []

    loop = tqdm.tqdm(data, leave=False)

    for batch in loop:

        batch = {k: v.to(device) for k, v in batch.items()}
            
        start_target = batch["start_positions"]
        end_target = batch["end_positions"]
        
        pred = model(**batch)

        start_pred = pred.start_logits
        end_pred = pred.end_logits
                
        start_loss = loss_function(start_pred, start_target)
        end_loss = loss_function(end_pred, end_target)

        total_loss = (start_loss + end_loss) / 2

        start_pred = start_pred.argmax(dim=-1)
        end_pred = end_pred.argmax(dim=-1)

        if evaluate:
            batch_f1 = []
            for i in range(len(start_target)):
                f1_val = calculate_f1(
                    start_pred[i].item(), 
                    end_pred[i].item(), 
                    start_target[i].item(), 
                    end_target[i].item()
                )
                batch_f1.append(f1_val)
            running_f1.append(sum(batch_f1) / len(batch_f1))
        else:
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        accuracy = ((start_pred == start_target) & (end_pred == end_target)).float().mean().item()

        running_accuracy.append(accuracy)

        running_loss.append(total_loss.item())

        mean_loss = sum(running_loss)/len(running_loss)

        mean_acc = sum(running_accuracy)/len(running_accuracy)

        if not evaluate:
            loop.set_description(f"Epoch: [{epoch+1}/{epochs}], train_loos: {mean_loss:.4f}, exact match: {mean_acc:.4f}")

    mean_acc = sum(running_accuracy)/len(running_accuracy)

    if evaluate:
        mean_f1 = sum(running_f1)/len(running_f1)
        return mean_loss, mean_acc, mean_f1
    
    return mean_loss, mean_acc
            
            

