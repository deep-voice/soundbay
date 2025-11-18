import torch
import torch.optim as optim
from tqdm import tqdm

from dclde_2026 import config, dataset, models
from torch.nn import BCELoss, L1Loss


def compute_loss(preds, targets):
    # This is a placeholder!
    # A real detection loss (e.g., HungarianMatcher + L1/GIoU + FocalLoss) is needed.
    # preds: [BATCH, N_BOXES_PRED, 5 + N_CLASSES]
    # targets: list of [N_BOXES_GT, 5] (class, x, y, w, h)
    
    # Simple placeholder: just try to predict one box
    l1_loss = L1Loss()
    total_loss = 0.0
    
    # Naive loss (just for code to run)
    for i in range(preds.shape[0]):
        if len(targets[i]) > 0:
            gt_box = targets[i][0, 1:5] # First GT box
            pred_box = preds[i, 0, :4]  # First pred box
            total_loss += l1_loss(pred_box, gt_box.to(config.DEVICE))
    
    return total_loss / preds.shape[0]


def main():
    print("Starting Transformer training...")
    
    # 1. Load data
    df_ann = dataset.create_full_annotation_df()
    df_chips = dataset.create_chip_list(df_ann)
    train_loader, val_loader, _ = dataset.get_dataloaders(
        df_ann, df_chips, 
        fold_id=0, 
        transformer_resize=True # Must resize for ViT
    )

    # 2. Init model
    model = models.get_model().to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    print("WARNING: Using a placeholder loss function.")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            if images.nelement() == 0: continue # Skip empty batches
            images = images.to(config.DEVICE)
            
            optimizer.zero_grad()
            preds = model(images)
            
            loss = compute_loss(preds, labels)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                if images.nelement() == 0: continue
                images = images.to(config.DEVICE)
                preds = model(images)
                loss = compute_loss(preds, labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader)}")
        
    print("Transformer training complete (with placeholder loss).")
    torch.save(model.state_dict(), "transformer_detector.pth")


if __name__ == "__main__":
    main()


