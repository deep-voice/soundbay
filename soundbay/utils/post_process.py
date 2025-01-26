import torch


def post_process_predictions(preds: torch.Tensor, label_type: str, th: float = None) -> tuple:
    """
    Post-process the predictions to probabilities
    """
    if th is None:
        th = 0.5
    if label_type == 'single_label':
        proba = torch.softmax(preds, 1).cpu().numpy()
        predicted = torch.max(preds, 1).indices.cpu().numpy()
    elif label_type == 'multi_label':
        proba = torch.special.expit(preds).cpu().numpy()
        predicted = (proba > th).astype(int)
    else:
        raise ValueError(f"Label type {label_type} is not allowed")
    return  proba, predicted
