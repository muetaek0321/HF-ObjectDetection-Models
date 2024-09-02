import torch


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['labels'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }