from transformers import DetrImageProcessor

PROCESSOR = DetrImageProcessor()

    
def collate_fn(batch):
  pixel_values = [item['pixel_values'] for item in batch]
  encoding = PROCESSOR.pad(pixel_values, return_tensors="pt")
  labels = [item['labels'] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['labels'] = labels
  
  return batch