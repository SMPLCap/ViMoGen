import os
import torch
from modules.t5 import T5EncoderModel
from tqdm import tqdm
from time import perf_counter
import json

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, text_key):

        self.data_list = json.load(open(json_file, 'r'))
        self.data_list = [data for data in self.data_list if data[text_key] is not None]
        self.text_key = text_key
    
    def __getitem__(self, data_id):
        data = self.data_list[data_id]
        text = data[self.text_key]
        if isinstance(text, list):
            text = text[0]
        if 'sample_id' in data:
            sample_id = data['sample_id']
        elif 'test_sample_id' in data:
            sample_id = str(data['test_sample_id'])
        elif 'global_id' in data:
            sample_id = str(data['global_id'])
        else:
            raise ValueError("No sample_id or test_sample_id found in the data.")

        data_dict = {"text": text, "sample_id": sample_id}
        
        return data_dict
    
    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Text Embedding Processing Script")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the json file containing the text data")
    parser.add_argument("--text_key", type=str, required=True, help="Key to access the text data in the json file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the text embeddings")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    # model loading
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    checkpoint_folder = './checkpoints/Wan2.1-T2V-1.3B'
    checkpoint_path = os.path.join(checkpoint_folder, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = os.path.join(checkpoint_folder, "google/umt5-xxl")
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        shard_fn=None,
    )

    # Load the dataset and dataloader with batch_size=128
    dataset = TextDataset(args.json_file, args.text_key)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True if device == "cuda" else False
    )

    # Initialize timing accumulators
    total_dataloader_time = 0
    total_prompt_time = 0
    total_save_time = 0
    total_batches = 0

    # Process batches with per-batch timing output using perf_counter
    for batch in tqdm(dataloader, desc="Processing batches"):
        # Measure dataloading time (Note: this is approximate as dataloading happens during iteration)
        dataload_start = perf_counter()
        texts = batch["text"]
        
        sample_ids = batch["sample_id"]
        save_paths = [os.path.join(args.save_dir, args.text_key, f"{sample_id}.pt") for sample_id in sample_ids]
        batch_size = len(texts)  # Actual size might be < 128 for last batch
        dataload_time = perf_counter() - dataload_start
        total_dataloader_time += dataload_time

        # Skip computation if all targets for this batch already exist
        if all(os.path.exists(path) for path in save_paths):
            print(f"\nSkipping batch (Size: {batch_size}) - all embeddings already saved.")
            continue

        # Measure prompt embedding time
        prompt_start = perf_counter()
        with torch.no_grad():
            try:
                prompt_embs = text_encoder(texts, device)
            except Exception as e:
                print(f"Error encoding text: {e}")
                print(f"Text: {texts}")
                continue
        prompt_time = perf_counter() - prompt_start
        total_prompt_time += prompt_time

        # Measure saving time
        save_start = perf_counter()
        for save_path, prompt_emb in zip(save_paths, prompt_embs):
            data = prompt_emb.cpu()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(data, save_path)
        save_time = perf_counter() - save_start
        total_save_time += save_time

        total_batches += 1

        # Print timing for this batch
        print(f"\nBatch {total_batches} (Size: {batch_size}):")
        print(f"  Dataloader Time: {dataload_time:.4f} seconds")
        print(f"  Prompt Embedding Time: {prompt_time:.4f} seconds")
        print(f"  Data Saving Time: {save_time:.4f} seconds")
        print(f"  Total Batch Time: {dataload_time + prompt_time + save_time:.4f} seconds")

    # Print overall summary
    print(f"\nOverall Summary:")
    print(f"Total Dataloader Time: {total_dataloader_time:.4f} seconds")
    print(f"Average Dataloader Time per Batch: {total_dataloader_time/total_batches:.4f} seconds" if total_batches else "Average Dataloader Time per Batch: N/A")
    print(f"Total Prompt Embedding Time: {total_prompt_time:.4f} seconds")
    print(f"Average Prompt Embedding Time per Batch: {total_prompt_time/total_batches:.4f} seconds" if total_batches else "Average Prompt Embedding Time per Batch: N/A")
    print(f"Total Data Saving Time: {total_save_time:.4f} seconds")
    print(f"Average Data Saving Time per Batch: {total_save_time/total_batches:.4f} seconds" if total_batches else "Average Data Saving Time per Batch: N/A")
    print(f"Total Processing Time: {total_dataloader_time + total_prompt_time + total_save_time:.4f} seconds")
    print(f"Number of Batches Processed: {total_batches}")
    print(f"Total Samples Processed: {len(dataset)}")

    # add the text embedding path to the json file
    json_file = args.json_file
    data_list = json.load(open(json_file, 'r'))

    new_data_list = []
    text_key = args.text_key
    for idx, data in enumerate(data_list):
        if 'sample_id' in data:
            sample_id = data['sample_id']
        elif 'test_sample_id' in data:
            sample_id = str(data['test_sample_id'])
        elif 'global_id' in data:
            sample_id = str(data['global_id'])
        else:
            raise ValueError("No sample_id or test_sample_id found in the data.")
        data[f'{text_key}_wanvideot5_embed_path'] = os.path.join(args.save_dir, text_key, f"{sample_id}.pt")
        new_data_list.append(data)
    
    with open(json_file, 'w') as f:
        json.dump(new_data_list, f, indent=4)
