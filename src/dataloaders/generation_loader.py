# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0



##### The dataloader for different tasks streams is implemented here.
import sys
sys.path.append(r'C:\Users\ntlinh\OneDrive - Hanoi University of Science and Technology\Documents\20242\PP\src')

import multiprocessing
import torch as th
from collections import defaultdict as ddict
import linecache
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from utils.utils import *
from datasets import load_dataset
from torch.utils.data import TensorDataset
import json


BIGQUERY = ['database', 'gui', 'networking', 'science', 'web']
class HFListDataset(Dataset):
    """
    Dataset đơn giản chứa sẵn input_ids và labels (đã tokenize).
    """
    def __init__(self, inputs, outputs, tokenizer, src_len, tgt_len):
        # token hóa ngay khi khởi tạo để DataLoader không cần collate_fn đặc biệt
        self.input_ids = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=src_len,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]

        with tokenizer.as_target_tokenizer():
            self.label_ids = tokenizer(
                outputs,
                padding="max_length",
                truncation=True,
                max_length=tgt_len,
                return_tensors="pt",
                return_attention_mask=False,
            )["input_ids"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.label_ids[idx]
def get_max_trg_len_by_task(task, sub_task):
    if task == "concode_none":
        task = "concode"
    if task == 'summarize':
        max_target_length = 128
    elif task == 'translate':
        max_target_length = 256
    elif task == 'refine':
        if sub_task == 'small':
            max_target_length = 120
        else:
            max_target_length = 240
    elif task == 'concode':
        max_target_length = 150
    elif task == 'defect':
        max_target_length = 3
    return max_target_length


def get_bs(cur_task, model_tag):
    task = cur_task.split('_')[0]
    sub_task = cur_task.split('_')[-1]
    if 'codet5_small' in model_tag:
        bs = 32
        if task == 'summarize' or task == 'translate' or (task == 'refine' and sub_task == 'small'):
            bs = 64
    else:
        # codet5_base
        bs = 28
        if task == 'translate':
            bs = 25
        elif task == 'summarize':
            bs = 40
    return bs
def get_data(task, split, sub_task=None):
    """
    Load data directly from HuggingFace datasets for a given task and split.
    Returns tuple of (inputs, outputs) lists.
    """
    # Map of tasks to HuggingFace dataset names and configurations
    task_to_dataset = {
        'summarize': ('code_x_glue_ct_code_to_text', None),
        'translate': ('code_x_glue_ct_code_to_code', 'java-cs'),
        'refine': ('code_x_glue_ct_code_refinement', None),
        'concode': ('code_x_glue_tc_text_to_code', None),
        'defect': ('code_x_glue_cc_defect_detection', None),
        'clone': ('code_x_glue_cc_clone_detection_big_clone_bench', None),
    }
    
    # Map tasks to their input/output field names in the dataset
    task_to_fields = {
        'summarize': ('code', 'summary'),
        'translate': ('java', 'cs'),
        'refine': ('source', 'target'),
        'concode': ('nl', 'code'),
        'defect': ('code', 'label'),
        'clone': ('code1', 'code2'),
    }
    
    # Map HF splits to CodeXGLUE splits
    split_mapping = {
        'train': 'train',
        'dev': 'validation',
        'validation': 'validation',
        'test': 'test'
    }
    hf_split = split_mapping.get(split, split)
    
    # Get dataset info
    dataset_name, config = task_to_dataset.get(task, (None, None))
    if not dataset_name:
        raise ValueError(f"Task {task} not supported for HuggingFace loading")
    
    # Get field names for input/output
    input_field, output_field = task_to_fields.get(task, (None, None))
    if not input_field or not output_field:
        raise ValueError(f"Field mapping not defined for task {task}")
    
    # Special handling for summarize task which needs language specification
    if task == 'summarize' and sub_task is not None:
        language = sub_task  # Use sub_task as language for summarize
        dataset = load_dataset(dataset_name, language, split=hf_split)
    else:
        dataset = load_dataset(dataset_name, config, split=hf_split)
    
    # Extract input and output data
    inputs = [example[input_field] for example in dataset]
    outputs = [str(example[output_field]) for example in dataset]  # Convert labels to string if needed
    
    return inputs, outputs, dataset


def get_args_by_task_model(task, sub_task=None, model_tag=None):
    # Khởi tạo giá trị mặc định cho tất cả biến
    ebs = 8
    tbs = 8
    lr = 5e-5
    src_len = 256
    trg_len = 128
    patience = 3
    epoch = 10
    
    # Cập nhật giá trị dựa trên task cụ thể
    if task == 'translate':
        src_len = 320
        trg_len = 256
        epoch = 100
        patience = 5
        tbs = 8
        ebs = 50
    elif task == 'summarize':
        src_len = 256
        trg_len = 128
        epoch = 15
        patience = 2
        tbs = 16
        ebs = 80
    elif task == 'refine':
        if sub_task == 'small':
            src_len = 130
            trg_len = 120
            tbs = 16
            ebs = 80
        elif sub_task == 'medium':
            src_len = 240
            trg_len = 240
            tbs = 8
            ebs = 50
        epoch = 50
        patience = 5
    elif task == 'concode':
        src_len = 320
        trg_len = 150
        epoch = 30
        patience = 3
        tbs = 16
        ebs = 50
        lr = 10e-5  # Ghi đè giá trị lr cho concode
    elif task == 'defect':
        src_len = 512
        trg_len = 3
        epoch = 10
        patience = 2
        tbs = 16
        ebs = 50
        lr = 2e-5  # Ghi đè giá trị lr cho defect
    elif task == 'clone':
        src_len = 400
        trg_len = 400
        epoch = 1
        patience = 2
        tbs = 8
        ebs = 50
    elif task in BIGQUERY:  # Giả sử BIGQUERY là một list hoặc set đã được định nghĩa trước đó
        src_len = 256
        trg_len = 256
        epoch = 10
        patience = 3
        tbs = 8
        ebs = 8
        lr = 5e-5  # Ghi đè giá trị lr cho BIGQUERY

    # Return đầy đủ tham số (tất cả biến đều đã được khởi tạo)
    return {"ebs": ebs, "tbs": tbs, "lr": lr, "src_len": src_len, "trg_len": trg_len, "patience": patience, "epoch": epoch}
class HFExample(object):
    """Example wrapper for HuggingFace dataset items."""
    def __init__(self, idx, source, target, task='', sub_task='', original_data=None):
        self.idx = idx
        self.source = source
        self.target = target
        self.task = task
        self.sub_task = sub_task
        self.original_data = original_data
class CodeXGlueDataModule:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.world_size = max(1, self.args.n_gpu)
        self.tokenizer = tokenizer
        self.pool = multiprocessing.Pool(self.args.cpu_cont)
        self.get_tasks_stream(args, args.stream, args.task)
        self.train_examples, self.train_data = {}, {}
        self.val_examples, self.val_data = {}, {}
        self.val_bleu_examples, self.val_bleu_data = {}, {}
        self.test_examples, self.test_data = {}, {}

    def get_tasks_stream(self, args, stream=None, taskname=None):
        """
        Thiết lập self.all_tasks, self.task_params, self.filenames
        - stream: chuỗi "task1,task2_sub2,..."
        - taskname: (không dùng ở đây)
        """
        # 1) Xác định danh sách task
        if stream == 'summarize':
            sub_tasks = ['java', 'php', 'javascript', 'ruby', 'python', 'go']
            self.all_tasks = [f"summarize_{st}" for st in sub_tasks]
        elif stream is not None:
            self.all_tasks = stream.split(',')
        else:
            raise NotImplementedError(f"Stream {stream} is not implemented")

        # 2) Chuẩn bị dicts
        self.task_params = ddict(dict)
        self.filenames   = ddict(dict)

        # 3) Cho mỗi task, phân tách base/sub rồi gán params + filenames
        for task in self.all_tasks:
            # tách base và sub_task (nếu có '_' thì sub là phần sau, còn không sub="")
            parts = task.split('_', 1)
            base_task = parts[0]
            sub_task  = parts[1] if len(parts) > 1 else ""

            # lấy các tham số học theo base/sub
            self.task_params[task] = get_args_by_task_model(
                base_task, sub_task, args.model_name_or_path
            )

            # định nghĩa đường dẫn dữ liệu: data_dir/base/sub/…
            train_fn, val_fn, test_fn = get_filenames(
                args.data_dir, base_task, sub_task, ''
            )
            self.filenames[task]['train'] = train_fn
            self.filenames[task]['val']   = val_fn
            self.filenames[task]['test']  = test_fn

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        print("SETTING UP THE DATA")
        for curr_idx, curr_task in enumerate(self.all_tasks):
            parts = curr_task.split('_', 1)
            task = parts[0]
            sub_task = parts[1] if len(parts) > 1 else ""
            
            # Set filenames for reference
            self.train_filename = self.filenames[curr_task]['train']
            self.dev_filename = self.filenames[curr_task]['val']
            self.test_filename = self.filenames[curr_task]['test']
            
            # Setup training data
            if stage in ("fit", None):
                if self.args.use_hf_dataset:
                    try:
                        hf_inputs, hf_outputs, raw_dataset = get_data(
                            task, 'train', sub_task=sub_task if sub_task else None
                        )
                        # Create examples for compatibility with bleu evaluation
                        self.train_examples[curr_task] = [
                            HFExample(idx=i, source=src, target=tgt, task=task, sub_task=sub_task, original_data=raw_item)
                            for i, (src, tgt, raw_item) in enumerate(zip(hf_inputs, hf_outputs, raw_dataset))
                        ]
                        self.train_data[curr_task] = HFListDataset(
                            hf_inputs, hf_outputs, self.tokenizer,
                            self.args.max_source_length[curr_idx],
                            self.args.max_target_length[curr_idx]
                        )
                    except Exception as e:
                        print(f"Error loading HF dataset for {curr_task} (train): {e}")
                        print("Falling back to local files")
                        # Fallback to local files
                        self.train_examples[curr_task], self.train_data[curr_task] = \
                            load_and_cache_gen_data(
                                self.args, task, sub_task, self.train_filename,
                                self.pool, self.tokenizer, 'train',
                                self.args.max_source_length[curr_idx],
                                self.args.max_target_length[curr_idx],
                                curr_task=curr_task)
                else:
                    # Use local files as before
                    self.train_examples[curr_task], self.train_data[curr_task] = \
                        load_and_cache_gen_data(
                            self.args, task, sub_task, self.train_filename,
                            self.pool, self.tokenizer, 'train',
                            self.args.max_source_length[curr_idx],
                            self.args.max_target_length[curr_idx],
                            curr_task=curr_task)
            
            # Setup validation data
            if stage in ("fit", None):
                if self.args.use_hf_dataset:
                    try:
                        hf_inputs, hf_outputs, raw_dataset = get_data(
                            task, 'dev', sub_task=sub_task if sub_task else None
                        )
                        self.val_examples[curr_task] = [
                            HFExample(idx=i, source=src, target=tgt, task=task, sub_task=sub_task, original_data=raw_item)
                            for i, (src, tgt, raw_item) in enumerate(zip(hf_inputs, hf_outputs, raw_dataset))
                        ]
                        self.val_data[curr_task] = HFListDataset(
                            hf_inputs, hf_outputs, self.tokenizer,
                            self.args.max_source_length[curr_idx],
                            self.args.max_target_length[curr_idx]
                        )
                    except Exception as e:
                        print(f"Error loading HF dataset for {curr_task} (validation): {e}")
                        print("Falling back to local files")
                        self.val_examples[curr_task], self.val_data[curr_task] = \
                            load_and_cache_gen_data(
                                self.args, task, sub_task, self.dev_filename,
                                self.pool, self.tokenizer, 'dev',
                                self.args.max_source_length[curr_idx],
                                self.args.max_target_length[curr_idx],
                                curr_task=curr_task)
                else:
                    self.val_examples[curr_task], self.val_data[curr_task] = \
                        load_and_cache_gen_data(
                            self.args, task, sub_task, self.dev_filename,
                            self.pool, self.tokenizer, 'dev',
                            self.args.max_source_length[curr_idx],
                            self.args.max_target_length[curr_idx],
                            curr_task=curr_task)
            
            # Setup test data
            if stage in ("test", None):
                if self.args.use_hf_dataset:
                    try:
                        hf_inputs, hf_outputs, raw_dataset = get_data(
                            task, 'test', sub_task=sub_task if sub_task else None
                        )
                        self.test_examples[curr_task] = [
                            HFExample(idx=i, source=src, target=tgt, task=task, sub_task=sub_task, original_data=raw_item)
                            for i, (src, tgt, raw_item) in enumerate(zip(hf_inputs, hf_outputs, raw_dataset))
                        ]
                        self.test_data[curr_task] = HFListDataset(
                            hf_inputs, hf_outputs, self.tokenizer,
                            self.args.max_source_length[curr_idx],
                            self.args.max_target_length[curr_idx]
                        )
                    except Exception as e:
                        print(f"Error loading HF dataset for {curr_task} (test): {e}")
                        print("Falling back to local files")
                        self.test_examples[curr_task], self.test_data[curr_task] = \
                            load_and_cache_gen_data(
                                self.args, task, sub_task, self.test_filename,
                                self.pool, self.tokenizer, 'test',
                                self.args.max_source_length[curr_idx],
                                self.args.max_target_length[curr_idx],
                                curr_task=curr_task)
                else:
                    self.test_examples[curr_task], self.test_data[curr_task] = \
                        load_and_cache_gen_data(
                            self.args, task, sub_task, self.test_filename,
                            self.pool, self.tokenizer, 'test',
                            self.args.max_source_length[curr_idx],
                            self.args.max_target_length[curr_idx],
                            curr_task=curr_task)
        
        self.total_train_data_num = sum([len(ds) for ds in self.train_data.values()])
    def train_dataloader(self):
        """Updated train_dataloader to handle HF datasets"""
        self.train_loaders = {}
        for curr_idx, curr_task in enumerate(self.all_tasks):
            # Get the dataset for this task
            dataset = self.train_data[curr_task]
            
            # Create sampler based on dataset type
            sampler = self.get_sampler(dataset, dist=False, nondist='random')
            
            # For HFListDataset, no special collate_fn is needed
            # For other dataset types, use the original approach
            batch_size = self.world_size * self.args.train_batch_size[curr_idx]
            
            self.train_loaders[curr_task] = DataLoader(
                dataset, 
                batch_size=batch_size,
                sampler=sampler, 
                num_workers=self.args.num_workers, 
                pin_memory=self.args.pin_memory
            )
        
        return self.train_loaders
    
    def get_key_prompt_init_dataloaders(self, n_prompts):
        """Updated to handle HF datasets"""
        per_task_init = n_prompts // len(self.all_tasks)
        query_key_init_dataloaders = {}
        for ti, task in enumerate(self.all_tasks):
            num_task_prompts = (n_prompts - (per_task_init * (len(self.all_tasks)-1))) if ti == 0 else per_task_init
            task_dataset = self.train_data[task]
            num_task_examples = len(task_dataset)
            # Thay dòng assertion
            if num_task_examples < num_task_prompts:
                print(f"Warning: Task {task} has only {num_task_examples} examples but {num_task_prompts} prompts requested.")
                num_task_prompts = num_task_examples  # Điều chỉnh số prompts xuống bằng số ví dụ
            
            # Get random indices
            task_indices = torch.randperm(num_task_examples)[:num_task_prompts]
            
            # Check if using HFListDataset
            if isinstance(task_dataset, HFListDataset):
                # For HFListDataset, create a subset using the indices
                input_ids = task_dataset.input_ids[task_indices]
                label_ids = task_dataset.label_ids[task_indices]
                key_init_dataset = TensorDataset(input_ids, label_ids)
            else:
                # For existing dataset formats, use original approach
                input_ids, label_ids = [], []
                for idx in task_indices:
                    i_ids, l_ids = task_dataset[idx]
                    input_ids.append(i_ids)
                    label_ids.append(l_ids)
                key_init_dataset = TensorDataset(torch.stack(input_ids), torch.stack(label_ids))
            
            key_init_loader = DataLoader(
                key_init_dataset, 
                batch_size=self.args.eval_batch_size[ti],
                sampler=torch.utils.data.SequentialSampler(key_init_dataset),
                num_workers=self.args.num_workers, 
                pin_memory=self.args.pin_memory
            )
            query_key_init_dataloaders[task] = key_init_loader
        
        return query_key_init_dataloaders
    
    def val_dataloader(self):
        """Updated val_dataloader to handle HF datasets"""
        self.val_dataloaders = {}
        for curr_idx, curr_task in enumerate(self.all_tasks):
            # Get the dataset for this task
            dataset = self.val_data[curr_task]
            
            # Create sequential sampler
            sampler_ppl = self.get_sampler(dataset, dist=False, nondist='sequential')
            
            # Create dataloader - no special collate_fn needed for HFListDataset
            batch_size = self.world_size * self.args.train_batch_size[curr_idx]
            
            self.val_dataloaders[curr_task] = DataLoader(
                dataset, 
                batch_size=batch_size,
                sampler=sampler_ppl, 
                num_workers=self.args.num_workers, 
                pin_memory=self.args.pin_memory
            )
        
        return self.val_dataloaders

    def get_bleu_dataloader(self, curr_task, all_bleu=False):
        """Updated to handle HF datasets"""
        task = curr_task.split('_')[0]
        sub_task = curr_task.split('_')[1] if '_' in curr_task else ""
        curr_idx = self.all_tasks.index(curr_task)
        self.dev_filename = self.filenames[curr_task]['val']

        if all_bleu:
            bleu_samples = 1000000000
        else:
            bleu_samples = self.args.bleu_samples

        # If we're using HF dataset and have already loaded val data
        if self.args.use_hf_dataset and curr_task in self.val_examples and self.val_examples[curr_task]:
            # Use existing examples, but limit to bleu_samples
            val_examples = self.val_examples[curr_task][:bleu_samples]
            
            # Create input and target lists for BLEU evaluation
            input_texts = [example.source for example in val_examples]
            target_texts = [example.target for example in val_examples]
            
            # Create tokenized dataset
            val_dataset = HFListDataset(
                input_texts, target_texts, self.tokenizer,
                self.args.max_source_length[curr_idx],
                self.args.max_target_length[curr_idx]
            )
            
            # Create DataLoader
            sampler_bleu = self.get_sampler(val_dataset, dist=False, nondist='sequential')
            self.bleu_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.args.eval_batch_size[curr_idx], 
                sampler=sampler_bleu,
                num_workers=self.args.num_workers, 
                pin_memory=False
            )
            return val_examples, val_dataset, self.bleu_dataloader
        else:
            # Fallback to original method
            self.val_bleu_examples, self.val_bleu_data = load_and_cache_gen_data(
                self.args, task, sub_task, self.dev_filename,
                self.pool, self.tokenizer, 'dev',
                self.args.max_source_length[curr_idx],
                self.args.max_target_length[curr_idx],
                only_src=False, is_sample=True, bleu_samples=bleu_samples
            )
            sampler_bleu = self.get_sampler(self.val_bleu_data, dist=False, nondist='sequential')
            self.bleu_dataloader = DataLoader(
                self.val_bleu_data, 
                batch_size=self.args.eval_batch_size[curr_idx], 
                sampler=sampler_bleu,
                num_workers=self.args.num_workers, 
                pin_memory=False
            )
            return self.val_bleu_examples, self.val_bleu_data, self.bleu_dataloader
    def test_dataloader(self):
        """Updated test_dataloader to handle HF datasets"""
        self.test_dataloaders = {}
        for curr_idx, curr_task in enumerate(self.all_tasks):
            # Get the dataset for this task
            dataset = self.test_data[curr_task]
            
            # Create sequential sampler
            sampler = self.get_sampler(dataset, dist=False, nondist='sequential')
            
            # Determine batch size based on zeroshot flag
            bs = self.args.eval_batch_size[0] if self.args.zeroshot else self.args.eval_batch_size[curr_idx]
            
            # Create a custom collate function to include examples if needed
            if self.args.use_hf_dataset and isinstance(dataset, HFListDataset):
                # For HFListDataset with examples needed for evaluation
                def collate_fn_with_examples(batch):
                    input_ids = torch.stack([item[0] for item in batch])
                    target_ids = torch.stack([item[1] for item in batch])
                    batch_indices = [idx for idx in range(len(batch))]
                    examples = [self.test_examples[curr_task][idx] for idx in batch_indices]
                    return input_ids, target_ids, examples
                
                self.test_dataloaders[curr_task] = DataLoader(
                    dataset, 
                    batch_size=bs, 
                    sampler=sampler,
                    num_workers=self.args.num_workers, 
                    pin_memory=self.args.pin_memory,
                    collate_fn=collate_fn_with_examples if self.test_examples.get(curr_task) else None
                )
            else:
                # For original dataset types
                self.test_dataloaders[curr_task] = DataLoader(
                    dataset, 
                    batch_size=bs, 
                    sampler=sampler,
                    num_workers=self.args.num_workers, 
                    pin_memory=self.args.pin_memory
                )
        
        return self.test_examples, self.test_data, self.test_dataloaders

    def get_sampler(self, data, dist=False, nondist='sequential'):
        if nondist == 'sequential':
            sampler = th.utils.data.SequentialSampler(data)
        elif nondist == 'random':
            sampler = th.utils.data.RandomSampler(data)
        else:
            raise NotImplementedError()
        if dist and self.args.local_rank != -1:
            sampler = th.utils.data.distributed.DistributedSampler(data, shuffle=False)
        return sampler


class BigQueryDataset(Dataset):
    def __init__(self, args, tokenizer, filename, task_params):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.load_file(filename, num_lines=args.data_num)
        self.task_params = task_params
        pass

    @staticmethod
    def load_file(filepath, num_lines=-1):
        with open(f'{filepath}') as f:
            lines = f.read().splitlines()
        if num_lines > 0:
            return lines[:num_lines]
        else:
            return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = json.loads(self.data[index])['content'].rstrip("\n")
        return {'line': sample, 'id': index}

    def collate_fn(self, batch):
        """Call prepare_seq2seq_batch."""

        source_batch = []
        target_batch = []
        for sample in batch:
            lines = sample['line'].splitlines()
            split_index = np.random.randint(len(lines)) - 1
            source_batch.append('\n'.join(l for l in lines[:split_index]))
            target_batch.append('\n'.join(l for l in lines[split_index:]))

        input_ids = self.tokenizer(
            source_batch,
            max_length=self.task_params['src_len'],
            padding='max_length', truncation=True,
            return_tensors="pt", return_attention_mask=False
        )['input_ids']
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_batch,
                max_length=self.task_params['trg_len'],
                padding='max_length', truncation=True,
                return_tensors="pt", return_attention_mask=False
            )['input_ids']

        return input_ids, labels

    def bleu_collate_fn(self, batch):
        """Call prepare_seq2seq_batch."""

        source_batch = []
        target_batch = []
        examples = []
        for sample in batch:
            lines = sample['line'].splitlines()
            split_index = np.random.randint(len(lines)) - 1
            source = '\n'.join(l for l in lines[:split_index])
            target = '\n'.join(l for l in lines[split_index:])
            source_batch.append(source)
            target_batch.append(target)
            examples.append(Example(
                    idx=sample['id'],
                    source=source,
                    target=target,
                ))

        input_ids = self.tokenizer(
            source_batch,
            max_length=self.task_params['src_len'],
            padding='max_length', truncation=True,
            return_tensors="pt", return_attention_mask=False
        )['input_ids']
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_batch,
                max_length=self.task_params['trg_len'],
                padding='max_length', truncation=True,
                return_tensors="pt", return_attention_mask=False
            )['input_ids']


        return input_ids, labels, examples

    def test_collate_fn(self, batch):
        """Call prepare test batch."""

        source_batch = []
        target_batch = []
        examples = []
        for sample in batch:
            lines = sample['line'].splitlines()
            source = '\n'.join(l for l in lines[:-1])
            target = '\n'.join(l for l in lines[-1:])
            source_batch.append(source)
            target_batch.append(target)
            examples.append(Example(
                    idx=sample['id'],
                    source=source,
                    target=target,
                ))

        input_ids = self.tokenizer(
            source_batch,
            max_length=self.task_params['src_len'],
            padding='max_length', truncation=True,
            return_tensors="pt", return_attention_mask=False
        )['input_ids']
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_batch,
                max_length=self.task_params['trg_len'],
                padding='max_length', truncation=True,
                return_tensors="pt", return_attention_mask=False
            )['input_ids']

        return input_ids, labels, examples

class BigQueryDataModule:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.world_size = self.args.n_gpu

        self.tokenizer = tokenizer
        self.all_tasks = args.stream.split(',')
        self.task_params = ddict(dict)
        self.filenames = ddict(dict)
        self.bigquery_dir = os.path.join(args.project_dir, args.data_dir, 'bigquery')
        for task in self.all_tasks:
            self.task_params[task] = get_args_by_task_model(task)
            self.filenames[task]['train'] = os.path.join(self.bigquery_dir, f"{task}_train.jsonl")
            self.filenames[task]['val'] = os.path.join(self.bigquery_dir, f"{task}_val.jsonl")
            self.filenames[task]['test'] = os.path.join(self.bigquery_dir, f"{task}_test.jsonl")
            assert all([os.path.exists(p) for  p in self.filenames[task].values()]), f'Atleast one of the files for task {task} does not exist.'

        self.train_examples, self.train_data = {}, {}
        self.val_examples, self.val_data = {}, {}
        self.val_bleu_examples, self.val_bleu_data = {}, {}
        self.test_examples, self.test_data = {}, {}


    def setup(self, stage = None):
        # Assign train/val/test datasets for use in dataloaders
        print("SETTING UP THE DATA")
        for curr_idx, curr_task in enumerate(self.all_tasks):
            # self.args.max_source_length = self.task_params[curr_task]['src_len']
            # self.args.max_target_length = self.task_params[curr_task]['trg_len']
            train_filename = self.filenames[curr_task]['train']
            val_filename = self.filenames[curr_task]['val']
            test_filename = self.filenames[curr_task]['test']
            task_params = self.task_params[curr_task]

            if stage == "fit" or stage is None:
                self.train_data[curr_task] = BigQueryDataset(self.args, self.tokenizer, train_filename, task_params)
                self.val_data[curr_task] = BigQueryDataset(self.args, self.tokenizer, val_filename, task_params)

            if stage == "test" or stage is None:
                self.test_data[curr_task] = BigQueryDataset(self.args, self.tokenizer, test_filename, task_params)

        self.total_train_data_num = sum([len(ds) for ds in self.train_data.values()])

    def get_key_prompt_init_dataloaders(self, n_prompts):
        per_task_init = n_prompts // len(self.all_tasks)
        query_key_init_dataloaders = {}
        for ti, task in enumerate(self.all_tasks):
            num_task_prompts = (n_prompts - (per_task_init * (len(self.all_tasks)-1))) if ti == 0 else per_task_init
            task_dataset = self.train_data[task]
            num_task_examples = len(task_dataset)
            assert num_task_examples >= num_task_prompts, "Not enough examples for the dataset."
            task_indices = torch.randperm(num_task_examples)[:num_task_prompts]
            task_input_ids, task_target_ids = task_dataset[task_indices]
            key_init_dataset = TensorDataset(task_input_ids, task_target_ids)
            key_init_loader = DataLoader(key_init_dataset, batch_size=self.args.eval_batch_size[ti], \
                                            sampler=torch.utils.data.SequentialSampler(key_init_dataset),\
                                            num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
            query_key_init_dataloaders[task] = key_init_loader
        return query_key_init_dataloaders

    def train_dataloader(self):
        self.train_loaders = {}
        for curr_idx, curr_task in enumerate(self.all_tasks):
            data = self.train_data[curr_task]
            sampler = self.get_sampler(data, dist=False, nondist='random')
            self.train_loaders[curr_task] = DataLoader(data, batch_size=self.world_size * self.args.train_batch_size[curr_idx], \
                                            sampler=sampler, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory,
                                            collate_fn=data.collate_fn)
        return self.train_loaders

    def val_dataloader(self):
        self.val_dataloaders = {}
        for curr_idx, curr_task in enumerate(self.all_tasks):
            data = self.val_data[curr_task]
            sampler_ppl = self.get_sampler(data, dist=False, nondist='sequential')
            self.val_dataloaders[curr_task] = DataLoader(data, batch_size=self.world_size * self.args.train_batch_size[curr_idx], \
                                                sampler=sampler_ppl,num_workers=self.args.num_workers, pin_memory=self.args.pin_memory,
                                                collate_fn=data.collate_fn)
        return self.val_dataloaders

    def get_bleu_dataloader(self, curr_task, all_bleu=False):
        curr_idx = self.all_tasks.index(curr_task)
        self.dev_filename = self.filenames[curr_task]['val']

        if all_bleu:
            bleu_samples = 1000000000
        else:
            bleu_samples = self.args.bleu_samples


        data = self.val_data[curr_task]
        subset_data = Subset(data, range(len(data))[:bleu_samples])
        sampler_bleu = self.get_sampler(subset_data, dist=False, nondist='sequential')
        self.bleu_dataloader = DataLoader(subset_data, batch_size=self.args.eval_batch_size[curr_idx], sampler=sampler_bleu,\
                                    num_workers=self.args.num_workers, pin_memory=False,
                                    collate_fn=data.bleu_collate_fn)
        return self.bleu_dataloader

    def test_dataloader(self):
        self.test_dataloaders = {}
        for curr_idx, curr_task in enumerate(self.all_tasks):
            data = self.test_data[curr_task]
            sampler = self.get_sampler(data, dist=False, nondist='sequential')
            bs = self.args.eval_batch_size [0] if self.args.zeroshot else self.args.eval_batch_size[curr_idx]
            self.test_dataloaders[curr_task] =  DataLoader(data, batch_size=bs, sampler=sampler,\
                num_workers=self.args.num_workers, pin_memory=self.args.pin_memory,
                collate_fn=data.test_collate_fn)
        return self.test_dataloaders

    def get_sampler(self, data, dist=False, nondist='sequential'):
        if nondist == 'sequential':
            sampler = th.utils.data.SequentialSampler(data)
        elif nondist == 'random':
            sampler = th.utils.data.RandomSampler(data)
        else:
            raise NotImplementedError()
        if dist and self.args.local_rank != -1:
            sampler = th.utils.data.distributed.DistributedSampler(data, shuffle=False)
        return sampler


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task