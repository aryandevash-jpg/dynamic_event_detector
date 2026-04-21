---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:18900
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: omg young girl provides home for stray bullet 😳
  sentences:
  - young girl provides home for stray bullet sounds made up but here we are 😭
  - Mystery China Pneumonia Outbreak Likely Caused by New Human Coronavirus via
  - 8.30am Castlebar - two vehicle collision at the jnctn of Main St &amp; Market
    St, Parsons Corner. AVOID driving up from…
- source_sentence: wtf this headline is so unserious 🚨
  sentences:
  - timeline is in chaos rn 😭
  - popeye's sign town's tallest monument
  - im shook because copycat criminals continue to mimic liquor store robbery is crazy
    work 🤯
- source_sentence: delulu timeline is undefeated
  sentences:
  - History is going to record that the US tried to kickstart a war and it backfired
    when the target accidentally shot…
  - everything burning so fast rn 🚨
  - 8.30am Castlebar - two vehicle collision at the jnctn of Main St &amp; Market
    St, Parsons Corner. AVOID driving up from…
- source_sentence: bro cant believe this 😳
  sentences:
  - Abuja furniture shop razed by fire
  - previously unknown prejudice against japanese surfaces during game of battleship
  - captain's hat really completes street lunatic's ensemble
- source_sentence: plane news getting worse 💥
  sentences:
  - bro said mike pence vows conservation funding discovering and expected us to be
    normal
  - warning repressive forces islamic republic fired but make it a meme page post
  - bro cant believe this 😭
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'plane news getting worse 💥',
    'bro said mike pence vows conservation funding discovering and expected us to be normal',
    'bro cant believe this 😭',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.0279, 0.9545],
#         [0.0279, 1.0000, 0.0504],
#         [0.9545, 0.0504, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 18,900 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                           |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                           |
  | details | <ul><li>min: 6 tokens</li><li>mean: 12.08 tokens</li><li>max: 33 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 16.86 tokens</li><li>max: 48 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.39</li><li>max: 0.95</li></ul> |
* Samples:
  | sentence_0                                        | sentence_1                                                                                                                        | label             |
  |:--------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|:------------------|
  | <code>nah this earthquake news is insane 😱</code> | <code>Both drivers were standing nearby, hopefully it was a low impact collision.</code>                                          | <code>0.9</code>  |
  | <code>bro cant believe this 😭</code>              | <code>Isolated thunderstorm accompanied with hail/lightning also likely over Punjab, Haryana, Chandigarh, Rajasthan, Utta…</code> | <code>0.9</code>  |
  | <code>wtf this headline is so unserious 🚨</code>  | <code>im shook because victoria's secret also andrew's secret is crazy work 🤯</code>                                              | <code>0.75</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4230 | 500  | 0.0546        |
| 0.8460 | 1000 | 0.0026        |
| 1.2690 | 1500 | 0.0012        |
| 1.6920 | 2000 | 0.0008        |
| 2.1151 | 2500 | 0.0006        |
| 2.5381 | 3000 | 0.0005        |
| 2.9611 | 3500 | 0.0005        |


### Training Time
- **Training**: 5.7 minutes

### Framework Versions
- Python: 3.13.12
- Sentence Transformers: 5.4.1
- Transformers: 5.5.4
- PyTorch: 2.11.0
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->