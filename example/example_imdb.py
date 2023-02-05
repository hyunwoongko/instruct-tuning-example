import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
    PPOTrainer,
)
from trl.core import LengthSampler


def tokenize(sample):
    # cut input tensor to use it as prompt.
    sample["input_ids"] = tokenizer.encode(sample["review"])[: input_length_sampler()]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


def collate_fn(data):
    return {key: [d[key] for d in data] for key in data[0]}


ppo_config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.5e-5,
    log_with="wandb",
    batch_size=8,
)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": ppo_config.forward_batch_size,
}

# Forward batching: Since the models can be fairly big, and we want to roll out large PPO batches.
# this can lead to out-of-memory errors when doing the forward passes for text generation and sentiment analysis.
# We introduce the parameter `forward_batch_size` to split the forward passes into smaller batches.
# Although this hurts performance a little, but this is neglectible compared to the computations of the backward passes when optimizing the model.
# The same parameter is used in the PPOTrainer when doing forward passes.
# The batch_size should multiple of forward_batch_size.

# You can see that we load a GPT2 model called gpt2_imdb.
# This model was additionally fine-tuned on the IMDB dataset for 1 epoch with the huggingface script (no special settings).
# The other parameters are mostly taken from the original paper "Fine-Tuning Language Models from Human Preferences".
# This model as well as the BERT model is available in the Huggingface model zoo here.
# The following code should automatically download the models.

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load models
model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name).cuda()
ref_model = create_reference_model(model)

# Load dataset
dataset = load_dataset("imdb", split="train")
dataset = dataset.rename_columns({"text": "review"})
dataset = dataset.filter(lambda x: len(x["review"]) > 200, batched=False)

input_length_sampler = LengthSampler(min_value=2, max_value=8)
dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type="torch")

# Initialize PPOTrainer
ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model,
    tokenizer,
    dataset=dataset,
    data_collator=collate_fn,
)

# Load reward model
device = ppo_trainer.accelerator.device
reward_model = pipeline(
    "sentiment-analysis",
    model="lvwerra/distilbert-imdb",
    device=device.index,
)

gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

output_length_sampler = LengthSampler(min_value=4, max_value=16)

for epoch, batch in tqdm(
    enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)
):
    query_tensors = [i.cuda() for i in batch["input_ids"]]
    response_tensors = []

    print("Generating responses...")
    for query in query_tensors:
        gen_len = output_length_sampler()
        gen_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    print("Calculating rewards...")
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    sentiment_output = reward_model(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in sentiment_output]
    # output[1] means positive sentiment score

    print("Updating model...")
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
