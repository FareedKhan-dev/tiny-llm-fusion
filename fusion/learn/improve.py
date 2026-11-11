"""learn/improve.py: DPO on the team's own answers, as a LoRA adapter hot-loaded into vLLM."""
import httpx
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from fusion.config import GATEWAY_KEY, MODELS
from fusion.platform import gateway
from fusion.platform.episodes import DB

PAIRS = """SELECT a.prompt, a.reply, b.reply FROM ep a JOIN ep b ON a.task = b.task AND a.model = b.model
           WHERE a.model = ? AND a.ok = 1 AND b.ok = 0 GROUP BY a.task"""     # one right and one wrong reply per task


def mine_pairs(model):
    """Tasks where this model was both right and wrong, with its own replies on both sides."""
    turn = lambda role, text: [{"role": role, "content": text}]
    return Dataset.from_list([{"prompt": turn("user", p), "chosen": turn("assistant", c), "rejected": turn("assistant", r)}
                              for p, c, r in DB.execute(PAIRS, (model,))])


def train(model, pairs):
    """DPO plus an NLL term (rpo_alpha), LoRA rank 16, on the training VM. Returns the hub id."""
    args = DPOConfig(output_dir=f"{model}-dpo", beta=0.1, rpo_alpha=1.0, learning_rate=5e-6, num_train_epochs=2,
                     per_device_train_batch_size=4, gradient_accumulation_steps=8, bf16=True,
                     push_to_hub=True, hub_private_repo=True)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=MODELS[model].id, args=args, train_dataset=pairs, peft_config=lora)
    trainer.train()
    trainer.push_to_hub()
    return trainer.hub_model_id


def hot_load(model, repo):
    """Load the adapter into every running replica, then point the gateway at it. Nothing restarts."""
    for r in gateway.REPLICAS[model]:
        httpx.post(f"{r.url}/v1/load_lora_adapter", json={"lora_name": f"{model}-dpo", "lora_path": repo},
                   headers={"Authorization": f"Bearer {GATEWAY_KEY}"}, timeout=300).raise_for_status()
    gateway.ADAPTER[model] = f"{model}-dpo"
