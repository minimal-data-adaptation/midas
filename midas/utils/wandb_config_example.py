"""Optional W&B settings example; prefer environment variables in production."""


def get_wandb_config() -> dict[str, str]:
    return {
        "WANDB_API_KEY": "",
        "WANDB_EMAIL": "",
        "WANDB_USERNAME": "",
        "WANDB_TEAM": "",
    }
