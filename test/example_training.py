from __future__ import annotations

from random import random
from time import sleep

from dlworkflow import log_training_run

MODEL_NAME = "tiny-mlp"
DATASET_NAME = "demo-dataset"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
TRAINING_NOTE = "Pseudo training example for inspecting decorator-based run logging."


@log_training_run(filename="example_pseudo_training")
def train_model() -> dict[str, float]:
    train_loss = 1.25
    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        sleep(0.1)
        train_loss *= 0.72
        val_accuracy = min(0.99, 0.55 + epoch * 0.09 + random() * 0.03)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    return {
        "final_train_loss": round(train_loss, 4),
        "best_val_accuracy": round(best_val_accuracy, 4),
    }


if __name__ == "__main__":
    result = train_model()
    print("training_result:", result)
