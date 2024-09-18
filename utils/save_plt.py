import matplotlib.pyplot as plt
from typing import Dict, List


save_path = '../GNAE/output_four/no_norm_four_delete_dim'

def save_loss_plot(losses: Dict[str, List[float]], path) -> None:
    """
    Saves loss plot.

    If validation loss is present, the plot includes both training and validation loss; otherwise, it includes only
    the training loss.

    Parameters
    ----------
    losses : dict
        A dictionary contains lists of losses. The keys are "tloss_e" for training loss and "vloss_e" for validation
        loss. The values are lists of recorded loss values.
    plots_path : str
        Path to save the loss plot.
    """
    path = save_path
    x_axis = list(range(len(losses["tloss_e"])))
    plt.plot(x_axis, losses["tloss_e"], c='r', label="Training")
    title = "Training"
    if len(losses["vloss_e"]) >= 1:
        # If validation loss is recorded less often, we need to adjust x-axis values by the factor of difference
        beta = len(losses["tloss_e"]) / len(losses["vloss_e"])
        x_axis = list(range(len(losses["vloss_e"])))
        # Adjust the values of x-axis by beta factor
        x_axis = [beta * i for i in x_axis]
        plt.plot(x_axis, losses["vloss_e"], c='b', label="Validation")
        title += " and Validation "
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title(title + " Loss", fontsize=12)
    plt.tight_layout()
    plt.savefig(path + "/loss.png")
    plt.clf()
