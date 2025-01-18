import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from scipy.stats import gaussian_kde
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data
from model import NQEModel
from utils import generate_layers


def plot_distributions_interactive(how_many_layer_list, data_dict, filename):
    """
    Creates an interactive plot with multiple distributions using Plotly.
    Args:
        data_dict (dict): A dictionary where keys are indices and values are lists of data points.
    """
    fig = go.Figure()
    # Iterate over the dictionary and add traces for each distribution
    for key, values in data_dict.items():
        # Perform Gaussian KDE for smoothing
        kde = gaussian_kde(values)
        x_vals = np.linspace(min(values), max(values), 1000)  # Generate a smooth range for x-axis
        y_vals = kde(x_vals)  # Evaluate the density over the range
        # Add a trace for the distribution curve
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name=f"Index {key}",
            line=dict(width=2),
            fill='tozeroy',
            opacity=0.6
        ))
    # Update layout with title and axis labels
    fig.update_layout(
        title=f"{how_many_layer_list}-{how_many_layer_list+5} Same",
        xaxis_title="Value",
        yaxis_title="Density",
        legend_title="Distributions",
        template="plotly_white",
        hovermode="x unified"
    )
    # Show the plot
    pio.write_html(fig, file=f'{filename}.html', auto_open=True)



if __name__ == "__main__":
    num_qubit = 4
    max_epoch_NQE = 100
    batch_size = 50
    num_layer = 64
    lr_NQE = 0.01
    num_gate_class = 5
    how_many_layer_list = 20

    layer_set = generate_layers(num_qubit, num_layer)
    X_train, X_test, Y_train, Y_test = dataprep(dataset='kmnist', reduction_sz=num_qubit)

    loss_fn = nn.MSELoss()

    layer_list_dict = {
        k: [torch.tensor(torch.randint(0, 64, (1,)).item()) for _ in range(5)]
        for k in range(how_many_layer_list)
    }

    same_value = [torch.tensor(torch.randint(0, 64, (1,)).item()) for _ in range(5)]
    for k in range(how_many_layer_list, how_many_layer_list+5):
        layer_list_dict[k] = same_value


    loss_dict = {}
    for index, layer_list in layer_list_dict.items():
        gate_list = [item for i in layer_list for item in layer_set[int(i)]]
        NQE_model = NQEModel(gate_list)
        NQE_model.train()
        NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=lr_NQE)

        for _ in range(max_epoch_NQE):
            X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
            pred = NQE_model(X1_batch, X2_batch)
            loss = loss_fn(pred, Y_batch)

            NQE_opt.zero_grad()
            loss.backward()
            NQE_opt.step()

        valid_loss_list = []
        NQE_model.eval()
        for _ in range(batch_size):
            X1_batch, X2_batch, Y_batch = new_data(batch_size, X_test, Y_test)
            with torch.no_grad():
                pred = NQE_model(X1_batch, X2_batch)
            valid_loss_list.append(loss_fn(pred, Y_batch).item())

        loss_dict[index] = valid_loss_list

    plot_distributions_interactive(how_many_layer_list, loss_dict, 'test')

