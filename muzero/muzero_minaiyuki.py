import random
from collections import deque
import math
import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data
import numpy as np
import collections
from collections import deque
import gym
import itertools
import random
import os
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Set your device
n_qubit = 4
dev = qml.device('default.qubit', wires=n_qubit)