import os
from typing import Iterable
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DuDQN(nn.Module):
    def __init__(self, lr: float, n_actions: int, name: str,
                input_dims: Iterable[int], checkpoint_dir: str) -> None:
        super().__init__()
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(self.fc_input_dims, 512)
        
        self.V = nn.Linear(512, 1)  # Value function
        self.A = nn.Linear(512, n_actions) # Advantage function
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))
    
    def forward(self, state):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        state = state.view(state.size()[0], -1)

        state = F.relu(self.fc1(state))
        V = self.V(state)
        A = self.A(state)
        
        return V, A
    
    def save_checkpoint(self):
        # sys.stdout.write("... Saving Checkpoint ...\n")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print("... Loading Checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
        