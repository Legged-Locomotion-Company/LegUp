from torch import nn


class Student(nn.Module):
    def __init__(self, cfg):
        super(Student, self).__init__()
        self.cfg = cfg

        self.height_encoder = nn.Sequential(
            nn.Linear(51, 80),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 60),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(60, 24),
        )

        # TODO: finish this later
        self.encoder = nn.Sequential(
            nn.GRU(24, 50, 1, batch_first=True),
        )
