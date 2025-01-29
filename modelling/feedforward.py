from torch import nn

class PointWiseFeedForward(nn.Module):

    def __init__(self, d_model=512, d_f=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_f)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_f, out_features=d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = x.relu()
        x = self.linear2(x)
        x = self.dropout2(x)
        return x