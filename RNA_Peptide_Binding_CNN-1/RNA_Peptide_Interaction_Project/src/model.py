# Define CNN model - Step 14
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    super(CNN, self).__init__()
    self.rna_conv1 = nn.Conv1d(4, 16, kernel_size=3, padding=1)
    self.rna_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
    self.rna_conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
    self.rna_pool = nn.AdaptiveAvgPool1d(1)
    self.pep_conv1 = nn.Conv1d(20, 16, kernel_size=3, padding=1)
    self.pep_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
    self.pep_conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
    self.pep_pool = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(128, 64)
    self.fc2 = nn.Linear(64, 1)

  def forward(self, rna_input, pep_input):
    if rna_input.dim() == 1:
      rna_input = F.one_hot(rna_input.long(), num_classes=4).float()
    if pep_input.dim() == 1:
      pep_input = F.one_hot(pep_input.long(), num_classes=20).float()

    if rna_input.dim() == 2:
      rna_input = rna_input.unsqueeze(0)
    if pep_input.dim() == 2:
      pep_input = pep_input.unsqueeze(0)

    # Permute to [batch, channels, seq_len]
    rna = rna_input.permute(0, 2, 1)
    pep = pep_input.permute(0, 2, 1)

    if rna.size(2) < 3:
        rna = F.pad(rna, (0, 3 - rna.size(2)))
    if pep.size(2) < 3:
        pep = F.pad(pep, (0, 3 - pep.size(2)))

    rna = F.relu(self.rna_conv1(rna))
    rna = F.relu(self.rna_conv2(rna))
    rna = F.relu(self.rna_conv3(rna))
    rna = self.rna_pool(rna).squeeze(2)

    pep = F.relu(self.pep_conv1(pep))
    pep = F.relu(self.pep_conv2(pep))
    pep = F.relu(self.pep_conv3(pep))
    pep = self.pep_pool(pep).squeeze(2)

    combined = torch.cat((rna, pep), dim=1)
    x = F.relu(self.fc1(combined))
    out = self.fc2(x)
    return out