# Initialize model, loss, optimizer
model = CNN()
total_positives=len(positive_labeled)
total_negatives=len(fasta_negative_pairs) +len(negative_labeled)
pos_weight_value = torch.tensor([total_negatives / total_positives])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for epoch in range(30):
  model.train()
  train_losses = []

  for batch in train_loader:
    if batch is None:
      continue
    print(f"Batch size: RNA={rna_batch.shape}, PEP={pep_batch.shape}, Labels={label_batch.shape}")

    rna_batch, pep_batch, label_batch = batch
    rna_batch = rna_batch.to(device)
    pep_batch = pep_batch.to(device)
    label_batch = label_batch.float().view(-1,1).to(device)

    scores = model(rna_batch, pep_batch)
    loss = criterion(scores, label_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

  if len(train_losses) > 0:
    avg_loss = sum(train_losses) / len(train_losses)
    print(f"Epoch [{epoch + 1}/30], Loss: {avg_loss:.4f}")

  # Validation
  model.eval()
  val_losses = []

  with torch.no_grad():
    for batch in val_loader:
      if batch is None:
        continue
      rna_batch, pep_batch, label_batch = batch
      rna_batch = rna_batch.to(device)
      pep_batch = pep_batch.to(device)
      label_batch = label_batch.float().view(-1,1).to(device)

      val_scores = model(rna_batch, pep_batch)
      val_loss = criterion(val_scores, label_batch)
      val_losses.append(val_loss.item())
    if len(val_losses) > 0:
      avg_val_loss = sum(val_losses) / len(val_losses)
      print(f"Epoch [{epoch + 1}/30], Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "/content/drive/MyDrive/Deep Learning/models/cnn.pt")