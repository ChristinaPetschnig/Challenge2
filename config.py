esc50_path = 'data/esc50'
runs_path = 'results'

# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# ratio to split off from training data
val_size = .2

model_constructor = "ResNet(block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=config.n_classes)"
# model_constructor = "ResNet(block=ResidualBlock, layers=[3, 4, 6, 3], num_classes=config.n_classes)"

# model checkpoints loaded for testing
# test_checkpoints = ['terminal.pt']
test_checkpoints = ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
#test_experiment = 'results/2024-04-01-00-00'
#test_experiment = 'results/sample-run'
test_experiment = 'results/2024-05-26-20-52'

# sampling rate for waves
sr = 44100

device_id = 0
batch_size = 32
num_workers = 6#16
persistent_workers = True

epochs = 200
patience = 20

# Änderung: Feinjustierung der Lernrate und des Weight Decay

lr = 5.5e-3
weight_decay = 3.5e-3
warm_epochs = 10
gamma = 0.9
step_size = 5









