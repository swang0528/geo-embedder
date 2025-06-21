# train.py
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import the scheduler
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# --- Local Imports ---
from dataset_gen import PolygonDataset
from model import CGEM, ComplexLoss


# +----------------------------------+
# |      NEW VISUALIZATION CLASS     |
# +----------------------------------+
class LivePlot:
    """A simple class to handle real-time plotting of training statistics."""

    def __init__(self, title='Training Progress'):
        print("Initializing real-time plot...")
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)

        self.epochs = []
        self.train_losses = []
        self.val_losses = []

        # Create empty lines for train and validation loss
        self.train_line, = self.ax.plot(self.epochs, self.train_losses, 'b-', label='Train Loss')
        self.val_line, = self.ax.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        self.ax.legend()
        print("Plot initialized. Training will begin shortly.")

    def update(self, epoch, train_loss, val_loss):
        """Appends new data and updates the plot."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Update the data of the lines
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)

        # Rescale the axes
        self.ax.relim()
        self.ax.autoscale_view()

        # Force a redraw of the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow GUI to update

    def close(self):
        """Turn off interactive mode and show final plot."""
        print("Training complete. Displaying final plot. Close the plot window to exit.")
        plt.ioff()
        plt.show()


def main(args):
    print(f"Using device: {args.device}")

    # --- Initialize Plotter ---
    plotter = None
    if not args.no_plot:
        plotter = LivePlot()

    # --- Smoke Test Override ---
    if args.smoke_test:
        print("--- RUNNING IN SMOKE TEST MODE ---")
        args.num_train_pairs = 128
        args.num_val_pairs = 64
        args.batch_size = 8
        args.epochs = 5  # Run a few epochs for the plot to be meaningful
        args.train_cache = "smoke_test_train.pkl"
        args.val_cache = "smoke_test_val.pkl"
        force_regen = True
    else:
        force_regen = args.force_regen

    # 1. Datasets and DataLoaders
    train_dataset = PolygonDataset(num_pairs=args.num_train_pairs, cache_path=args.train_cache,
                                   force_regenerate=force_regen)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    val_dataset = PolygonDataset(num_pairs=args.num_val_pairs, cache_path=args.val_cache, force_regenerate=force_regen)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # 2. Model, Loss, Optimizer
    model = CGEM().to(args.device)
    loss_fn = ComplexLoss(lambda_sim=args.lambda_sim)
    # --- UPDATED: Use AdamW optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # --- NEW: Add learning rate scheduler ---
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')

    # 3. Training Loop
    print("--- Starting Training ---")
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss_accumulator = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)
        for batch in progress_bar:
            g1, g2, rot_target, sim_target = batch['g1'].to(args.device), batch['g2'].to(args.device), batch[
                'rot_target'].to(args.device), batch['sim_target'].to(args.device)

            optimizer.zero_grad()
            u1, u2 = model(g1), model(g2)
            loss = loss_fn(u1, u2, rot_target, sim_target)
            loss.backward()

            # --- NEW: Add gradient clipping ---
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            train_loss_accumulator += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_train_loss = train_loss_accumulator / len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_loss_accumulator = 0
        with torch.no_grad():
            for batch in val_loader:
                g1, g2, rot_target, sim_target = batch['g1'].to(args.device), batch['g2'].to(args.device), batch[
                    'rot_target'].to(args.device), batch['sim_target'].to(args.device)
                u1, u2 = model(g1), model(g2)
                loss = loss_fn(u1, u2, rot_target, sim_target)
                val_loss_accumulator += loss.item()

        avg_val_loss = val_loss_accumulator / len(val_loader)

        # --- NEW: Step the scheduler after each epoch ---
        scheduler.step()

        # --- Update Plot ---
        if plotter:
            plotter.update(epoch + 1, avg_train_loss, avg_val_loss)

        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 5. Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # TODO: need more robust dir naming
            save_model_dir = rf"C:\Users\Siyang_Wang_work\Documents\A_IndependentResearch\GenAI\LayoutML\Geo-Embedder\dataset\{args.model_id}"
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
            torch.save(model.state_dict(), os.path.join(save_model_dir, 'cgem_best_model.pth'))
            tqdm.write(f"    -> New best model saved with validation loss: {best_val_loss:.4f}")

    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f} seconds ---")
    if plotter:
        plotter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Complex Geometric Embedding Model (C-GEM)")
    # Paths & Data
    parser.add_argument('--train-cache', type=str, default="data/train_dataset.pkl",
                        help='Path to training dataset cache')
    parser.add_argument('--val-cache', type=str, default="data/val_dataset.pkl",
                        help='Path to validation dataset cache')
    parser.add_argument('--force-regen', action='store_true', help='Force dataset regeneration')
    parser.add_argument('--num-train-pairs', type=int, default=50000, help='Number of pairs for the training set')
    parser.add_argument('--num-val-pairs', type=int, default=5000, help='Number of pairs for the validation set')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda-sim', type=float, default=1.0, help='Weight for the similarity loss component')
    # --- NEW Hyperparameters ---
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for AdamW optimizer')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')

    # System & Visualization
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--no-plot', action='store_true', help='Disable real-time plotting')

    # Testing
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test with minimal data')
    parser.add_argument('--model-id', type=str, default="cgem_test", )

    args = parser.parse_args()
    main(args)
