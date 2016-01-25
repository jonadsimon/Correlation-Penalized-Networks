import matplotlib.pyplot as plt
import pandas as pd
import os

input_dir = os.path.join(os.path.split(__file__)[0], '..', '..', 'output', 'MLP')

cor0_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000000_Drop1.000000.csv'
cor1e5_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000010_Drop1.000000.csv'
cor5e5_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000050_Drop1.000000.csv'
cor1e4_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000100_Drop1.000000.csv'
cor5e4_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000500_Drop1.000000.csv'

cor0_df = pd.read_csv(input_dir+'/'+cor0_file)
cor1e5_df = pd.read_csv(input_dir+'/'+cor1e5_file)
cor5e5_df = pd.read_csv(input_dir+'/'+cor5e5_file)
cor1e4_df = pd.read_csv(input_dir+'/'+cor1e4_file)
cor5e4_df = pd.read_csv(input_dir+'/'+cor5e4_file)

plt.plot(cor0_df["Epoch"], cor0_df["Error"], linewidth=1.5, label="cor_reg=0.0")
plt.plot(cor1e5_df["Epoch"], cor1e5_df["Error"], linewidth=1.5, label="cor_reg=1e-5")
plt.plot(cor5e5_df["Epoch"], cor5e5_df["Error"], linewidth=1.5, label="cor_reg=5e-5")
plt.plot(cor1e4_df["Epoch"], cor1e4_df["Error"], linewidth=1.5, label="cor_reg=1e-4")
plt.plot(cor5e4_df["Epoch"], cor5e4_df["Error"], linewidth=1.5, label="cor_reg=5e-4")

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Validation Error with Activation Correlation Penalties")
plt.legend()

outfile_path = os.path.join(os.path.split(__file__)[0], 'Validation_Loss_Comparison_100epochs.png')

plt.savefig(outfile_path)

plt.show()
