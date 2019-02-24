import seaborn as sns; sns.set()
import matplotlib.pyplot as plt






def save_plot(seq_len, exp_id, epoch, mean_s, std_s):
#
#     x_axes = [i for i in range(dev_size)]
#     if epoch == 0 or epoch == 10 or epoch == 19:
#         mae = np.array(mae)
#         std = np.array(std)
#         std_upper = mae + std
#         std_lower = mae - std
#         plt.plot(seq_len, std_upper,'b',linestyle=':',alpha=0.3)
#         plt.plot(seq_len, std_lower,'b',linestyle=':',alpha=0.3)
#         plt.plot(seq_len, mae, 'r',linestyle=':', alpha=0.5) # plotting t, a separately
#         plt.fill_between(seq_len, std_upper, std_lower, alpha=0.1)
#         plt.xlabel('Months')
#         plt.ylabel('μ (red), [μ - std, μ + std] (blue)')

        file_name = 'plots/' + 'exp_' + str(exp_id) + '/ep_' + str(epoch)
        plt.savefig(file_name + '.png')
