import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np






def save_plot(seq_len, exp_id, epochs, MSE_vals):

    sns_plot = data = None
    fig = None
    file_name = 'plots/' + 'exp_' + str(exp_id)

    # for i in range(epochs):
    #     print(MSE_vals[i])
    #     df = convert_to_dataframe(seq_len, MSE_vals[i])
    #     sns.palplot(sns.color_palette("Blues_d", epochs)
    #     sns_plot = sns.lineplot(x='Months', y='MSE', data=df)
    #
    # fig = sns_plot.get_figure()
    # fig.savefig(file_name + '.png')





def convert_to_dataframe(seq_len, MSE_vals):
    months = np.arange(seq_len)
    MSE_vals = np.asarray(MSE_vals)

    data = pd.DataFrame({'MSE':MSE_vals, 'Months':months })

    return data
