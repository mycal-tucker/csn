import matplotlib.pyplot as plt
params = {'ytick.labelsize': 16}
plt.rcParams.update(params)

def plot_bar_chart(x_labels, means, vars, y_label, rand, y_max, title, filename):
    x_label_pos = [i for i in range(len(x_labels))]
    plt.bar(x_label_pos, means, yerr=vars)
    plt.ylabel(y_label, fontsize=15)
    plt.xticks(x_label_pos, x_labels, fontsize=15)
    plt.xlim(-0.5, 2.5)
    plt.ylim(50, y_max)
    plt.title(title, fontsize=20)
    plt.hlines(rand, -5, 5, colors='red')
    plt.savefig(filename)
    plt.show()


# GERMAN
labels = ['VFAE', 'Adv.', 'CSN']
german_s_mean = [81 for _ in range(3)]
german_s_std = [0 for _ in range(3)]
german_s_rand = 81
german_y_mean = [72.7, 74.4, 73]
german_y_std = [0, 0, 2]
german_y_rand = 71

plot_bar_chart(labels, german_s_mean, german_s_std, y_label='S Accuracy %', rand=german_s_rand, y_max=90,
               title='German Age Prediction', filename='s_acc_german.png')
plot_bar_chart(labels, german_y_mean, german_y_std, y_label='Y Accuracy %', rand=german_y_rand, y_max=90,
               title='German Credit Prediction', filename='y_acc_german.png')


# ADULT
labels = ['VFAE', 'Adv.', 'CSN']
label_pos = [i for i in range(3)]
adult_s_mean = [67 for _ in range(3)]
adult_s_std = [0 for _ in range(3)]
adult_s_rand = 67
adult_y_mean = [81.3, 84.4, 85.0]
adult_y_std = [0, 0, 1.2]
adult_y_rand = 75

plot_bar_chart(labels, adult_s_mean, adult_s_std, y_label='S Accuracy %', rand=adult_s_rand, y_max=90,
               title='Adult Male/Female Prediction', filename='s_acc_adult.png')
plot_bar_chart(labels, adult_y_mean, adult_y_std, y_label='Y Accuracy %', rand=adult_y_rand, y_max=90,
               title='Adult Income Prediction', filename='y_acc_adult.png')
