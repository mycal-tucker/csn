import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import gridspec
from sklearn.decomposition import PCA


# Save the models and images of reconstructions, predictions, and prototypes.
def plot_single_img(img, ax=None, savepath=None):
    side_length = int(np.sqrt(img.shape[1]))
    assert side_length * side_length == img.shape[1]  # Make sure didn't truncate anything.
    new_base_fig = ax is None
    if new_base_fig:
        fig, ax = plt.subplots()
    figure = img.reshape(side_length, side_length)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(figure, cmap='Greys_r')
    if savepath is not None:
        plt.savefig(savepath)
        return
    if new_base_fig:
        plt.show()


def plot_rows_of_images(images, savepath, show=True):
    num_types_of_imgs = len(images)
    fig = plt.figure(figsize=(images[0].shape[0], num_types_of_imgs))
    gs = gridspec.GridSpec(num_types_of_imgs, images[0].shape[0])
    for i, type_of_img in enumerate(images):
        for j in range(type_of_img.shape[0]):
            new_ax = plt.subplot(gs[i, j])
            plot_single_img(np.reshape(type_of_img[j], (1, -1)), ax=new_ax)
    plt.savefig(savepath)
    if show:
        plt.show()
    plt.close('all')


def plot_multiple_runs(x_data, y_data, y_stdev, labels, x_axis, y_axis, window_size=500, top=11, bottom=0):
    assert len(x_data) == len(y_data) == len(labels)
    if y_stdev is None:
        y_stdev = [[0 for _ in y_data[i]] for i in range(len(y_data))]
    # Smooth stuff out
    plot_y = []
    plot_y_err = []
    plot_x = []
    for i in range(len(x_data)):
        for j in range(len(x_data[i]) - window_size):
            val = np.mean(y_data[i][j: j + window_size])
            y_data[i][j] = val
        # There's a tiny bit of trickiness here to include the last element in the window.
        plot_y.append(y_data[i][:-window_size])
        plot_y[-1].append(y_data[i][-window_size])
        plot_y_err.append(y_stdev[i][:-window_size])
        plot_y_err[-1].append(y_stdev[i][-window_size])
        plot_x.append(x_data[i][:-window_size])
        plot_x[-1].append(x_data[i][-window_size])
    fig, ax = plt.subplots()
    for run_idx in range(len(x_data)):
        markers, caps, bars = ax.errorbar(plot_x[run_idx], plot_y[run_idx], yerr=plot_y_err[run_idx], label=labels[run_idx])
        [bar.set_alpha(0.01) for bar in bars]  # Tune this alpha param depending on how opaque you want the error bars.
    plt.legend([str(label) for label in labels])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.ylim(top=top, bottom=bottom)
    plt.show()


def plot_bar_chart(x_data, y_data, y_stdev, labels, x_axis, y_axis, top=1, bottom=0):
    assert len(x_data) == len(y_data) == len(labels)
    if y_stdev is None:
        y_stdev = [0 for _ in y_data]
    fig, ax = plt.subplots()
    width_increment = 1.0 / (len(x_data) + 1)
    for run_idx in range(len(x_data)):
        ax.bar(np.asarray(x_data[run_idx]) + width_increment * (run_idx - (len(x_data) - 1) / 2),
               y_data[run_idx],
               yerr=y_stdev[run_idx],
               width=width_increment)
    ax.legend([str(label) for label in labels])
    ax.set_xticks(x_data[0])
    if len(x_data[0]) == 3:
        # Train, Prune, Tune case
        ax.set_xticklabels(['Trained', 'Pruned', 'Tuned'])
    elif len(x_data[0]) == 10:  # Sigma case
        ax.set_xticklabels(['s' + str(i) for i in range(10)])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    plt.ylim(top=top, bottom=bottom)
    plt.show()


def plot_encodings(encodings, coloring_labels=None, num_to_plot=500, ax=None, coloring_name='digit'):
    enc = encodings[-num_to_plot:]
    plot_in_color = coloring_labels is not None
    array_version = np.asarray(enc)
    pca = None
    if array_version.shape[1] > 2:
        pca = PCA(n_components=2)
        pca.fit(array_version)
        transformed = pca.transform(array_version)
    else:
        transformed = array_version
    x = transformed[:, 0]
    y = transformed[:, 1]
    new_base_fig = ax is None
    if new_base_fig:
        fig, ax = plt.subplots()
    if plot_in_color:
        colors = coloring_labels[-num_to_plot:]
        num_labels = np.max(colors) - np.min(colors)
        color_map_name = 'coolwarm' if num_labels == 1 else 'RdBu'
        cmap = plt.get_cmap(color_map_name, num_labels + 1)
        pcm = ax.scatter(x, y, s=20, marker='o', c=colors, cmap=cmap, vmin=np.min(colors) - 0.5, vmax=np.max(colors) + 0.5)
        if new_base_fig:
            min_tick = 0
            max_tick = 10 if np.max(colors) > 2 else 2
            fig.colorbar(pcm, ax=ax, ticks=np.arange(min_tick, max_tick))
    else:
        pcm = ax.scatter(x, y, s=20, marker='o', c='gray')
    if new_base_fig:
        ax.set_title('Encodings colored by ' + coloring_name)
        plt.show()
    return pca, pcm


def plot_latent_prompt(encoding, labels=None, test_encodings=None, block=True, ax=None, classes=None, savepath=None, show=True):
    params = {'ytick.labelsize': 16,
              'xtick.labelsize': 16}
    plt.rcParams.update(params)
    if ax is None:
        fig, (ax) = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("Weather Encodings", fontsize=20)
        # ax.set_title("Adult Encodings", fontsize=20)
    if encoding is not None:
        pca, pcm = plot_encodings(encoding, labels, ax=ax, coloring_name='class')
    if test_encodings is not None:
        for i, enc in enumerate(test_encodings):
            marker = 'x' if i < 2 else 'o'
            if i == 0:
                color = 'blue'
            elif i == 1:
                color = 'red'
            else:
                color = 'black'
            plot_gray_encoding(np.reshape(enc, (1, -1)), ax, pca, marker=marker, color=color)
    if labels is not None:
        if classes is None:
            classes = [i for i in range(10)]  # Assume digits
        cbar = fig.colorbar(pcm, ticks=np.arange(0, len(classes)), ax=ax)  # Assumes that ax is None and have new figure.
        cbar.ax.set_yticklabels(classes)
    if block:
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            # plt.xlim([-0.5, 0.5])
            # plt.ylim([-0.5, 0.5])
            plt.show()
    else:
        plt.draw()
        plt.pause(0.001)
        if savepath is not None:
            plt.savefig(savepath)


def plot_gray_encoding(encoding, ax, pca, marker='x', color=None):
    array_version = np.asarray(encoding)
    if pca is None:
        transformed = array_version
    else:
        transformed = pca.transform(array_version)
    x = transformed[:, 0]
    y = transformed[:, 1]
    new_base_fig = ax is None
    if new_base_fig:
        fig, ax = plt.subplots()
    size = 400 if marker == 'x' else 200
    ax.scatter(x, y, s=size, linewidths=5, marker=marker, c=color)
    plt.ylabel("Z 1", fontsize=15)
    plt.xlabel("Z 0", fontsize=15)


def plot_mst(tree):
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')
    nx.draw(tree, pos, with_labels=True, arrows=True)
    labels = nx.get_edge_attributes(tree, 'weight')
    nx.draw_networkx_edge_labels(tree, pos, edge_labels=labels)
    plt.show()



# Given a list of trajectories, plots them in 3D space.
def visualize_trajectories(trajectories, coloring=None, targets=None, title=''):
    # Pull out the number of dimensions from the trajectories.
    viz_3d(trajectories, coloring=coloring, targets=targets, title=title)


def viz_3d(trajectories, coloring=None, targets=None, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, trajectory in enumerate(trajectories):
        xs = trajectory[0, :]
        ys = trajectory[1, :]
        zs = trajectory[2, :]
        ax.plot(xs, ys, zs)
        if coloring is None or i >= coloring.shape[0]:
            continue
        colors = [c_int for c_int in coloring[i, :]]
        for j, subx in enumerate(xs):
            if j % 20 == 0:
                ax.text(subx, ys[j], zs[j], str(colors[j]), 'z')
    # if targets is not None:
    #     ax.scatter(targets[0, :], targets[1, :], targets[2, :], marker='^')
    #     for i in range(7):
    #         ax.text(targets[0, i], targets[1, i], targets[2, i], str(i), 'z')
    plt.title(title)
    plt.grid(True)
    ax.set_xlabel('X Direction')
    ax.set_ylabel('Y Direction')
    ax.set_zlabel('Z Direction')
    plt.show()
