import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def ioplot(
    A,
    coords,
    fill=True,
    samples=None,
    kde=True,
    x=0,
    y=1,
    n_levels=10,
    pre_cmap="Purples",
    post_cmap="Greens",
    alpha=1,
    fig=None,
    figsize=(5, 5),
    s=4,
):
    samples = samples or len(coords)
    pre_coords = np.vstack(
        [coords[np.nonzero(A[:, i])] - coords[i] for i in range(samples)]
    )
    post_coords = np.vstack(
        [coords[np.nonzero(A[i, :])] - coords[i] for i in range(samples)]
    )

    fig = fig or plt.figure(figsize=figsize)
    if kde:
        ax0 = sns.kdeplot(
            x=pre_coords[:, x],
            y=pre_coords[:, y],
            n_levels=n_levels,
            cmap=pre_cmap,
            fill=fill,
            # shade_lowest=False,
            label="pre",
            zorder=11,
        )
        # if shade:
        # [h.set_alpha(a) for h,a, in zip(ax0.collections,np.linspace(0,1,len(ax0.collections)))]
        ax1 = sns.kdeplot(
            x=post_coords[:, x],
            y=post_coords[:, y],
            n_levels=n_levels,
            ax=ax0,
            cmap=post_cmap,
            fill=fill,
            # shade_lowest=True,
            label="post",
            zorder=10,
        )
        # if fill:
        #     # plt.gca().set_facecolor(
        #     #     ax0.collections[n_levels].get_facecolor()[0]
        #     # )  # fix seaborn's kdeplot bug
        #     # ax0.collections[n_levels].set_alpha(0)
        #     [
        #         h.set_alpha(a)
        #         for h, a, in zip(
        #             ax1.collections,
        #             np.sqrt(
        #                 np.hstack(
        #                     [
        #                         np.linspace(0, 1, n_levels),
        #                         np.linspace(0, 1, n_levels)[1:],
        #                     ]
        #                 )
        #             ),
        #         )
        #     ]  # because ax0==ax1, a BUG?

        # plt.axis('equal')
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("$\\Delta$" + "xyz"[x])
        plt.ylabel("$\\Delta$" + "xyz"[y])
    else:
        ax = fig.add_subplot(projection="3d")
        # ax.view_init(elev=90, azim=-90)  # x,y
        # ax.view_init(elev=0, azim=90)    # x,z
        ax.view_init(elev=10, azim=45)  # x,y,z

        # ax.set_aspect('equal')

        ax.scatter(
            pre_coords[:, 0],
            pre_coords[:, 1],
            pre_coords[:, 2],
            marker="D",
            s=s,
            c=[sns.color_palette(pre_cmap, 3)[-1]],
        )
        ax.scatter(
            post_coords[:, 0],
            post_coords[:, 1],
            post_coords[:, 2],
            s=s,
            c=[sns.color_palette(post_cmap, 3)[-1]],
        )
        ax.set_xlabel("$\\Delta$x")
        ax.set_ylabel("$\\Delta$y")
        ax.set_zlabel("$\\Delta$z")
        # ax.set_yticklabels([])

    plt.title("%d samples (out of %s nodes)" % (samples, len(coords)))


if __name__ == "__main__":
    import itertools

    coords = np.array(list(itertools.product(range(3), repeat=3)))

    ii = coords[:, None, :]
    jj = coords[None, :, :]
    A = (
        (ii[:, :, 2] > jj[:, :, 2]) & (((ii - jj)[:, :, :2] ** 2).sum(axis=2) <= 1)
    ).astype(int)

    ioplot(A, coords, kde=False)
    ioplot(A, coords, x=0, y=2)

    # (ii + jj * 0)[:5, :5, 2] + (ii * 0 + jj)[:5, :5, 2] / 10

    # import pandas as pd

    # pd.concat(
    #     [
    #         pd.DataFrame((ii + jj * 0).reshape(-1, 3)).add_prefix("from_"),
    #         pd.DataFrame((ii * 0 + jj).reshape(-1, 3)).add_prefix("to_"),
    #         pd.DataFrame(A.reshape(-1, 1)),
    #     ],
    #     axis=1,
    # ).tail(10)
