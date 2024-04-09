import copy

import torchvision.transforms

from ooddetectors import open_and_process
from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import ResNetVAE
from classifier.resnetclassifier import ResNetClassifier
import yaml
from domain_datasets import build_nico_dataset
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import matplotlib
from scipy.special import kl_div
import os
from ooddetectors import RabanserSD
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from bias_samplers import *
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mape
import numpy as np
import seaborn as sns
import math
# pd.set_option('display.precision', 2)
pd.options.display.float_format = '{:.2f}'.format



def aggregate_semantic_data():
    table_data = []

    for dataset in ["CIFAR10", "CIFAR100", "EMNIST", "MNIST"]:
        for sample_size in [30, 50, 100, 200, 500]:
            for dsd_type in ["ks", "typicality", "grad_magnitude", "odin", "cross_entropy"]:
                for k in ["", "_5NN"]:
                    dsd = f"{dsd_type}{k}"
                    fname = f"data/Semantic_{dataset}_{dsd}_{sample_size}.csv"
                    data = open_and_process_semantic(fname, filter_noise=True)
                    if data is None:
                        continue
                    data_cov = open_and_process(f"data/{dataset}_normal_{dsd}_{sample_size}.csv",
                                                filter_noise=True)
                    # print(data_cov)
                    # input(f"data/{dataset}_normal_{dsd}_{sample_size}.csv")
                    if data_cov is not None:
                        data.loc[data['fold'] == 'ind', 'pvalue'] = data_cov.loc[
                            data_cov['fold'] == 'ind', 'pvalue']

                    data["Dataset"] = dataset
                    data["OOD Detector"] = dsd
                    data["Sample Size"] = sample_size
                    table_data.append(data)
    df = pd.concat(table_data)
    return df

def aggregate_covariate_data(mode="normal"):
    table_data = []

    for dataset in ["CIFAR10", "CIFAR100", "imagenette", "NICO", "Njord", "Polyp"]:
        for sample_size in [30, 50, 100, 200, 500]:
            for dsd_type in ["ks", "typicality_ks_glow", "grad_magnitude", "odin", "cross_entropy"]:
                for k in ["", "_5NN"]:
                    dsd = f"{dsd_type}{k}"
                    fname = f"data/{dataset}_{mode}_{dsd}_{sample_size}.csv"
                    data = open_and_process(fname, filter_noise=False)
                    if data is None:
                        continue
                    data["Dataset"] = dataset
                    data["OOD Detector"] = dsd
                    data["Sample Size"] = sample_size
                    data["Type"] = data["OOD Detector"].apply(lambda x: "kNNDSD" if "5NN" in x else "Vanilla")
                    data["Base"] = data["OOD Detector"].apply(lambda x: x.split("_5NN")[0] if "5NN" in x else x)
                    table_data.append(data)
    df = pd.concat(table_data)
    return df
def get_threshold(data):
    ood = data[data["oodness"]>1]
    ind = data[data["oodness"]<=1]
    random_sampler_ind_data = ind[(ind["sampler"] == "RandomSampler")]
    if "RandomSampler" not in data["sampler"].unique():
        random_sampler_ind_data = ind[(ind["sampler"] == "ClusterSamplerWithSeverity_1.0")]
    threshold = random_sampler_ind_data["pvalue"].min()
    return threshold

def fpr(data, threshold):
    """
    :param ood_ps
    :param ind_ps
    Find p-value threshold that results in 95% TPR. Then find FPR.
    If threshold is given, use that instead.
    :return:
    """
    ood_ps = data[data["oodness"]>1]["pvalue"]
    ind_ps = data[data["oodness"]<=1]["pvalue"]
    thresholded = ind_ps<threshold
    return thresholded.mean()

def fnr(data, threshold):
    """
    :param ood_ps
    :param ind_ps
    Find p-value threshold that results in 95% TPR. Then find FPR.
    If threshold is given, use that instead.
    :return:
    """
    ood_ps = data[data["oodness"]>1]["pvalue"]

    ind_ps = data[data["oodness"]<=1]["pvalue"]
    thresholded = ood_ps>=threshold
    return thresholded.mean()
def balanced_accuracy(data, threshold):
    ood_ps = data[data["oodness"]>1]["pvalue"]
    ind_ps = data[data["oodness"]<=1]["pvalue"]
    sorted_ps = sorted(ind_ps)
    # ba = ((ind_ps>=threshold).mean()+(ood_ps<threshold).mean()) /2
    ba = 1-fpr(data, threshold) + 1-fnr(data, threshold)
    ba = ba/2
        # print(f"{(ind_ps >= threshold).mean()}+ {(ood_ps < threshold).mean()} / 2")
    return ba

def auroc(data):
    ood_ps = data[data["oodness"]>1]["pvalue"]
    ind_ps = data[data["oodness"]<=1]["pvalue"]
    true = [0]*len(ood_ps)+[1]*len(ind_ps)
    probs = list(ood_ps)+list(ind_ps)
    auc = roc_auc_score(true, probs)
    return auc

def aupr(data):
    ood_ps = data[data["oodness"]>1]["pvalue"]

    ind_ps = data[data["oodness"]<=1]["pvalue"]
    true = [0] * len(ood_ps) + [1] * len(ind_ps)
    probs = list(ood_ps) + list(ind_ps)
    auc = average_precision_score(true, probs)
    return auc



def plot_regplots():
    merged = aggregate_covariate_data()
    merged.replace(["RandomSampler", "ClassOrderSampler", "SequentialSampler", "ClusterSampler"],
                             ["None", "Class", "Temporal", "Synthetic"], inplace=True)
    palette = sns.color_palette("muted", n_colors=10)
    colors = dict(zip(["None", "Class", "Temporal", "Synthetic"], [palette[7], palette[0], palette[2], palette[8]]))

    merged = merged[merged["Dataset"]=="CIFAR10"]
    merged = merged[merged["Sample Size"]==100]
    merged = merged[merged["Base"]=="ks"]
    # print(merged[(merged['fold'] == 'ind')]['pvalue'])
    merged.rename(columns={"sampler":"Bias"}, inplace=True)

    min_value = merged[(merged['fold'] == 'ind') & (merged['Bias'] == 'None')]['pvalue'].min()
    # print(merged.columns)
    g = sns.FacetGrid(data=merged, col="Type", row="Bias", sharey=False, sharex=False, margin_titles=True, height=2.5, aspect=2)
    g.map_dataframe(sns.scatterplot, y="pvalue", x="loss", hue="fold", palette="mako")
    # g.map_dataframe(sns.kdeplot, x="pvalue", hue="fold", palette="mako")
    # print(merged[merged['fold'] == 'ind'])
    print(min_value)
    def draw_min_line(*args, **kwargs):
        plt.axhline(y=min_value, color='red', linestyle='--', lw=2)

    g.map(draw_min_line)
    g.set(yscale="log").set(ylim=0)

    plt.savefig("figures/regplots.eps")
    plt.show()

def plot_variances():
    merged = aggregate_covariate_data()
    merged.replace(["RandomSampler", "ClassOrderSampler", "SequentialSampler", "ClusterSampler"],
                             ["None", "Class", "Temporal", "Synthetic"], inplace=True)
    palette = sns.color_palette("muted", n_colors=10)
    colors = dict(zip(["None", "Class", "Synthetic"], [palette[0], palette[1], palette[2]]))

    merged = merged[merged["Dataset"]=="CIFAR10"]
    merged = merged[merged["Sample Size"]==100]
    merged = merged[merged["Base"]=="ks"]
    merged = merged[merged["fold"]=="ind"]

    merged["pvalue"] = merged["pvalue"].apply(lambda x: math.log(x, 10) if x!=0 else -255)
    # merged.rename(columns={"Sampler":"Bias"}, inplace=True)
    merged.rename(columns={"pvalue":"Log p-value"}, inplace=True)

    g = sns.FacetGrid(data=merged, row="Type", hue="sampler", sharey=False, sharex=False, margin_titles=True, height=2.5, aspect=2, palette=colors)
    g.map_dataframe(sns.kdeplot, x="Log p-value", palette=colors, common_norm=False)

    # After plotting, explicitly add a legend. This step may need adjustments based on your specific needs.
    # Generate the legend manually
    handles = [plt.Line2D([], [], color=colors[label], label=label) for label in colors]
    plt.legend(handles=handles, title='Bias')

    plt.savefig("figures/knn_kdes.eps")
    plt.show()
    print("plotted")

def illustrate_clustersampler():
    fig, ax = plt.subplots(5,1, sharex=False, sharey=False)
    for i, severity in enumerate([0, 0.1, 0.25, 0.5, 1]):
        dataset_classes = np.array(sum([[i] * 10 for i in range(10)], []))  # sorted
        shuffle_indeces = np.random.choice(np.arange(len(dataset_classes)), size=int(len(dataset_classes) * severity),
                                           replace=False)
        to_shuffle = dataset_classes[shuffle_indeces]
        np.random.shuffle(to_shuffle)
        dataset_classes[shuffle_indeces] = to_shuffle
        ax[i].imshow(dataset_classes.reshape((1,len(dataset_classes))).repeat(16,0), cmap="viridis")
        # ax[i].axis("off")
        ax[i].set_ylabel(f"{1-severity}       ", rotation=0)
        ax[i].xaxis.set_visible(False)
        # make spines (the box) invisible
        plt.setp(ax[i].spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax[i].tick_params(left=False, labelleft=False)
        # remove background patch (only needed for non-white background)
        ax[i].patch.set_visible(False)
    fig.text(0.5, 0.05, 'Index Order', ha='center')
    fig.text(0.1, 0.5, 'Bias Severity', va='center', rotation='vertical')

    plt.savefig("bias_severity.eps")
    plt.show()


def breakdown_by_sample_size(placeholder=False, metric="DR"):
    df = get_classification_metrics_for_all_experiments(placeholder=placeholder)
    df["Type"] = df["OOD Detector"].apply(lambda x: "kNNDSD" if "5NN" in x else "Vanilla")
    df["Base"] = df["OOD Detector"].apply(lambda x: x.split("_5NN")[0] if "5NN" in x else x)
    df["Dataset"] = df["Dataset"].apply(lambda x: x.split("_")[0].capitalize())
    df.replace(["typicality_ks_glow", "grad_magnitude", "cross_entropy", "ks", "odin"], ["Typicality", "GradNorm", "CrossEntropy", "KS", "ODIN"], inplace=True)
    g = sns.FacetGrid(data=df, col="Dataset", row="Base", sharey=True, sharex=False, margin_titles=True)
    g.map_dataframe(sns.lineplot, x="Sample Size", y=metric, hue="Type")

    # Adjust the legend
    g.add_legend()
    # g.add_legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
    # plt.subplots_adjust(bottom=0.5)  # Increase bottom margin
    # plt.tight_layout(pad=10.0)  # Adjust the layout to make room for the legend
    plt.savefig("figures/samplesizebreakdown.png")
    plt.show()

def breakdown_by_sampler(placeholder=False, metric="DR"):
    df = get_classification_metrics_for_all_experiments(placeholder=placeholder)
    df = df.groupby(["Dataset", "Sampler", "OOD Detector"])[metric].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


def summarize_results(placeholder=False):
    df = get_classification_metrics_for_all_experiments(placeholder=placeholder)
    df = df.groupby(["Dataset", "OOD Detector"])[["FPR", "FNR", "DR", "AUROC", "AUPR"]].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def threshold_plots(dataset_name, sample_size, filter_noise=False):
    df = get_classification_metrics_for_all_experiments()
    df = df[(df["Dataset"]==dataset_name)&(df["Sample Size"]==sample_size)]
    print(df)

def get_semantic_metrics_for_all_experiments(placeholder=False):
    table_data = []

    for dataset in ["CIFAR10", "CIFAR100", "EMNIST", "MNIST"]:
        for sample_size in [10, 30, 50, 100, 200, 500]:
            for dsd_type in ["ks", "typicality_ks_glow", "grad_magnitude", "odin", "cross_entropy", "jacobian"]:
                for k in ["", "_5NN"]:
                    dsd = f"{dsd_type}{k}"
                    fname = f"data/Semantic_{dataset}_{dsd}_{sample_size}.csv"
                    data = open_and_process_semantic(fname, filter_noise=True)
                    if data is None:
                        continue
                    data_cov = open_and_process(f"data/{dataset}_normal_{dsd}_{sample_size}.csv", filter_noise=True)
                    # print(data_cov)
                    # input(f"data/{dataset}_normal_{dsd}_{sample_size}.csv")
                    if data_cov is not None:
                        data.loc[data['fold'] == 'ind', 'pvalue'] = data_cov.loc[data_cov['fold'] == 'ind', 'pvalue']

                    threshold = get_threshold(data)
                    for sampler in pd.unique(data["sampler"]):
                        for fold in pd.unique(data["fold"]):

                            if fold=="ind":
                                continue
                            subset = data[(data["sampler"] == sampler)&((data["fold"] == fold)|(data["fold"] == "ind"))]
                            table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size, "fold": fold,
                                               "Sampler": sampler,
                                               "KNNDSD": "KNNDSD" if "5NN" in dsd else "Baseline",
                                               "FPR": fpr(subset, threshold=threshold),
                                               "FNR": fnr(subset, threshold=threshold),
                                               "DR": balanced_accuracy(subset, threshold=threshold),
                                               "AUROC": auroc(subset),
                                               "AUPR": aupr(subset),
                                               })
    df = pd.DataFrame(data=table_data)
    return df


def get_classification_metrics_for_all_experiments(placeholder=False, filter_noise=True):
    #summarize overall results;
    table_data = []
    for dataset in ["CIFAR10_normal", "CIFAR100_normal", "NICO_normal", "Njord_normal", "Polyp_normal", "imagenette_normal"]:
        for sample_size in [30, 50, 100, 200, 500]:
            for dsd_type in ["ks", "grad_magnitude", "odin", "cross_entropy", "typicality_ks_glow"]:
                for k in ["", "_5NN"]:
                    dsd = f"{dsd_type}{k}"
                    fname = f"data/{dataset}_{dsd}_{sample_size}.csv"
                    data = open_and_process(fname, filter_noise=filter_noise)
                    if data is None:
                        if placeholder:
                            table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                               "FPR": float("nan"),
                                               "FNR": float("nan"),
                                               "DR": float("nan"),
                                               "AUROC": float("nan")})
                        continue
                    threshold = get_threshold(data)
                    for sampler in pd.unique(data["sampler"]):
                        subset = data[data["sampler"] == sampler]
                        table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                           "Sampler": sampler,
                                           "FPR": fpr(subset, threshold=threshold),
                                           "FNR": fnr(subset, threshold=threshold),
                                           "DR": balanced_accuracy(subset, threshold=threshold),
                                           "AUROC": auroc(subset),
                                           "AUPR": aupr(subset),})
    df = pd.DataFrame(data=table_data)
    return df


def plot_semantic_kdes():
    df = aggregate_semantic_data()
    df = df[df["Sample Size"]==100]
    df = df[df["Dataset"]=="MNIST"]
    g = sns.FacetGrid(data=df, col="OOD Detector", sharey=False, sharex=False, margin_titles=True)
    g.map_dataframe(sns.histplot, x="pvalue", hue="fold", palette="mako", bins=50).set(xscale="log", yscale="log")
    g.add_legend()
    plt.savefig("figures/semantic_kde.png")
    plt.show()



if __name__ == '__main__':
    """
    # Classification
    """

    from features import *
    from testbeds import *
    # plot_variances()
    # summarize_results(placeholder=False)


    #sampler_breakdown
    # breakdown_by_sampler()
    # input()
    #
    # sample_size_breakdown
    # breakdown_by_sample_size()

    # thresholding_plots
    plot_regplots()

