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

def correlation(pandas_df, plot=False, split_by_sampler=False):
    # pandas_df = pandas_df[pandas_df["sampler"]!="ClassOrderSampler"]

    # merged_p=pandas_df["pvalue"].apply(lambda x: math.log(x, 10) if x!=0 else -255)
    # merged_loss = pandas_df["loss"]
    # return pearsonr(merged_p, merged_loss)[0]

    merged_p = pandas_df["pvalue"]
    merged_loss = pandas_df["loss"]
    return spearmanr(merged_p, merged_loss)[0]


def scatterplot_wasserstein_normalized(pandas_df):

    # Normalize the data for pvalue and loss
    full_data = {}
    for fold in pandas_df["fold"].unique():
        data = {}
        by_fold = pandas_df[pandas_df["fold"]==fold]
        random_sampler_p = by_fold[(by_fold["sampler"] == "RandomSampler")]["pvalue"]
        for sampler in by_fold["sampler"].unique():
            if sampler == "RandomSampler":
                continue
            subdata_p = by_fold[((by_fold["sampler"] == sampler))]["pvalue"]
            # Compute Wasserstein distance for each normalized variable
            wasserstein_p = ks_2samp(random_sampler_p, subdata_p)[1]
            # print(f"Fold {fold}, sampler {sampler}, wasserstein distance {wasserstein_p}")
            data[sampler] = wasserstein_p
        full_data[fold] = data
    return full_data

def get_kl():
    merged = []
    for dataset in ["CIFAR10_normal", "CIFAR100_normal", "NICO_noise", "Njord_noise", "Polyp_noise", "imagenette_normal"]:
        for dsd in ["ks", "ks_5NN", "typicality"]:
            for sample_size in [100]:
                fname = f"data/{dataset}_{dsd}_{sample_size}.csv"
                data = open_and_process(fname)
                if data is None:
                    continue
                KL = scatterplot_wasserstein_normalized(data)
                for fold in data["fold"].unique():
                    for sampler in data["sampler"].unique():
                        if sampler=="RandomSampler":
                            continue
                        table_data = {}
                        table_data["Dataset"]=dataset.split("_")[0]
                        table_data["Fold"]=fold
                        table_data["OOD Detector"]=dsd
                        table_data["Sampler"]=sampler
                        table_data["p"]=KL[fold][sampler]
                        merged.append(table_data)
    df = pd.DataFrame(merged)
    # df = df[df["Sampler"]!="ClusterSampler"]
    grouped = df.groupby(["Dataset","Fold",  "OOD Detector"])["p"].mean().reset_index()
    print(grouped)
    # g = sns.FacetGrid(data=grouped, col="Dataset", margin_titles=True, col_wrap=3)
    # g.map_dataframe(sns.barplot, x="OOD Detector", y="KL", palette="mako")
    # grouped = grouped[grouped["Fold"]=="ind"]
    sns.barplot(data=grouped, x="Dataset", y="p", hue="OOD Detector", palette="mako")
    plt.yscale("log")
    # g.set(yscale="log").set(ylim=(df["KL"].min(), df["KL"].max()))
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left')
    plt.tight_layout()
    plt.savefig("figures/sample_bias_effect.eps")
    plt.show()



def linreg_smape(pandas_df_orig, random_sampler_df_orig):
    # pandas_df = pandas_df[pandas_df["sampler"]!="ClassOrderSampler"]
    try:
        pandas_df = copy.deepcopy(pandas_df_orig)
        random_sampler_df = copy.deepcopy(random_sampler_df_orig)

        random_sampler_df = random_sampler_df[random_sampler_df["Dataset"]==pandas_df["Dataset"].unique()[0]]
        random_sampler_df = random_sampler_df[random_sampler_df["OOD Detector"]==pandas_df["OOD Detector"].unique()[0]]

        ps = pandas_df["pvalue"].apply(lambda x: np.log10(x))
        pandas_df["pvalue"]=ps
        pandas_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        pandas_df.dropna(inplace=True)

        rand_ps = random_sampler_df["pvalue"].apply(lambda x: np.log10(x))
        random_sampler_df["pvalue"]=rand_ps
        random_sampler_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        random_sampler_df.dropna(inplace=True)
        rand_losses = random_sampler_df["loss"]


        ps_r = np.array(pandas_df["pvalue"]).reshape(-1, 1)
        rand_ps = np.array(random_sampler_df["pvalue"]).reshape(-1, 1)

        # model = LinearRegression()
        model = Pipeline([('poly', PolynomialFeatures(degree=5)),
                       ('linear', LinearRegression())])
        model.fit(rand_ps, rand_losses)
        preds = model.predict(ps_r)
        if pandas_df["Dataset"].unique()[0]=="NICO_noise" and pandas_df["fold"].unique()[0]=="ind":
            print("plotting")
            sns.scatterplot(data=random_sampler_df, x="pvalue", y="loss")

            sns.scatterplot(data=pandas_df, x="pvalue", y="loss", color="orange", s=10)
            plt.title(
                f"{pandas_df['Dataset'].unique()} + {pandas_df['OOD Detector'].unique()}: {pandas_df['sampler'].unique()}")
            print(preds)
            print(np.array(pandas_df["loss"]))
            sns.lineplot(x=rand_ps.flatten(), y=model.predict(rand_ps), color="blue")
            plt.show()
        return mape(preds, pandas_df["loss"])
    except:
        print("filling in...")
        return -1000


def get_loss_pdf_from_ps(ps, loss, test_ps, test_losses, bins=15):
    """
        Computes a pdf for the given number of bins, and gets the likelihood of the test loss at the given test_ps bin.
        #todo: collect a new noise dataset with the right predictor
        :returns the average likelihood of the observed test-loss as bootstrapped from the pdf w/noise.
        Higher likelihood ~ more likely that the model is correct more often.
    """
    #split ps into unevenly sized bins with equal number of entries
    pargsort = np.argsort(ps)
    sorted_ps = np.array(ps)[pargsort]
    sorted_loss = np.array(loss)[pargsort]
    p_bins = sorted_ps[::len(sorted_ps)//bins] #defines bin limits
    [min_val, max_val] = [sorted_ps[0], sorted_ps[-1]]
    p_bins = np.append(p_bins, max_val)

    #there are now 15 bins
    # print(p_bins)
    loss_samples_per_bin = [sorted_loss[i:j] for i, j in zip(range(0, len(sorted_loss), len(sorted_loss)//bins),
                                                             range(len(sorted_loss)//bins, len(sorted_loss)+len(sorted_loss)//bins, len(sorted_loss)//bins))]

    loss_pdfs = [np.histogram(losses_in_pbin, bins=len(loss_samples_per_bin[0])//10) for losses_in_pbin in loss_samples_per_bin]
    #loss_pdfs is essentially a probability funciton for each bin that shows the likelihood of some loss value given a certain p

    test_p_indexes = np.digitize(test_ps, p_bins[:-1])
    loss_diff = []
    for p, loss in zip(test_ps, test_losses):
        index = np.digitize(p, p_bins[:-1])
        predicted_loss = np.mean(loss_samples_per_bin[np.clip(index, 0, len(loss_samples_per_bin)-1)])
        loss_diff.append(np.abs(loss-predicted_loss))
    return np.mean(loss_diff)


def risk(data, threshold):

    ood = data[data["oodness"]>1]
    ind = data[data["oodness"]<=1]
    fprate =fpr(data, threshold)
    fnrate = fnr(data, threshold)
    return (fprate + fnrate)/2 * ood["loss"].mean()
    nopredcost = 0
    # data["risk"] = data["loss"]
    # data.loc[((data["pvalue"] < threshold) & (data["oodness"] >= 1)), "risk"] = nopredcost
    # data.loc[((data["pvalue"] >= threshold) & (data["oodness"] < 1)), "risk"] = nopredcost

    data["risk"]=0
    data.loc[((data["pvalue"] < threshold) & (data["oodness"] < 1)), "risk"] = data[data["oodness"]>=1]["loss"].mean() #false positive
    data.loc[((data["pvalue"] >= threshold) & (data["oodness"] >= 1)), "risk"] = data[data["oodness"]>=1]["loss"].mean() #false negative, cost is the average loss of the ood samples

    assert -1 not in pd.unique(data["risk"]) #sanity check

    return data["risk"].mean()







def correlation_summary():
    df = get_correlation_metrics_for_all_experiments()
    print(df.groupby(["Dataset", "Sampler",  "OOD Detector"])["Correlation"].mean())

def plot_regplots_organic_shifts():
    # summarize overall results;
    table_data = []
    for dataset in ["NICO_normal", "NICO_noise"]:
        for dsd in ["ks", "ks_5NN", "typicality"]:
            for sample_size in [100]:
                fname = f"data/{dataset}_{dsd}_{sample_size}.csv"
                data = open_and_process(fname)
                if data is None:
                    continue
                data["Dataset"]=dataset
                data["OOD Detector"]=dsd
                table_data.append(data)
    merged = pd.concat(table_data)
    noise = merged[merged["Dataset"]=="NICO_noise"]
    organic = merged[merged["Dataset"]=="NICO_normal"]
    random_noise = noise[noise["sampler"]=="RandomSampler"]
    smapes = organic.groupby(["Dataset", "sampler", "OOD Detector", "fold"]).apply(lambda x: linreg_smape(x, random_noise))
    print(smapes)


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
    fig, ax = plt.subplots(5,1, sharex=False, sharey=Fa)
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


def plot_bias_severity_impact(filename):
    data = pd.read_csv(filename)
    full_bias = pd.read_csv("lp_data_cifar10_noise.csv")
    full_bias = full_bias[full_bias["sampler"]=="ClusterSampler"]
    full_bias["sampler"]="ClusterSamplerWithSeverity_0.0"
    data = pd.concat((data, full_bias))

    data = data[data["sample_size"]==100]
    dataframe = []
    for ood_dataset in np.unique(data["ood_dataset"]):
        das_kn = [[], []]
        das_vn = [[], []]
        for sampler in np.unique(data["sampler"]):
            subset = data[data["sampler"]==sampler]
            ood = subset[subset["ood_dataset"] == ood_dataset]
            ind = subset[subset["ood_dataset"] == "cifar10_0.0"]
            das_kn[1].append(auroc(ood["kn_p"], ind["kn_p"]))
            das_vn[1].append(auroc(ood["vanilla_p"], ind["vanilla_p"]))
            das_vn[0].append(1-float(sampler.split("_")[-1]))
            das_kn[0].append(1 - float(sampler.split("_")[-1]))
            dataframe.append({"ood_dataset": ood_dataset, "Bias Severity":1 - float(sampler.split("_")[-1]), "AUROC": das_kn[1][-1], "Method":"KNNDSD"})
            dataframe.append({"ood_dataset": ood_dataset, "Bias Severity":1 - float(sampler.split("_")[-1]), "AUROC": das_vn[1][-1], "Method":"Rabanser et Al."})

    df = pd.DataFrame(data=dataframe)
    sns.lineplot(data=df, x="Bias Severity", y="AUROC", hue="Method")
    plt.savefig("figures/bias_severity_lineplot.png")
    plt.show()
        # plt.plot(das_kn[0], das_kn[1],label="KNNDSD")
        # plt.plot(das_vn[0], das_vn[1],label="Rabanser et Al.")
        # plt.title(str(ood_dataset))
        # plt.legend()
        # plt.show()


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

def plot_severity(dataset,sample_size):
    df_knn = open_and_process(f"data/{dataset}_severity_ks_5NN_{sample_size}.csv", filter_noise=False)
    df_rab = open_and_process(f"data/{dataset}_severity_ks_{sample_size}.csv", filter_noise=False)
    df_typ = open_and_process(f"data/{dataset}_severity_typicality_{sample_size}.csv", filter_noise=False)

    df_rab["OOD Detector"] = "Rabanser et Al."
    df_knn["OOD Detector"] = "KNNDSD"
    df_typ["OOD Detector"] = "Typicality"

    threshold_rab = get_threshold(df_rab)
    threshold_knn = get_threshold(df_knn)
    threshold_typ = get_threshold(df_typ)
    thres_dict = dict(zip(["Rabanser et Al.", "KNNDSD","Typicality"], [threshold_rab, threshold_knn, threshold_typ]))
    merged = pd.concat((df_knn, df_rab, df_typ))
    # g = sns.FacetGrid(data=merged, col="OOD Detector", row="sampler", sharey=False, sharex=False, margin_titles=True)
    # g.map_dataframe(sns.scatterplot, x="pvalue", y="loss", hue="fold",  palette="mako")
    # g.set(xscale="log").set(ylim=0)
    # plt.tight_layout()
    # plt.show()
    data = []
    # print("Sampler & KNNDSD & Rabanser\\\\")

    for sampler in merged["sampler"].unique():
        print(sampler)
        sampler_val =1-float(sampler.split('_')[-1][:4])
        # print(f"{1-float(sampler.split('_')[-1][:4]):.3}",end=": ")
        by_sampler = merged[merged["sampler"]==sampler]
        for ood_detector in by_sampler["OOD Detector"].unique():
            by_dsd = by_sampler[by_sampler["OOD Detector"]==ood_detector]
            corr = correlation(by_dsd)
            risk_val = balanced_accuracy(by_dsd, threshold=thres_dict[ood_detector])
            data.append({"Severity":sampler_val, "OOD Detector":ood_detector, "Correlation":corr, "BA": risk_val})
            # print(f"\t& {corr:.4}", end=", ")
            # print(f"\t& {risk_val:.4}", end=", ")

        # print()
    data = pd.DataFrame(data)
    # plt.ylim(0,1)
    sns.lineplot(data=data[data["OOD Detector"]=="Rabanser et Al."], x="Severity", y="BA", hue="OOD Detector")
    plt.show()
    sns.lineplot(data=data, x="Severity", y="BA", hue="OOD Detector")
    plt.show()
def summarize_results(placeholder=False):
    df = get_classification_metrics_for_all_experiments(placeholder=placeholder)
    df = df.groupby(["Dataset", "OOD Detector"])[["FPR", "FNR", "DR", "AUROC", "AUPR"]].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def threshold_plots(dataset_name, sample_size, filter_noise=False):
    df = get_classification_metrics_for_all_experiments()
    df = df[(df["Dataset"]==dataset_name)&(df["Sample Size"]==sample_size)]
    print(df)

def get_correlation_metrics_for_all_experiments(placeholder=False):
    #summarize overall results;
    table_data = []
    for dataset in ["CIFAR100_normal", "CIFAR100_normal", "NICO_noise", "Njord_noise", "Polyp_noise", "imagenette_noise"]:
        for dsd in ["ks", "ks_5NN", "typicality"]:
            for sample_size in [10, 20, 50, 100, 200, 500]:
                fname = f"data/{dataset}_{dsd}_{sample_size}.csv"

                data = open_and_process(fname)
                if data is None:
                    if placeholder:
                        table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                           "FPR": -1,
                                           "DR": -1,
                                           "Risk": -1})
                    continue
                for sampler in pd.unique(data["sampler"]):
                    subset = data[data["sampler"] == sampler]
                    table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                       "Sampler": sampler,
                                       "Correlation": correlation(subset)})
    df = pd.DataFrame(data=table_data).replace("ks_5NN", "KNNDSD").replace("ks", "KS").replace(
        "typicality", "Typicality")
    return df

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
                                           "AUPR": aupr(subset),
                                           "Correlation": correlation(subset)})
    df = pd.DataFrame(data=table_data)
    return df

def get_classification_metrics_for_all_grad_experiments(placeholder=False):
    #summarize overall results;
    table_data = []
    for dataset in ["CIFAR10_normal", "CIFAR100_normal", "NICO_normal", "Njord_normal", "Polyp_normal", "imagenette_normal"]:
        for sample_size in [10, 20, 50, 100]:
            for dsd in ["jacobian", "grad_magnitude", "condition_number", "tall_jacobian_eig", "jtjsvd", "jtjmag", "odin", "adv_jacobian"]:
                fname = f"grad_data/{dataset}_{dsd}{sample_size}.csv"
                data = open_and_process(fname, filter_noise=True)
                if data is None:
                    if placeholder:
                        table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                           "FPR": -1,
                                           "FNR": -1,
                                           "DR": -1,
                                           "Risk": -1})
                    continue
                threshold = get_threshold(data)
                for sampler in pd.unique(data["sampler"]):
                    subset = data[data["sampler"] == sampler]
                    table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                       "Sampler": sampler,
                                       "FPR": fpr(subset, threshold=threshold),
                                       "FNR": fnr(subset, threshold=threshold),
                                       "DR": balanced_accuracy(subset, threshold=threshold),
                                       "Correlation": correlation(subset)})
    df = pd.DataFrame(data=table_data).replace("ks_5NN", "KNNDSD").replace("ks", "KS").replace(
        "typicality", "Typicality")
    return df


def experiment_prediction(fname):
    data = open_and_process(fname, combine_losses=True)
    data["pvalue"]=data["pvalue"].apply(lambda x: np.log10(x))
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(axis=1, how='any', inplace=True)

    # probability approach
    num_bins = 10
    labels = range(1, num_bins + 1)  # The bin labels; ensures unique labels for each bin
    data["bin"], bin_edges = pd.qcut(data["pvalue"], q=num_bins, retbins=True, labels=labels)

    # Convert the bin labels back to integers for consistency
    data["bin"] = data["bin"].astype(int)
    # bins = np.linspace(min(data["pvalue"]), max(data["pvalue"]), num_bins)
    # data["bin"] = np.digitize(data["pvalue"], bins)
    data["bin"] = data["bin"].apply(lambda x: round(bin_edges[x-1]))
    print(data["bin"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=data, x="bin", y="loss", scatter=False)
    sns.violinplot(data=data, ax=ax, x="bin", y="loss", positions=np.unique(data["bin"]))
    plt.show()
    x = data["loss"]
    g = data["bin"]
    # df = pd.DataFrame(dict(x=x, g=g))

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(data, row="bin", hue="bin", aspect=7, height=1, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "loss",
          bw_adjust=.4, clip_on=False,
          fill=True, alpha=1, linewidth=1.5, clip=(0,None))
    g.map(sns.kdeplot, "loss", clip_on=False, color="w", lw=2, bw_adjust=.4)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=16)

    g.map(label, "loss")


    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    ylabel= g.axes[0][0].set_ylabel("log(p) bins", fontsize=16)
    ylabel.set_position((ylabel.get_position()[0], -3))
    plt.xlabel("Loss", fontsize=16)
    plt.savefig(f"figures/{fname.split('/')[-1]}_loss_vs_pvalue_pdf.eps")
    plt.show()

def plot_lossvp_for_fold():
    # def wasserstein_from_random(data):
    #
    #     random = np.array(data[data["sampler"]=="RandomSampler"]["pvalue"]).reshape(-1,1)
    #     # print(data["sampler"].unique())
    #     bias = np.array(
    #         data[(data["sampler"]!="RandomSampler")&(data["sampler"]!="ClusterSampler")]["pvalue"]
    #
    #         ).reshape(-1,1)
    #     s = MinMaxScaler()
    #     random = s.fit_transform(random)
    #     bias = s.transform(bias)
    #     return np.abs(random.mean()-bias.mean())
    #     # return
    #     # return ks_2samp(random["pvalue"], cluster["pvalue"])[1]
    # def fold_to_float(fold):
    #     return round(float(fold.split("_")[-1]),2) if fold!="ind" else 0
    #
    df = aggregate_covariate_data()
    njord = df[df["Dataset"]=="Njord"]
    njord = njord[njord["Sample Size"]==200]
    print(njord)
    g = sns.FacetGrid(data=njord,col="OOD Detector", row="sampler", sharex=False, sharey=False, margin_titles=True)
    g.map_dataframe(sns.scatterplot, y="loss", x="pvalue", hue="fold", palette="mako")
    g.set(xscale="log")
    plt.show()
    g.add_legend(bbox_to_anchor=(1.05, 0.6), loc='upper left')
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left')
    g.tight_layout()
    plt.savefig("figures/lossvp_fold.png")
    plt.show()

def plot_severities():
    dfs = []
    for dataset in ["CIFAR10", "imagenette", "CIFAR100"]:
        for dsd in ["ks_5NN", "ks", "typicality"]:
            df = open_and_process(f"{dataset}_severity_{dsd}_100.csdv")
            df["Dataset"]=dataset
            df["OOD Detector"]=dsd
            dfs.append(df)
    merged = pd.concat(dfs)
    grouped = sns.groupby(["Dataset", "sampler", "OOD Detector"]).apply(lambda x: balanced_accuracy())
    # sns.lineplot(dfs, x="sampler", y=)

def compare_organic_and_synthetic_shifts(dataset):
    for detector in ["ks_5NN", "ks"]:
        for sample_size in [100]:
            noise_data = open_and_process(f"data/{dataset}_noise_{detector}_{sample_size}.csv")
            organic_data = open_and_process(f"data/{dataset}_normal_{detector}_{sample_size}.csv")
            noise_data["type"]="noise"
            organic_data["type"]="organic"
            merged = pd.concat((noise_data, organic_data))
            sns.scatterplot(data=noise_data, x="pvalue", y="loss", hue="fold", palette="mako")
            sns.scatterplot(data=organic_data, x="pvalue", y="loss", hue="fold", palette="rocket")
            plt.legend([],[], frameon=False)
            plt.xscale("log")
            plt.show()


def illustrate_bias_types():
    from sklearn.decomposition import PCA
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sampler_lookup = {"RandomSampler": "No Bias", "ClassOrderSampler": "Class Bias",
                      "SequentialSampler": "Temporal Bias",
                      "ClusterSampler": "Synthetic Bias"}
    for i, testbed in enumerate([CIFAR10TestBed(100, "classifier", mode="normal"),
                    CIFAR100TestBed(100, "classifier", mode="normal"),
                    ImagenetteTestBed(100, "classifier", mode="noise"),
        ]):
        dsd = RabanserSD(testbed.classifier)
        train = dsd.get_encodings(testbed.ind_loader())
        ind_val_loaders = testbed.ind_val_loaders()
        val_encodings = dict(
                zip(ind_val_loaders.keys(),
                    [dict(zip(loader_w_sampler.keys(),
                             [dsd.get_encodings(loader)
                              for sampler_name, loader in loader_w_sampler.items()]
                             )) for
                         loader_w_sampler in ind_val_loaders.values()]))
        val_encodings = val_encodings["ind"]
        pca = PCA(n_components=2)
        train = pca.fit_transform(train)
        trans_val_rand = pca.transform(val_encodings["RandomSampler"])
        # ax[i].scatter(trans_val_rand[:,0],trans_val_rand[:,1], c="grey", alpha=0.5)
        sns.kdeplot(x=trans_val_rand[:,0],y=trans_val_rand[:,1], alpha=0.9, ax=ax[i], levels=10, fill=True, cmap="viridis")
        for j, sampler in enumerate(val_encodings.keys()):
            trans_val = pca.transform(val_encodings[sampler])
            sns.scatterplot(x=trans_val[:100, 0], y=trans_val[:100, 1], label=sampler_lookup[sampler], ax=ax[i], color=sns.color_palette()[j],legend=False)
        ax[i].set_title(testbed.__class__.__name__[:-7])
        ax[i].set(yticks=[])
        ax[i].set(xticks=[])
    ax[1].legend(title="Bias Type", bbox_to_anchor=(0.5, -0.2), loc='upper center', fancybox=True, ncol=3)
    plt.tight_layout()
    plt.savefig("figures/biased_encodings.png")
    plt.show()


def compare_testbed_encs():
    testbed1 = SemanticTestBed32x32(100, "classifier", mode="MNIST")
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    ind_val_loader = MNIST3("../../Datasets/MNIST", train=False, transform=trans)
    ood_loader = EMNIST3("../../Datasets/EMNIST", train=False, transform=trans)
    dsd = RabanserSD(testbed1.classifier)
    encodings = np.zeros((len(ind_val_loader), 32, dsd.rep_model.latent_dim))
    ys = np.zeros((len(ind_val_loader), 32))
    from tqdm import tqdm
    loader = DataLoader(ind_val_loader, batch_size=32, drop_last=True, shuffle=True)
    for i, (x,y) in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            x = x.to("cuda")
            encodings[i] = dsd.rep_model.get_encoding(x).cpu().numpy()
            ys[i]=y
    print(ys)
    encodings = encodings.reshape(-1, dsd.rep_model.latent_dim)[:500]
    ys = ys.reshape(-1)[:500]
    #
    #
    #
    #
    # test_encodings =
    #
    # val_encodings = val_encodings["ind"]["RandomSampler"][:500]
    # test_encodings = test_encodings["EMNIST"]["RandomSampler"][:500]
    from sklearn.manifold import Isomap
    # print("sneeding")
    pca = PCA(n_components=2)
    #
    trans = pca.fit_transform(encodings)
    print("sneeeded")
    sns.scatterplot(x=trans[:len(encodings),0],y=trans[:len(encodings),1], alpha=0.9, hue=ys)
    plt.xscale("log")
    plt.show()
    # # sns.scatterplot(x=trans[len(val_encodings):,0],y=trans[len(val_encodings):,1], alpha=0.9, label="EMNIST")
    # plt.legend()
    # plt.show()
    plt.savefig("figures/mnist_classes.png")
    print("all done")
def plot_pvaluedist():
    df = open_and_process("data/CIFAR10_ks_100_fullloss.csv", combine_losses=False)
    df["pvalue"]=df["pvalue"].apply(lambda x: np.log10(x))
    print(df)
    sns.kdeplot(data=df, x="pvalue", hue="fold", palette="mako")
    plt.show()

def boxplot_test():
    data = get_classification_metrics_for_all_experiments()
    data["Type"] = data["OOD Detector"].apply(lambda x: "5NN" in x)

    sns.boxplot(data=data, x="Sampler", y="DR", hue="KNNDSD")
    plt.savefig("figures/boxplot_test.png")
    plt.show()
    plt.close()

def test_simple_regressor():
    pass

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

