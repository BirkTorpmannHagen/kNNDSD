# from yellowbrick.features import PCA
from testbeds import CIFAR10TestBed, NicoTestBed
import torch
from torch.utils.data import DataLoader
from utils import *
from yellowbrick.features.manifold import Manifold
from sklearn.manifold import SpectralEmbedding, Isomap
from scipy.stats import ks_2samp
from bias_samplers import *
from torch.utils.data.sampler import RandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import CIFAR10,CIFAR100,MNIST
import pickle as pkl
import torch.utils.data as data
from domain_datasets import *
from torch.utils.data import RandomSampler
from classifier.resnetclassifier import ResNetClassifier
from ooddetectors import *
from testbeds import *


def compute_stats(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold, fname):
    df = convert_to_pandas_df(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold)
    df.to_csv(fname)


def collect_covariate_data(sample_range, testbed_constructor, dataset_name, feature_fn, mode="normal", k=0):
    for sample_size in sample_range:
        if feature_fn==typicality_ks_glow:
            if "Njord" in dataset_name:
                bench = testbed_constructor(sample_size, mode=mode, rep_model="vae")
            else:
                bench = testbed_constructor(sample_size, mode=mode, rep_model="glow")
            tsd = FeatureSD(bench.vae, feature_fn, k=k)
        else:
            bench = testbed_constructor(sample_size, "classifier", mode=mode)
            tsd = FeatureSD(bench.classifier, feature_fn, k=k)

        tsd.register_testbed(bench)
        if k!=0:
            name = f"new_data/{dataset_name}_{mode}_{feature_fn.__name__}_{k}NN_{sample_size}.csv"
        else:
            name = f"new_data/{dataset_name}_{mode}_{feature_fn.__name__}_{sample_size}.csv"

        compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                      fname=name)


def collect_all_covariate_data(sample_range):
    for testbeds in [CIFAR10TestBed, CIFAR100TestBed, NicoTestBed, PolypTestBed, NjordTestBed, ImagenetteTestBed]:
        for k in [0, 5]:
            for feature_fn in [cross_entropy, typicality_ks_glow, grad_magnitude, odin]:
                collect_covariate_data(sample_range, testbeds, testbeds.__name__.split("TestBed"), feature_fn, mode="normal", k=k)

def collect_all_semantic_data():
    for sample_size in [30, 50, 100, 200, 500]:
        for fold in ["CIFAR10", "CIFAR100", "MNIST", "EMNIST"]:
            for k in [0, 5]:
                for feature_fn in [cross_entropy, typicality_ks_glow, grad_magnitude, odin]:
                    if k!=0:
                        bench = SemanticTestBed32x32(sample_size, 10, mode=fold, rep_model="classifier")
                        tsd = FeatureSD(bench.classifier, feature_fn, k=k)
                        tsd.register_testbed(bench)
                        compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                                      fname=f"new_data/Semantic_{fold}_{feature_fn.__name__ }_{k}NN_{sample_size}.csv")
                    else:
                        bench = SemanticTestBed32x32(sample_size, 10, mode=fold, rep_model="classifier")
                        tsd = FeatureSD(bench.classifier, feature_fn)
                        tsd.register_testbed(bench)
                        compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                                      fname=f"new_data/Semantic_{fold}_{feature_fn.__name__}_{sample_size}.csv")

                if k != 0:
                    bench = SemanticTestBed32x32(sample_size, 10, mode=fold, rep_model="classifier")
                    tsd = RabanserSD(bench.classifier, feature_fn, k=k)
                    tsd.register_testbed(bench)
                    compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                                  fname=f"new_data/Semantic_{fold}_{feature_fn.__name__}_{k}NN_{sample_size}.csv")
                else:
                    bench = SemanticTestBed32x32(sample_size, 10, mode=fold, rep_model="classifier")
                    tsd = RabanserSD(bench.classifier, select_samples=True, k=k)
                    tsd.register_testbed(bench)
                    compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                                  fname=f"new_data/Semantic_{fold}_{feature_fn.__name__}_{sample_size}.csv")



if __name__ == '__main__':
    from features import *
    torch.multiprocessing.set_start_method('spawn')
    collect_all_semantic_data()
    collect_covariate_data([100], NicoTestBed, "NICO", feature_fn=cross_entropy, k=0, mode="normal")
