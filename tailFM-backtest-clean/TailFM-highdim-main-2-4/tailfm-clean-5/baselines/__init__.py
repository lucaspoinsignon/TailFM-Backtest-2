"""Baseline generative models, all consuming and producing (N, n, f) windows so they
plug into the same scoring code as tailfm.

    timevae   PyTorch port of TimeVAE   (Desai et al. 2021; ref. impl. TF2/Keras)
    timegan   PyTorch port of TimeGAN   (Yoon et al., NeurIPS 2019; ref. impl. TF1)
    tailgan   adaptation of Tail-GAN    (Cont, Cucuringu, Xu, Zhang 2022; PyTorch)

Each module exposes

    fit_and_generate(train_windows, num_gen, seed=0, device=None, **hparams)
        -> np.ndarray of shape (num_gen, n, f)

and documents its deviations from the reference code.
"""

from .timevae import fit_and_generate as timevae_fit_and_generate
from .timegan import fit_and_generate as timegan_fit_and_generate
from .tailgan import fit_and_generate as tailgan_fit_and_generate

BASELINES = {
    "timevae": timevae_fit_and_generate,
    "timegan": timegan_fit_and_generate,
    "tailgan": tailgan_fit_and_generate,
}
