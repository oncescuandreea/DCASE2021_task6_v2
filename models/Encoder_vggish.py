import numpy as np
import torch
import torch.nn as nn
from torch import hub
from typing import Iterable
import models.vggish_utils as utils

from models.vggish_utils import waveform_to_examples, wavfile_to_examples


class VGG(nn.Module):
    def __init__(self, features, use_relu: bool = False):
        super(VGG, self).__init__()
        self.use_relu = use_relu
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
        )
        self.upscale = nn.Linear(128, 512)
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        x = self.embeddings(x)
        x = self.upscale(x)
        if self.use_relu:
            x = self.final_relu(x)
        return x


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (utils.EMBEDDING_SIZE, utils.EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (utils.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == utils.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, utils.QUANTIZE_MIN_VAL, utils.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - utils.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (utils.QUANTIZE_MAX_VAL - utils.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers(in_channels: int):
    layers = []
    # in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())

class VGGish(VGG):
    def __init__(self, urls, config, device=None, pretrained=False, preprocess=True, postprocess=False, progress=True):
        super().__init__(make_layers(config.data.batch_size))
        if pretrained:
            state_dict = hub.load_state_dict_from_url(urls['vggish'], progress=progress)
            super().load_state_dict(state_dict)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.config = config
        if self.postprocess:
            self.pproc = Postprocessor()
            if pretrained:
                state_dict = hub.load_state_dict_from_url(urls['pca'], progress=progress)
                # TODO: Convert the state_dict to torch
                state_dict[utils.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                    state_dict[utils.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
                )
                state_dict[utils.PCA_MEANS_NAME] = torch.as_tensor(
                    state_dict[utils.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
                )

                self.pproc.load_state_dict(state_dict)
        self.to(self.device)

    def forward(self, x, fs=None):
        if self.preprocess:
            x = self._preprocess(x, fs)
        x = x.to(self.device)
        x = VGG.forward(self, x)
        if self.postprocess:
            x = self._postprocess(x)
        return x

    def _preprocess(self, x, fs):
        dataset = self.config.data.type
        if isinstance(x, np.ndarray):
            x = waveform_to_examples(x, fs)
        elif isinstance(x, str):
            x = wavfile_to_examples(x)
        elif isinstance(x, Iterable):
            if isinstance(x[0], str):
                x_batch = []
                for element in x:
                    x_prev = wavfile_to_examples(f'{self.config.path[dataset].data_location}/{element}')
                    # print(f'x_prev shape {x_prev.shape}')
                    max_dim = self.config.path[dataset].max_feats

                    x_padded = torch.zeros(max_dim, *x_prev.shape[1:])
                    x_padded[:x_prev.shape[0]] = x_prev
                    # print(f'x_padded shape {x_padded.shape}')
                    x_batch.append(x_padded)
                x = torch.cat(x_batch, dim=1)
            else:
                raise AttributeError
        else:
            raise AttributeError
        return x

    def _postprocess(self, x):
        return self.pproc(x)