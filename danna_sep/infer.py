import torch
import torch.nn.functional as F
import numpy as np
import norbert
from .utils import *
# from tqdm import tqdm


def demucs_sep(
    audio,
    model,
    rate,
    shifts=5
):

    max_shift = int(rate * 0.5)
    batch_size = 1
    valid_length = audio.size(1)
    padded_length = valid_length + max_shift
    padded_audio = F.pad(audio, [max_shift] * 2)

    with torch.no_grad():
        y = torch.zeros_like(audio).repeat(4, 1, 1)
        s = []
        x = []
        for shift in torch.linspace(0, max_shift, steps=shifts, dtype=torch.long).tolist():
            s.append(shift)
            x.append(padded_audio[:, shift:shift+padded_length])

        x = torch.stack(x)
        for i in range(0, shifts, batch_size):
            shifted_y = model(x[i:i+batch_size])
            for j in range(shifted_y.shape[0]):
                shift = s[i + j]
                start = max_shift - shift
                length = min(valid_length, shifted_y.shape[-1] - start)
                y[..., :length] += shifted_y[j, ..., start:start + length]
        y /= shifts

    y = y.clamp(-0.99, 0.99)
    return y


def tf_sep(
    audio,
    model,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    batching=False
):

    # convert numpy audio to torch
    X = stft(audio)

    with torch.no_grad():
        mag = X.abs()
        if batching:
            frames = 2584
            masked_tf_rep = []
            for chunk in mag.split(frames, dim=-1):
                masked_tf_rep += [model(chunk.unsqueeze(0)).squeeze()]
            masked_tf_rep = torch.cat(masked_tf_rep, -1)
        else:
            masked_tf_rep = model(mag.unsqueeze(0)).squeeze()
        masked_tf_rep *= mag

    Vj = masked_tf_rep.numpy()
    if softmask:
        Vj **= alpha
    V = np.transpose(Vj, (3, 2, 1, 0))
    X = X.permute(2, 1, 0).numpy()

    if residual_model:
        V = norbert.residual_model(V, X, alpha if softmask else 1)

    Y = norbert.wiener(V, X.astype(np.complex128),
                       niter, use_softmask=softmask)

    Y = torch.from_numpy(Y).permute(3, 2, 1, 0)
    estimates = istft(
        Y.view(-1, *Y.shape[2:])).view(*Y.shape[:2], -1)

    return estimates.float()
