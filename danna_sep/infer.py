import torch
import torch.nn.functional as F
import numpy as np
import norbert
try:
    from .utils import *
except:
    from utils import *
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


def demucs_sep_gpu(
    audio,
    model,
    rate,
    shifts=5
):
    use_half = False
    device = torch.device('cuda')
    model.to(device)
    if use_half:
        model.eval().half()
    else:
        model.eval()

    with torch.no_grad():
        # Increase if you have more GPU memory (tuned for 11 GB)
        split_batch_size = 4000000
        step = split_batch_size // 2
        max_shift = int(rate * 0.5)
        delta = 2000
        batch_size = 1
        all_parts = torch.zeros((4,) + audio.shape, dtype=torch.float32)
        count = torch.zeros((4,) + audio.shape, dtype=torch.float32)
        for z in range(0, audio.shape[-1], step):
            if z + 2 * step > audio.shape[-1]:
                # For very small parts it failed (so don't allow small parts)
                audio_part = audio[..., z:]
                start_0 = 0
                end_0 = audio_part.shape[-1]
                start_save = z
                end_save = z + audio_part.shape[-1]
            elif z == 0:
                audio_part = audio[..., :, z:z + split_batch_size + delta]
                start_0 = 0
                end_0 = split_batch_size
                start_save = 0
                end_save = split_batch_size
            else:
                audio_part = audio[..., :, z - delta:z + split_batch_size + delta]
                start_0 = delta
                end_0 = split_batch_size + delta
                start_save = z
                end_save = z + split_batch_size

            # print(z, audio_part.shape, start_0, end_0)

            valid_length = audio_part.size(1)
            padded_length = valid_length + max_shift
            padded_audio = F.pad(audio_part, [max_shift] * 2)

            y = torch.zeros_like(audio_part).repeat(4, 1, 1).to(device)
            s = []
            x = []
            for shift in torch.linspace(0, max_shift, steps=shifts, dtype=torch.long).tolist():
                s.append(shift)
                x.append(padded_audio[:, shift:shift+padded_length])

            if use_half:
                x = torch.stack(x).to(device).half()
            else:
                x = torch.stack(x).to(device)
            for i in range(0, shifts, batch_size):
                shifted_y = model(x[i:i + batch_size])
                for j in range(shifted_y.shape[0]):
                    shift = s[i + j]
                    start = max_shift - shift
                    length = min(valid_length, shifted_y.shape[-1] - start)
                    y[..., :length] += shifted_y[j, ..., start:start + length]
            y /= shifts

            y = y[..., start_0:end_0]
            y = y.clamp(-0.99, 0.99)
            all_parts[..., start_save:end_save] = y
            count[..., start_save:end_save] += 1
            del y
            if z + 2 * step > audio.shape[-1]:
                break

        y = all_parts / count
        y1 = y.cpu()

    del y
    del model
    del all_parts
    del count
    torch.cuda.empty_cache()
    return y1


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


def tf_sep_gpu(
    audio,
    model,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    batching=False
):
    use_half = False
    device = torch.device('cuda')
    model.to(device)
    if use_half:
        model.eval().half()
    else:
        model.eval()

    # convert numpy audio to torch
    X = stft(audio)

    with torch.no_grad():
        if use_half:
            mag = X.abs().to(device).half()
        else:
            mag = X.abs().to(device)
        if batching:
            frames = 2584
            masked_tf_rep = []
            for chunk in mag.split(frames, dim=-1):
                # print(chunk.shape)
                if chunk.shape[-1] < 128:
                    chunk_large = torch.zeros(chunk.shape[:-1] + (128,)).to(device)
                    chunk_large[..., :chunk.shape[-1]] = chunk
                    rs = model(chunk_large.unsqueeze(0)).squeeze()
                    rs = rs[..., :chunk.shape[-1]]
                else:
                    rs = model(chunk.unsqueeze(0)).squeeze()
                # print(rs.shape)
                masked_tf_rep += [rs]
            masked_tf_rep = torch.cat(masked_tf_rep, -1)
        else:
            masked_tf_rep = model(mag.unsqueeze(0)).squeeze()
        masked_tf_rep *= mag
        del mag

    Vj = masked_tf_rep.cpu().numpy()

    if softmask:
        Vj **= alpha
    V = np.transpose(Vj, (3, 2, 1, 0))
    X = X.permute(2, 1, 0).numpy()

    if residual_model:
        V = norbert.residual_model(V, X, alpha if softmask else 1)

    Y = norbert.wiener(V, X.astype(np.complex128),
                       niter, use_softmask=softmask)

    Y = torch.from_numpy(Y).permute(3, 2, 1, 0)
    estimates = istft(Y.view(-1, *Y.shape[2:])).view(*Y.shape[:2], -1)

    del masked_tf_rep
    del model
    torch.cuda.empty_cache()
    return estimates.float()