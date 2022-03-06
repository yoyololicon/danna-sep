import torch
import torch.nn.functional as F
import torchaudio
import gdown
import os
import argparse
from functools import partial
import onnxruntime as ort

from .infer import demucs_sep, tf_sep

package_dir = os.path.dirname(__file__)
checkpoint_env = "DANNA_CHECKPOINTS"
default_checkpoints = os.path.expanduser('~/danna-sep-checkpoints')
jitted_dir = os.environ.get(checkpoint_env, default_checkpoints)

model_file_url = {
    'xumx_old': 'https://drive.google.com/uc?id=1DCTJQ1ei4klR9L69fjbBGILMdJAoNN78',
    'xumx': 'https://drive.google.com/uc?id=1rpkvcaoKmHbSP5fk0zwhmDmqiSGVXXT8',
    'unet': 'https://drive.google.com/uc?id=1qzBbT8UIKKWNMs-kiB3MNHGccdwdBJVf',
    'demucs': 'https://drive.google.com/uc?id=1slSKOd7P-YmJ0HnjzaTOKQaOn3pQN0Ge'
}

model_file_path = {
    'xumx_old': os.path.join(jitted_dir, 'xumx_mwf.pth',),
    'xumx': os.path.join(jitted_dir, 'xumx.onnx'),
    'unet': os.path.join(jitted_dir, 'unet_attention.pth'),
    'demucs': os.path.join(jitted_dir, 'demucs_4_decoders.pth')
}


blending_weights_3 = torch.tensor([[0.2, 0.17, 0.5, 0.4],
                                   [0.6, 0.73, 0.5, 0.4],
                                   [0.2, 0.1, 0., 0.2]])

blending_weights_2 = torch.tensor([[0.4, 0.27, 0.5, 0.6],
                                   [0.6, 0.73, 0.5, 0.4]])

SAMPLERATE = 44100

parser = argparse.ArgumentParser('A music source separation tool')
parser.add_argument('infile', type=str, help='input audio file')
parser.add_argument('--outdir', type=str, default='./',
                    help='output directory. Default to current working directory')
parser.add_argument('--fast', action='store_true',
                    help='faster inference using only two of the models')


def entry():
    args = parser.parse_args()
    infile = args.infile
    file_name, ext = os.path.splitext(os.path.basename(infile))

    y, sr = torchaudio.load(infile, channels_first=True)
    if sr != SAMPLERATE:
        y = torchaudio.functional.resample(y, sr, SAMPLERATE)

    orig_length = y.size(1)
    y = F.pad(y, [0, 1024])

    if args.fast:
        models = ['unet', 'demucs']
        blending_weights = blending_weights_2
        sep_func = [
            partial(tf_sep, batching=True),
            partial(demucs_sep, rate=SAMPLERATE, shifts=1)
        ]
    else:
        models = ['unet', 'demucs', 'xumx']
        blending_weights = blending_weights_3
        sep_func = [
            partial(tf_sep, batching=True),
            partial(demucs_sep, rate=SAMPLERATE, shifts=1),
            tf_sep
        ]

    os.makedirs(jitted_dir, exist_ok=True)

    result = 0
    for model_name, weights, func in zip(models, blending_weights, sep_func):
        file_path = model_file_path[model_name]
        ensure_model_exist(
            model_file_url[model_name], file_path)

        if file_path.endswith('.onnx'):
            ort_session = ort.InferenceSession(file_path)

            def sep(y):
                y = torch.from_numpy(
                    ort_session.run(
                        None,
                        {ort_session.get_inputs()[0].name: y.numpy()}
                    )[0])
                return y
            pred = func(y, sep)
        else:
            jitted = torch.jit.load(file_path)
            pred = func(y, jitted)
        result += pred[..., :orig_length] * weights[:, None, None]

    for i, target in enumerate(['drums', 'bass', 'other', 'vocals']):
        path = os.path.join(args.outdir, f'{file_name}_{target}.wav')
        torchaudio.save(path, result[i], SAMPLERATE)

    return


def ensure_model_exist(url, file_path):
    if not os.path.isfile(file_path):
        print("downloading pre-trained model ...")
        gdown.download(url, output=file_path)
