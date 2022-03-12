if __name__ == '__main__':
    import os

    gpu_use = '1'
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import torch
import torch.nn.functional as F
import torchaudio
import os
import time
import argparse
from functools import partial
import onnxruntime as ort

try:
    from cli import model_file_url, blending_weights_3, blending_weights_2, ensure_model_exist
    from infer import demucs_sep, tf_sep, tf_sep_gpu, demucs_sep_gpu
except:
    from .cli import model_file_url, blending_weights_3, blending_weights_2, ensure_model_exist
    from .infer import demucs_sep, tf_sep, tf_sep_gpu, demucs_sep_gpu

package_dir = os.path.dirname(__file__)
checkpoint_env = "DANNA_CHECKPOINTS"
default_checkpoints = os.path.dirname(os.path.abspath(__file__)) + '/models/'
jitted_dir = os.environ.get(checkpoint_env, default_checkpoints)

model_file_path = {
    'xumx_old': os.path.join(jitted_dir, 'xumx_mwf.pth',),
    'xumx': os.path.join(jitted_dir, 'xumx.onnx'),
    'unet': os.path.join(jitted_dir, 'unet_attention.pth'),
    'demucs': os.path.join(jitted_dir, 'demucs_4_decoders.pth')
}


blending_weights_1 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])


SAMPLERATE = 44100

parser = argparse.ArgumentParser('A music source separation tool')
parser.add_argument('infile', type=str, help='input audio file')
parser.add_argument('--outdir', type=str, default='./',
                    help='output directory. Default to current working directory')
parser.add_argument('--fast', action='store_true',
                    help='faster inference using only two of the models')
parser.add_argument('--only_demux', action='store_true',  help='only_demux')
parser.add_argument('--only_unet', action='store_true',  help='only_unet')


def entry(args=None):
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
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
            partial(tf_sep_gpu, batching=True),
            partial(demucs_sep_gpu, rate=SAMPLERATE, shifts=1)
        ]
    elif args.only_demux:
        models = ['demucs']
        blending_weights = blending_weights_1
        sep_func = [
            partial(demucs_sep_gpu, rate=SAMPLERATE, shifts=1),
        ]
    elif args.only_unet:
        models = ['unet']
        blending_weights = blending_weights_1
        sep_func = [
            partial(tf_sep_gpu, batching=True),
        ]
    else:
        models = ['unet', 'demucs', 'xumx']
        blending_weights = blending_weights_3
        sep_func = [
            partial(tf_sep_gpu, batching=True),
            partial(demucs_sep_gpu, rate=SAMPLERATE, shifts=1),
            tf_sep
        ]

    os.makedirs(jitted_dir, exist_ok=True)

    result = 0
    for model_name, weights, func in zip(models, blending_weights, sep_func):
        file_path = model_file_path[model_name]
        ensure_model_exist(model_file_url[model_name], file_path)

        print('Run model: {} GPU'.format(model_name))
        start_time = time.time()
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
            model = torch.jit.load(file_path)
            pred = func(y, model)
        print('Time: {:.2f} sec'.format(time.time() - start_time))
        result += pred[..., :orig_length] * weights[:, None, None]

    for i, target in enumerate(['drums', 'bass', 'other', 'vocals']):
        path = os.path.join(args.outdir, f'{file_name}_{target}.wav')
        torchaudio.save(path, result[i], SAMPLERATE)

    return result


if __name__ == '__main__':
    entry()