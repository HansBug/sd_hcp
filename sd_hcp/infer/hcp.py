import glob
import logging
import os.path
from typing import List

from hbutils.system import TemporaryDirectory
from hcpdiff.infer_workflow import WorkflowRunner
from hcpdiff.utils import load_config_with_cli
from imgutils.data import load_image

from ..utils import data_to_cli_args

_CFG_FILE = os.path.abspath(os.path.join('cfgs', 'workflow', 'anime', 'highres_fix_anime.yaml'))
_DEFAULT_INFER_MODEL = 'Meina/MeinaMix_V11'


def sample_method_to_config(method):
    if method == 'DPM++ SDE Karras':
        return {
            '_target_': 'diffusers.DPMSolverSDEScheduler',
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'use_karras_sigmas': True,
        }
    elif method == 'DPM++ 2M Karras':
        return {
            '_target_': 'diffusers.DPMSolverMultistepScheduler',
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'algorithm_type': 'dpmsolver++',
            'beta_schedule': 'scaled_linear',
            'use_karras_sigmas': True
        }
    elif method == 'Euler a':
        return {
            '_target_': 'diffusers.EulerAncestralDiscreteScheduler',
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
        }
    else:
        raise ValueError(f'Unknown sample method - {method!r}.')


def infer_images(prompts: List[str], neg_prompts: List[str], seeds: List[int],
                 n_repeats: int = 2, pretrained_model: str = _DEFAULT_INFER_MODEL,
                 firstpass_width: int = 512, firstpass_height: int = 768, width: int = 832, height: int = 1216,
                 cfg_scale: float = 7, infer_steps: int = 30,
                 clip_skip: int = 2, sample_method: str = 'DPM++ 2M Karras'):
    with TemporaryDirectory() as output_dir:
        cli_args = data_to_cli_args({
            'bs': 1,
            'seed': seeds,

            'pretrained_model': pretrained_model,
            'prompt': prompts,
            'neg_prompt': neg_prompts,
            'N_repeats': n_repeats,

            'clip_skip': clip_skip - 1,
            # 'models_dir': os.path.join(workdir, 'ckpts'),
            # 'emb_dir': emb_dir,

            'infer_args': {
                'init_width': firstpass_width,
                'init_height': firstpass_height,
                'width': width,
                'height': height,
                'guidance_scale': cfg_scale,
                'num_inference_steps': infer_steps,
                'scheduler': sample_method_to_config(sample_method),
            },

            'output_dir': output_dir,
        })
        logging.info(f'Infer based on {_CFG_FILE!r}, with {cli_args!r}')
        cfgs = load_config_with_cli(_CFG_FILE, args_list=cli_args)  # skip --cfg

        runner = WorkflowRunner(cfgs)
        runner.start()

        files = sorted([
            (int(os.path.basename(png_file).split('-')[0]), png_file)
            for png_file in glob.glob(os.path.join(output_dir, '*.png'))
        ])
        images = []
        for _, png_file in files:
            image = load_image(png_file)
            image.load()
            images.append(image)

        return images
