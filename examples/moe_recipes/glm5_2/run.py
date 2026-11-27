# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Render or execute a GLM-5.2 example inside the qualified runtime environment."""

import argparse
import os
import shlex
import sys
from pathlib import Path
from string import Template

import yaml

# Run by --prewarm-deepgemm in the recipe environment on one GPU: prints where the runtime
# artifacts are imported from, then compiles DeepGEMM's indexer kernels into DG_JIT_CACHE_DIR (or
# loads them), so the training ranks only load cached kernels.
_PREWARM_DEEPGEMM = '''
import importlib, os, sys
import torch
for name in ("cudnn", "cutlass", "mok", "emerging_optimizers", "ds_tune", "deep_gemm"):
    module = importlib.import_module(name)
    print(f"{name} {getattr(module, '__version__', '-')} {os.path.dirname(module.__file__)}")
from megatron.core.transformer.experimental_attention_variant import dsa_deepgemm_scorer
dsa_deepgemm_scorer.warmup(num_heads=int(sys.argv[1]))
torch.cuda.synchronize()
print(f"DeepGEMM JIT cache ready: {os.environ['DG_JIT_CACHE_DIR']} ({sys.argv[1]} indexer heads)")
'''


def _arguments(values: dict) -> list[str]:
    result = []
    for name, value in values.items():
        if value is None or value is False:
            continue
        result.append('--' + name.replace('_', '-'))
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def _variant_arguments(recipe: dict, variant: str | None) -> dict:
    arguments = dict(recipe['ARGS'])
    if variant is not None:
        variants = recipe.get('VARIANTS', {})
        if variant not in variants:
            raise ValueError(f'Unknown variant {variant!r}; the recipe defines {sorted(variants)}.')
        unknown = sorted(set(variants[variant]) - set(arguments))
        if unknown:
            raise ValueError(f'Variant {variant!r} sets arguments absent from ARGS: {unknown}.')
        arguments.update(variants[variant])
    return arguments


def _read_recipe(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    """A topology recipe may reuse a sibling's workload and override its rank layout."""
    recipe = yaml.safe_load(path.read_text())
    if 'BASE_RECIPE' not in recipe:
        return recipe

    parent = path.with_name(recipe['BASE_RECIPE'])
    if parent.resolve().parent != path.resolve().parent:
        raise ValueError('BASE_RECIPE must be a file in the same directory.')
    ancestors = (*ancestors, path.resolve())
    if parent.resolve() in ancestors:
        raise ValueError(f'Cyclic BASE_RECIPE reference to {parent.name}.')
    base = _read_recipe(parent, ancestors)
    # A parent's named workload is resolved before applying the child's shared rank layout.
    variants = {
        name: {**_variant_arguments(base, source), **recipe['ARGS']}
        for name, source in recipe.get('BASE_VARIANTS', {}).items()
    }
    dependencies = dict(base['DEPENDENCIES'])
    dependencies['hardware'] = {**dependencies['hardware'], **recipe['HARDWARE']}
    return {
        **base,
        'DEPENDENCIES': dependencies,
        'ARGS': {**_variant_arguments(base, recipe.get('BASE_VARIANT')), **recipe['ARGS']},
        'VARIANTS': {**variants, **recipe.get('VARIANTS', {})},
    }


def main() -> None:
    """Translate ENV_VARS and ARGS without importing the GPU training stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument(
        '--variant', help="Apply one entry of the recipe's VARIANTS (e.g. frozen_indexer) to ARGS."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the environment and command; do not run training.',
    )
    mode.add_argument(
        '--prewarm-deepgemm',
        action='store_true',
        help="Fill DG_JIT_CACHE_DIR with DeepGEMM's indexer kernels on one GPU; do not train.",
    )
    options = parser.parse_args()
    try:
        recipe = _read_recipe(options.recipe)
        arguments = _variant_arguments(recipe, options.variant)
    except ValueError as error:
        parser.error(str(error))

    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, MEGATRON_ROOT=str(root))
    overrides = {}
    try:
        for name, value in recipe['ENV_VARS'].items():
            overrides[name] = Template(str(value)).substitute(environment)
    except KeyError as error:
        parser.error(f'Missing environment variable {error.args[0]}; see the recipe README.')
    environment.update(overrides)
    if options.prewarm_deepgemm:
        if arguments.get('dsa_indexer_scorer_backend') != 'deepgemm':
            parser.error('This recipe does not score the DSA indexer with DeepGEMM.')
        command = [sys.executable, '-c', _PREWARM_DEEPGEMM, str(arguments['dsa_indexer_n_heads'])]
    else:
        entrypoint = recipe.get('ENTRYPOINT', 'pretrain_gpt.py')
        if entrypoint not in {'pretrain_gpt.py', 'pretrain_hybrid.py'}:
            parser.error(f'Unsupported training entry point {entrypoint!r}.')
        command = [sys.executable, str(root / entrypoint), *_arguments(arguments)]
        if entrypoint == 'pretrain_hybrid.py' and (project := os.environ.get('WANDB_PROJECT')):
            command.extend(['--wandb-project', project])
            name = os.environ.get('WANDB_NAME', options.recipe.stem)
            command.extend(['--wandb-exp-name', name])

    if options.dry_run:
        exports = [f'{name}={value}' for name, value in overrides.items()]
        print(f'cd {shlex.quote(str(root))} && {shlex.join(["env", *exports, *command])}')
        return

    if not options.prewarm_deepgemm:
        expected_world_size = recipe['DEPENDENCIES']['hardware']['num_gpus']
        if environment.get('WORLD_SIZE') != str(expected_world_size):
            parser.error(
                f'This recipe requires {expected_world_size} distributed ranks; '
                'launch it as the recipe README describes.'
            )
    os.chdir(root)
    os.execvpe(sys.executable, command, environment)


if __name__ == '__main__':
    main()
