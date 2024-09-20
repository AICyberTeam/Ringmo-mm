import copy
import traceback
import functools
import torch


class StageWrapper:
    def __init__(self, func, name=None):
        self.func = func
        self.name = name

    def __call__(self, input, **kwargs):
        return self.func(input, **kwargs)

    def __repr__(self):
        return f"stage_name:{self.name}"


class ModelFlow:
    def __init__(self, model):

        self.model = model
        self.stage_wrappers = model.components \
            if hasattr(model, "components") else [getattr(model, "forward")]
        self.roll_back()

    def roll_back(self):
        self.kwargs = {}
        self.current_stage_idx = 0
        self.output_values = []

    def is_end(self):
        if self.current_stage_idx < len(self.stage_wrappers):
            return False
        return True

    @property
    def outputs(self):
        return self.output_values

    def __len__(self):
        return len(self.stage_wrappers)

    @property
    def current_stage(self):
        return self.stage_wrappers[self.current_stage_idx]

    def next(self):
        self.current_stage_idx += 1

    def args_update(self, kw_out):
        if 'output' in kw_out:
            if isinstance(kw_out['output'], list) or isinstance(kw_out['output'], tuple):
                self.output_values += kw_out['output']
            elif isinstance(kw_out["output"], torch.Tensor):
                self.output_values.append(kw_out['output'])
            else:
                raise NotImplementedError
        self.kwargs.update(
            {
                key: value for key, value in kw_out.items() if key != 'output'
            }
        )
        self.current_stage_idx = (self.current_stage_idx + 1) % len(self)

    def one_stage(self, *args, stage, **kwargs):
        self.kwargs.update(kwargs)
        stage_result = stage(*args, **self.kwargs)

        if isinstance(stage_result, tuple) and len(stage_result) == 2 and isinstance(stage_result[-1], dict):
            x, kwargs_out = stage_result
        else:
            x = stage_result
            kwargs_out = {}
        self.args_update(kwargs_out)
        return x

    def __getitem__(self, idx):
        if idx == 0:
            self.roll_back()
        func = lambda *args, **kwargs: self.one_stage(*args, stage=self.stage_wrappers[idx], **kwargs)
        return func

    def __call__(self, *args, **kwargs):
        for stage_idx in range(len(self)):
            args = self[stage_idx](*args, **kwargs)
            if not isinstance(args, list) and not isinstance(args, tuple):
                args = [args]
        # print(type(args), len(args))
        if self.outputs == []:
            self.args_update(
                dict(output=args)
            )
        # print(len(self.outputs), type(self.outputs))#, self.model)
        return self.outputs


def cond_run(stage, feat_in):
    if isinstance(feat_in, tuple):
        feat_out = stage(*feat_in)
    elif isinstance(feat_in, dict):
        feat_out = stage(**feat_in)
    else:
        feat_out = stage(feat_in)
    return feat_out
