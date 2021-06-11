import torch.nn as nn
import torch.nn.functional as F


class LIBatchNorm(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(LIBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
    
    def interpolated_forward(self, x, alpha, w1, w2, b1, b2):
        self._check_input_dim(x)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        
        if w1 is not None:
            w = (1 - alpha) * w1 + alpha * w2
        else:
            w = None
        if b1 is not None:
            b = (1 - alpha) * b1 + alpha * b2
        else:
            b = None
        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            w, b, bn_training, exponential_average_factor, self.eps)


class LIBatchNorm1d(LIBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class LIBatchNorm2d(LIBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
