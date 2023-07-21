import torch.nn as nn


def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.DataParallel))


# class MultiGPU(nn.DataParallel):
#     def __getattr__(self, item):
#         try:
#             return super().__getattr__(item)
#         except:
#             pass
#         return getattr(self.module, item)
    
class MultiGPU(nn.DataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
