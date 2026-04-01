import logging, os, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def cal_dice_old(pred, mask, nclass):
    """
    计算每个类别的 Dice 系数。

    参数:
    pred (torch.Tensor): 预测的标签，形状为 (b, D, H, W)。
    mask (torch.Tensor): 真实的标签，形状为 (b, D, H, W)。
    num_classes (int): 类别数量。

    返回:
    list: 每个类别的 Dice 系数列表。
    """
    pseduo_mask = pred
    if pseduo_mask.shape[0] == 1:
        pred_np = pseduo_mask.squeeze(0).cpu().numpy() # let batchsize=1
    else:
        pred_np = pseduo_mask.cpu().numpy()
    mask_np = mask.squeeze(0).cpu().numpy()
    dice_scores = []
    num = 0
    for c in range(nclass):
        pred_c = (pred_np == c)
        mask_c = (mask_np == c)
        intersection = np.sum(pred_c * mask_c)
        pred_sum = np.sum(pred_c)
        mask_sum = np.sum(mask_c)
        if pred_sum == 0 and mask_sum == 0:
            dice = 0.0
        else:
            dice = (2. * intersection) / (pred_sum + mask_sum + 1e-8)
            num += 1
        dice_scores.append(dice)
    mdice = np.array(dice_scores).sum() / num
    mdice = float(f"{mdice:.4g}")
    dice_scores = [float(f"{x:.4g}") for x in dice_scores]
    return mdice, dice_scores


def cal_dice(pred: torch.Tensor, mask: torch.Tensor, nclass: int) -> tuple[float, list[float]]:
    """
    高效计算每个类别的 Dice 系数 (全 PyTorch/CUDA 向量化)。

    参数:
    pred (torch.Tensor): 预测的标签，形状为 (D, H, W) 或 (B, D, H, W)。
                         !!! 注意: 预期 pred 是 Argmax 后的整数标签 (Dtype: Long/Int) !!!
    mask (torch.Tensor): 真实的标签，形状为 (D, H, W) 或 (B, D, H, W)。
    nclass (int): 类别数量。

    返回:
    tuple: (平均 Dice 系数 (float), 每个类别的 Dice 系数列表 (list[float]))
    """
    pred = pred.long()
    mask = mask.long()

    dice_scores_tensor = torch.zeros(nclass, dtype=torch.float32, device=pred.device)
    epsilon = 1e-8
    valid_classes = 0.0
    
    for c in range(nclass):
        pred_c = (pred == c)
        mask_c = (mask == c)
        
        intersection = torch.sum(pred_c & mask_c).float() # 交集
        pred_sum = torch.sum(pred_c).float() # 预测中 c 类的像素数
        mask_sum = torch.sum(mask_c).float() # 真实中 c 类的像素数
        denominator = pred_sum + mask_sum
        
        if denominator > epsilon:
            dice = (2. * intersection) / (denominator + epsilon)
            dice_scores_tensor[c] = dice
            valid_classes += 1.0
        else:
            # 如果两者都没有该类别，将其保持为 0.0 (初始化值)，不计入有效类别
            pass
    if valid_classes > 0:
        mdice = dice_scores_tensor.sum() / valid_classes
    else:
        mdice = torch.tensor(0.0, device=pred.device)

    mdice_float = float(f"{mdice.item():.4g}")
    dice_scores_list = [float(f"{x.item():.4g}") for x in dice_scores_tensor]
    
    return mdice_float, dice_scores_list

# control the teacher model to update
def update_ema_model(ema_model, std_model, ema_ratio):
    """
    use ema method
    """
    # ema_model_params = ema_model.state_dict()
    # std_model_params = std_model.state_dict()
    
    # for name, param in std_model_params.items():
    #     ema_param = ema_model_params[name]
    #     # 更新规则: ema_param = decay * ema_param + (1 - decay) * student_param
    #     ema_model_params[name].copy_(
    #         decay * ema_param + (1 - decay) * param
    #     )
    for param, param_ema in zip(std_model.parameters(), ema_model.parameters()):
        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
    for buffer, buffer_ema in zip(std_model.buffers(), ema_model.buffers()):
        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
    
    return ema_model
    
logs = set()
def init_log(name, level=logging.INFO, log_file=None, console_output=True):
    if (name, level, log_file, console_output) in logs:
        return logging.getLogger(name)
    # logs.add((name, level))
    logs.add((name, level, log_file, console_output))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
        formatter = logging.Formatter(format_str)
        if console_output:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            if "SLURM_PROCID" in os.environ:
                rank = int(os.environ["SLURM_PROCID"])
                logger.addFilter(lambda record: rank == 0)
            # else:
            #     rank = 0
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, index):
        # target = target.float()
        # smooth = 1e-10
        # intersect = torch.sum(score * target)
        # y_sum = torch.sum(target * target)
        # z_sum = torch.sum(score * score)
        # loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # loss = 1 - loss
        # return loss
        mask_positive_target = target == index
        mask_positive_score = score == index

        new_target = torch.zeros_like(target, dtype=torch.float)
        new_score = torch.zeros_like(score, dtype=torch.float)

        new_target[mask_positive_target] = 1.0
        new_score[mask_positive_score] = 1.0

        smooth = 1e-10
        intersect = torch.sum(new_score * new_target)
        y_sum = torch.sum(new_target)
        z_sum = torch.sum(new_score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        predicted_classes = torch.argmax(inputs, dim=1)
        if weight is None:
            weight = [0] * self.n_classes
        assert predicted_classes.size() == target.size(), 'predict & target shape do not match'
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            diceloss = self._dice_loss(predicted_classes, target, i)
            # class_wise_dice.append(1.0 - dice.item())
            loss += diceloss * (1.0 - weight[i])
        return loss / self.n_classes


class DiceLoss_most(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_most, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
class ClassProportionTracker():
    def __init__(self, num_classes):
        """
        初始化类别比例跟踪器。
        Args:
            num_classes (int): 类别数。
            device (str): 硬件。
        """
        self.num_classes = num_classes
        self.total_counts = torch.zeros(self.num_classes, dtype=torch.float32)
        self.iter_counts  = torch.zeros(self.num_classes, dtype=torch.float32)
        self.iter_times = 0.0
        
    def update_iter(self, label_tensor_batch):
        """
        更新一个轮次的类别计数。
        Args:
            label_tensor_batch (torch.Tensor): (batch_size, D, H, W)。
        """
        batch_pixels = label_tensor_batch.numel() # 批次中的总像素数
        label = label_tensor_batch.cpu()
        self.iter_times += 1.0
        
        for i in range(self.num_classes): # 统计每个类别的像素数量
            self.iter_counts[i] += (label == i).sum() / batch_pixels
    
    def update_epoch(self, epoch):
        """
        更新已经训练到的epoch的各个类别的比例。
        Args:
            epoch: int
        """
        if epoch > 0:
            self.iter_counts =  self.iter_counts / (self.iter_times + 1e-15)
        for i in range(self.num_classes):
            self.total_counts[i] = (self.total_counts[i] * epoch + self.iter_counts[i]) / (epoch + 1)
        self.iter_counts.zero_() # 重置所有计数器，以便重新开始统计。
        self.iter_times = 0.0
    
    def get_ratio(self, ):
        return self.total_counts

def get_current_consistency_weight(t, time1, time2, consist):
    """
    基于 Sigmoid 曲线的增长函数。
    
    参数:
    t: 时间点，可以是单个值或 NumPy 数组。
    time1: 增长开始的时间。
    time2: 增长结束的时间。
    consist: 最终达到的稳定值。
    """
    # Sigmoid 的拉伸和平移参数
    # 这里的参数是经验值，确保曲线在 (time1, time2) 之间完成增长, 当 t <= time1 时，函数值应为0
    if t < time1:
        return 0
    alpha = 10
    k = (2 * alpha) / (time2 - time1)

    # 当 t > time1 时，计算 Sigmoid 部分
    x = np.maximum(0, t - time1)
    output = consist / (1 + np.exp(-k * x + alpha))

    return output

class GACE(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, k=10, gama=0.5):
        self.k = k
        self.gama = gama
        super(GACE, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target.long()
        self.n_classes = inp.size()[1]

        i0 = 1
        i1 = 2
        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, self.n_classes)
        target = target.view(-1,)
        res = super(GACE, self).forward(inp, target)      
  
  
        n_instance = np.prod(res.shape)
        res, indices = torch.topk(res.view((-1, )), int(n_instance * self.k / 100), sorted=False)      
        target = torch.gather(target, 0, indices)        
        assert res.size() == target.size(), 'predict & target shape do not match'
        
        bg_w = np.power(int(n_instance * self.k / 100), self.gama)
        loss = 0.0
        smooth = 1e-10
        lista = []
        for i in range(0, self.n_classes):        
            target_cls = (target == i).float()  
            w = torch.pow(torch.sum(target_cls) + smooth, 1-self.gama) * bg_w
            lista.append(w)
            loss_cls = torch.sum(res * target_cls) / (w + smooth)
            loss += loss_cls

        return loss
    

class GADice(nn.Module):
    def __init__(self, GA=True):
        self.GA = GA
        super(GADice, self).__init__()

    def _one_hot_encoder(self, input_tensor):        
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, cls, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10

        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        
        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        self.n_classes = inputs.size()[1]
        if len(inputs.size()) == len(target.size())+1:
            target = target.unsqueeze(1)
        
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        smooth = 1e-10
        loss = 0.0
        for i in range(0, self.n_classes):
            if torch.sum(target[:, i]) > 0:
                dice_loss = self._dice_loss(i, inputs[:, i], target[:, i], weighted_pixel_map)
            else:
                if self.GA:
                    beta = inputs[:, i] / (torch.sum(1 - target[:, i]))
                    dice_loss = torch.sum(beta.detach() * inputs[:, i])                
            loss += dice_loss * weight[i]
        
        return loss / self.n_classes
    

class GACE_new(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, k=10, gama=0.5):
        self.k = k
        self.gama = gama
        super(GACE_new, self).__init__(weight, False, ignore_index, reduce=False)
    def forward(self, inp, target):
        target = target.long()
        self.n_classes = inp.size()[1]

        i0 = 1
        i1 = 2
        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, self.n_classes)
        target = target.view(-1,)
        res = super(GACE, self).forward(inp, target)      
  
  
        n_instance = np.prod(res.shape)
        res, indices = torch.topk(res.view((-1, )), int(n_instance * self.k / 100), sorted=False)      
        target = torch.gather(target, 0, indices)        
        assert res.size() == target.size(), 'predict & target shape do not match'
        
        bg_w = np.power(int(n_instance * self.k / 100), self.gama)
        loss = 0.0
        smooth = 1e-10
        for i in range(0, self.n_classes):        
            target_cls = (target == i).float()  
            w = torch.pow(torch.sum(target_cls) + smooth, 1-self.gama) * bg_w
            loss_cls = torch.sum(res * target_cls) / (w + smooth)
            loss += loss_cls

        return loss
    