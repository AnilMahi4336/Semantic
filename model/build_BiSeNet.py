import torch
from torch import nn
from model.build_contextpath import build_contextpath
import warnings
warnings.filterwarnings(action='ignore')
#from model.mod import MobileNet
from model.MobileNetV2 import MobileNetV2
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.x= MobileNet()
        self.x = MobileNetV2(output_stride=8, BatchNorm=nn.BatchNorm2d)
    
    def forward(self, input):
        #print(input.size())
        y,z=self.x(input)
        m = nn.MaxPool2d(3, stride=2,padding=1)
        o = m(z)
        #print(o.size())
        #print(y.size())
        a = torch.cat((y,o),dim=1)
        return a

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        self.in_channels = 3416
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)
        
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()
        
        # build context path
        self.context_path = build_contextpath(name=context_path)
        
        # build attention refinement module
        self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
        self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
        
        # build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(num_classes)
        
        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        
        # supervision block
        self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
    
    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)
        #print(sx.size())
        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)
        
        if self.training == True:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, scale_factor=8, mode='nearest')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, scale_factor=8, mode='nearest')
        # output of feature fusion module
        #cx=cx.avgpool(cx)
        result = self.feature_fusion_module(sx, cx)
        
        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='nearest')
        result = self.conv(result)
        
        if self.training == True:
            return result, cx1_sup, cx2_sup
            
    return result


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
    model = BiSeNet(3)
    model = nn.DataParallel(model)
    
    model = model.cuda()
    for key in model.named_parameters():
        print(key[1].device)
    x = torch.rand(2, 3, 256, 256)
    record = model.parameters()
    # for key, params in model.named_parameters():
    #     if 'bn' in key:
    #         params.requires_grad = False
    y = model(x)
