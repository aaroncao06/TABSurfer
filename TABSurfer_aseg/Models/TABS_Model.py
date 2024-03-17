import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

if __name__ == '__main__':
    from Transformer import TransformerModel
    from PositionalEncoding import LearnedPositionalEncoding
else:
    try: 
        from TABSurfer_aseg.Models.Transformer import TransformerModel
    # from BMEN4460_Final.TABS_model.Models.Transformer import TransformerModel
        from TABSurfer_aseg.Models.PositionalEncoding import LearnedPositionalEncoding
    except:
        from Models.Transformer import TransformerModel
    # from BMEN4460_Final.TABS_model.Models.Transformer import TransformerModel
        from Models.PositionalEncoding import LearnedPositionalEncoding

# from BMEN4460_Final.TABS_model.Models.PositionalEncoding import LearnedPositionalEncoding

class up_conv_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_3D, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class resconv_block_3D_new(nn.Module): #half size of og
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D_new, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):

        residual = self.Conv_1x1(x)
        x = self.conv(x)
        return residual + x
    
class resconv_block_3D_double(nn.Module): #2 resids
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D_new, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )
        self.Conv_1x1_resid1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)
        self.Conv_1x1_resid2 = nn.Conv3d(ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):
        resid1 = self.Conv_1x1_resid1(x)
        x = self.conv1(x)+resid1
        resid2 = self.Conv_1x1_resid2(x)
        x = self.conv2(x)+resid2
        return x
    
class resconv_block_3D(nn.Module): #og tabs block
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):

        residual = self.Conv_1x1(x)
        x = self.conv(x)
        return residual + x

class TABS_new(nn.Module): #TABSurfer v1, no WM
    def __init__(
        self,
        img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 32,
        embedding_dim = 1024,
        num_heads = 16,
        num_layers = 8,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        ):
        super(TABS_new,self).__init__()

        self.hidden_dim = int((img_dim/16)**3) * 8
        #self.hidden_dim = 2744

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Conv1 = resconv_block_3D_new(ch_in=img_ch,ch_out=32) # 96
        self.Conv2 = resconv_block_3D_new(ch_in=32,ch_out=64) # 48
        self.Conv3 = resconv_block_3D_new(ch_in=64,ch_out=128) # 24
        self.Conv4 = resconv_block_3D_new(ch_in=128,ch_out=256) # 12
        
        self.Up4 = up_conv_3D(ch_in=256,ch_out=128)
        self.Up_conv4 = resconv_block_3D_new(ch_in=256, ch_out=128)
        self.Up3 = up_conv_3D(ch_in=128,ch_out=64)
        self.Up_conv3 = resconv_block_3D_new(ch_in=128, ch_out=64)
        self.Up2 = up_conv_3D(ch_in=64,ch_out=32)
        self.Up_conv2 = resconv_block_3D_new(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv3d(32,output_ch,kernel_size=1,stride=1,padding=0)

        self.gn = nn.GroupNorm(8, 256)
        self.relu = nn.ReLU(inplace=True)
        self.act = nn.Softmax(dim=1)

        self.num_patches = int((img_dim // patch_dim) ** 3)

        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, self.hidden_dim
        )

        #self.reshaped_conv = conv_block_3D(512, 128)
        self.reshaped_conv = conv_block_3D(embedding_dim, 256)


        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            self.hidden_dim,

            dropout_rate,
            attn_dropout_rate,
        )

        self.conv_x = nn.Conv3d(
            256,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.embedding_dim = embedding_dim

        self.num_layers = num_layers

    def forward(self,input):
        # encoding path
        x1 = self.Conv1(input)
        del input
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x = self.Conv4(x4)
        del x4
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.position_encoding(x)
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        encoder_outputs = {}
        all_keys = []
        for i in range(1, self.num_layers+1):
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()
        x = encoder_outputs[all_keys[0]]
        x = self.reshape_output(x) # 1 1024 12 12 12
        x = self.reshaped_conv(x)
        
        d4 = self.Up4(x)
        d4 = torch.cat((x3,d4),dim=1)
        #d4 = torch.maximum(x3, d4)
        d4 = self.Up_conv4(d4)
        del x, x3
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        #d3 = torch.maximum(x2, d3)
        d3 = self.Up_conv3(d3)
        del d4, x2
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        #d2 = torch.maximum(x1, d2)
        d2 = self.Up_conv2(d2)
        del d3, x1
        d1 = self.Conv_1x1(d2)
        d1 = self.act(d1)
        del d2

        return d1

    def reshape_output(self, x):
        x = x.view(
            x.size(0), # 1
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # 1 1024 12 12 12

        return x
        
class TABSurfer_full(nn.Module): # all 34 aseg classes
    def __init__(
        self,
        img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 34,
        embedding_dim = 1024,
        num_heads = 16,
        num_layers = 8,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        ):
        super(TABSurfer_full,self).__init__()

        self.hidden_dim = int((img_dim/16)**3) * 8

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Conv1 = resconv_block_3D_new(ch_in=img_ch,ch_out=32) # 96
        self.Conv2 = resconv_block_3D_new(ch_in=32,ch_out=64) # 48
        self.Conv3 = resconv_block_3D_new(ch_in=64,ch_out=128) # 24
        self.Conv4 = resconv_block_3D_new(ch_in=128,ch_out=256) # 12
        
        self.Up4 = up_conv_3D(ch_in=256,ch_out=128)
        self.Up_conv4 = resconv_block_3D_new(ch_in=256, ch_out=128)
        self.Up3 = up_conv_3D(ch_in=128,ch_out=64)
        self.Up_conv3 = resconv_block_3D_new(ch_in=128, ch_out=64)
        self.Up2 = up_conv_3D(ch_in=64,ch_out=32)
        self.Up_conv2 = resconv_block_3D_new(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv3d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.gn = nn.GroupNorm(8, 256)
        self.relu = nn.ReLU(inplace=True)
        self.act = nn.Softmax(dim=1)

        self.num_patches = int((img_dim // patch_dim) ** 3)

        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, self.hidden_dim
        )

        #self.reshaped_conv = conv_block_3D(512, 128)
        self.reshaped_conv = conv_block_3D(embedding_dim, 256)


        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            self.hidden_dim,

            dropout_rate,
            attn_dropout_rate,
        )

        self.conv_x = nn.Conv3d(
            256,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.embedding_dim = embedding_dim

        self.num_layers = num_layers

    def forward(self,input):
        # encoding path
        x1 = self.Conv1(input)
        del input
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x = self.Conv4(x4)
        del x4
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.position_encoding(x)
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        encoder_outputs = {}
        all_keys = []
        for i in range(1, self.num_layers+1):
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()
        x = encoder_outputs[all_keys[0]]
        x = self.reshape_output(x) # 1 1024 12 12 12
        x = self.reshaped_conv(x)
        
        d4 = self.Up4(x)
        d4 = torch.cat((x3,d4),dim=1)
        #d4 = torch.maximum(x3, d4)
        d4 = self.Up_conv4(d4)
        del x, x3
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        #d3 = torch.maximum(x2, d3)
        d3 = self.Up_conv3(d3)
        del d4, x2
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        #d2 = torch.maximum(x1, d2)
        d2 = self.Up_conv2(d2)
        del d3, x1
        d1 = self.Conv_1x1(d2)
        d1 = self.act(d1)
        del d2

        return d1

    def reshape_output(self, x):
        x = x.view(
            x.size(0), # 1
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # 1 1024 12 12 12

        return x

class TABSurfer_full_big(nn.Module): #larger resconv blocks
    def __init__(
        self,
        img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 34,
        embedding_dim = 1024,
        num_heads = 16,
        num_layers = 8,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        ):
        super(TABSurfer_full_big,self).__init__()

        self.hidden_dim = int((img_dim/16)**3) * 8
        #self.hidden_dim = 2744
        last_size = 256

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Conv1 = resconv_block_3D(ch_in=img_ch,ch_out=32) # 96
        self.Conv2 = resconv_block_3D(ch_in=32,ch_out=64) # 48
        self.Conv3 = resconv_block_3D(ch_in=64,ch_out=128) # 24
        self.Conv4 = resconv_block_3D(ch_in=128,ch_out=last_size) # 12
        
        self.Up4 = up_conv_3D(ch_in=last_size,ch_out=128)
        self.Up_conv4 = resconv_block_3D(ch_in=last_size, ch_out=128)
        self.Up3 = up_conv_3D(ch_in=128,ch_out=64)
        self.Up_conv3 = resconv_block_3D(ch_in=128, ch_out=64)
        self.Up2 = up_conv_3D(ch_in=64,ch_out=32)
        self.Up_conv2 = resconv_block_3D(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv3d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.gn = nn.GroupNorm(8, last_size)
        self.relu = nn.ReLU(inplace=True)
        self.act = nn.Softmax(dim=1)

        self.num_patches = int((img_dim // patch_dim) ** 3)

        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, self.hidden_dim
        )

        #self.reshaped_conv = conv_block_3D(512, 128)
        self.reshaped_conv = conv_block_3D(embedding_dim, last_size)


        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            self.hidden_dim,

            dropout_rate,
            attn_dropout_rate,
        )

        self.conv_x = nn.Conv3d(
            last_size,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.embedding_dim = embedding_dim

        self.num_layers = num_layers

    def forward(self,input):
        # encoding path
        x1 = self.Conv1(input)
        del input
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x = self.Conv4(x4)
        del x4
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.position_encoding(x)
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        encoder_outputs = {}
        all_keys = []
        for i in range(1, self.num_layers+1):
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]

        all_keys.reverse()
        x = encoder_outputs[all_keys[0]]
        x = self.reshape_output(x) # 1 1024 12 12 12
        x = self.reshaped_conv(x)
        
        d4 = self.Up4(x)
        d4 = torch.cat((x3,d4),dim=1)
        #d4 = torch.maximum(x3, d4)
        d4 = self.Up_conv4(d4)
        del x, x3
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        #d3 = torch.maximum(x2, d3)
        d3 = self.Up_conv3(d3)
        del d4, x2
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        #d2 = torch.maximum(x1, d2)
        d2 = self.Up_conv2(d2)
        del d3, x1
        d1 = self.Conv_1x1(d2)
        d1 = self.act(d1)
        del d2

        return d1

    def reshape_output(self, x):
        x = x.view(
            x.size(0), # 1
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # 1 1024 7 7 7

        return x

if __name__ == '__main__':
    
    gpu_id = 1
    torch.cuda.empty_cache()
    memory_before = torch.cuda.mem_get_info(gpu_id)[0] / 1e9
    
    size = 96
    '''model = TABSurfer_full_64(img_dim = size,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 34,
        embedding_dim = 1024,
        num_heads = 16,
        num_layers = 8) # 7.7 gig'''
    model = TABSurfer_full_big(img_dim = size)
    #size = 96
    print(sum(p.numel() for p in model.parameters()), flush = True)
    test = torch.rand([1,1,size,size,size])
    #test = torch.rand([1])
    
    test = test.cuda(gpu_id)
    model = model.cuda(gpu_id)
    model.eval()
    with torch.no_grad():
        out = model(test)
    
    #with torch.no_grad():
    #    out = model(test)
    print(out.shape)
    memory_after = torch.cuda.mem_get_info(gpu_id)[0] / 1e9
    print(memory_before, memory_after)
    print(f'{memory_before - memory_after} gigs used')
    