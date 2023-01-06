import torch

batchsize = 10
dim = 32
patch_nums = 9
patch_w_n = 3
patch_h_n = 3
patch_w = 2
patch_h = 2
pix_nums = 4
target_w = 6
target_h = 6

x = torch.range(1,batchsize*dim*target_w*target_w)
x = x.reshape(batchsize, dim, target_h, target_w)
x = torch.tensor(x,dtype=torch.int)


def unfold(x):
    x = x.reshape(batchsize, dim, patch_h_n, patch_h, patch_w_n, patch_w)
    x = x.transpose(3,4)
    x = x.reshape(batchsize, dim, patch_h_n**2, patch_h**2)  #(10,32,9,4)
    x = x.transpose(1,3)
    x = x.reshape(batchsize*pix_nums,patch_nums,-1)
    return x

def my_self(x: torch.Tensor): #(8,32,32,32)
    # [B, C, H, W] -> [B, C, n_h, p_h, n_w, p_w]
    x = x.reshape(10, 32, 3, 2, 3, 2) #(8,32,16,2,16,2)
    # [B, C, n_h, p_h, n_w, p_w] -> [B, C, n_h, n_w, p_h, p_w]
    x = x.transpose(3, 4) #(8,32,16,16,2,2)
    # [B, C, n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    """
    P:一个patch里面一共有多少个像素 (4)
    N:一共有多少个patch (256)
    """
    x = x.reshape(10, 32, 9, 4) #(8,32,256,4)
    # [B, C, N, P] -> [B, P, N, C]
    x = x.transpose(1, 3)  #(8,4,256,32)
    # [B, P, N, C] -> [BP, N, C]
    x = x.reshape(10 * 4, 9, -1)  #(32,256,32)
    return x

y = (unfold(x)==my_self(x))