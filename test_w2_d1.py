from vit import positional_encoding
pe = positional_encoding(max_len=50, d_model=64)
print(pe.shape)     # 期望 torch.Size([50, 64])
print(pe[0, :4])    # 期望接近 tensor([0., 1., 0., 1.])

