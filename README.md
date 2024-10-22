# extract shape from sampled input
#inshape = next(generator)[0][0].shape[1:-1]
inshape=(256,256,64)  # Could change it later
flow_in=torch.zeros(1, 3,256, 256, 64, device=device)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]