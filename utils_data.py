from monai import transforms

def get_val_transforms_base(args):
    if args.affine:
        rotate = args.rotate * 3.141592 / 180
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
                transforms.AddChanneld(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
                transforms.Orientationd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                        axcodes="RAS"),
    
                transforms.Spacingd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest","bilinear", "nearest")),
    
                transforms.Affined(keys=["pct","pct_msk"],
                                           rotate_params=[rotate,rotate,rotate],
                                           translate_params=[args.trans,args.trans,args.trans],
                                           mode = ("bilinear", "nearest"),
                                           padding_mode  = ("border","border")
                                        ),
    
                transforms.ScaleIntensityRanged(keys=["pct","cbct"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                
                transforms.CropForegroundd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                           source_key="pct"),
                
                transforms.SpatialPadd(keys=["pct", "pct_msk", "cbct", "cbct_msk"], 
                                       spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
    
                #Here center crop the image with 256 256 64 
                transforms.RandSpatialCropSamplesd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                                   roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                                   random_size=False,
                                                   num_samples=1,
                                                   random_center=False),
    
                transforms.ToTensord(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
            ]
        )
    
    else:
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
                transforms.AddChanneld(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
                transforms.Orientationd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                        axcodes="RAS"),
        
                transforms.Spacingd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest","bilinear", "nearest")),
        
                transforms.ScaleIntensityRanged(keys=["pct","cbct"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
        
                transforms.CropForegroundd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                           source_key="pct"),
        
                transforms.SpatialPadd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                       spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
        
                transforms.RandSpatialCropSamplesd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                                   roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                                   random_size=False,
                                                   num_samples=1,
                                                   random_center=False),
        
                transforms.ToTensord(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
            ]
        )
    
    return val_transform

def get_val_transforms_affine(args):
    rotate = args.rotate * 3.141592 / 180
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
            transforms.AddChanneld(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
            transforms.Orientationd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                    axcodes="RAS"),

            transforms.Spacingd(keys=["pct", "pct_msk", "cbct", "cbct_msk"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest","bilinear", "nearest")),
           
            transforms.ScaleIntensityRanged(keys=["pct","cbct"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            
             transforms.Affined(keys=["pct","pct_msk"],
                                           rotate_params=(0,0,rotate),
                                           translate_params=(0,0,args.trans),
                                           mode = ("bilinear", "nearest"),
                                           padding_mode  = 'zeros'
                                        ),
             
            transforms.SpatialPadd(keys=["pct", "pct_msk", "cbct", "cbct_msk"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ToTensord(keys=["pct", "pct_msk", "cbct", "cbct_msk"]),
        ]
    )

    
    return val_transform


def copy_info(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    # dst.CopyInfomation(src)
    return dst