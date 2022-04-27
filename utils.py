import numpy as np
import SimpleITK as sitk

# helper functions copy pasted
def resample_by_res(mov_img_obj, new_spacing, interpolator = sitk.sitkLinear, logging = True):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_img_obj.GetDirection())
    resample.SetOutputOrigin(mov_img_obj.GetOrigin())
    mov_spacing = mov_img_obj.GetSpacing()

    resample.SetOutputSpacing(new_spacing)
    RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
    new_size = np.array(mov_img_obj.GetSize()) *  RES_COE 

    resample.SetSize( [int(sz+1) for sz in new_size] )
    if logging:
        print("Spacing: {} -> {}".format(mov_spacing, new_spacing))
        print("Size {} -> {}".format( mov_img_obj.GetSize(), new_size ))

    return resample.Execute(mov_img_obj)

def resample_lb_by_res(mov_lb_obj, new_spacing, interpolator = sitk.sitkLinear, ref_img = None, logging = True):
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)
    lbvs = np.unique(src_mat)
    if logging:
        print("Label values: {}".format(lbvs))
    for idx, lbv in enumerate(lbvs):
        _src_curr_mat = np.float32(src_mat == lbv) 
        _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
        _src_curr_obj.CopyInformation(mov_lb_obj)
        _tar_curr_obj = resample_by_res( _src_curr_obj, new_spacing, interpolator, logging )
        _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv
        if idx == 0:
            out_vol = _tar_curr_mat
        else:
            out_vol[_tar_curr_mat == lbv] = lbv
    out_obj = sitk.GetImageFromArray(out_vol)
    out_obj.SetSpacing( _tar_curr_obj.GetSpacing() )
    if ref_img != None:
        out_obj.CopyInformation(ref_img)
    return out_obj
        
## Then crop ROI
def get_label_center(label):
    nnz = np.sum(label > 1e-5)
    return np.int32(np.rint(np.sum(np.nonzero(label), axis = 1) * 1.0 / nnz))

def image_crop(ori_vol, crop_size, referece_ctr_idx, padval = 0., only_2d = True):
    """ crop a 3d matrix given the index of the new volume on the original volume
        Args:
            refernce_ctr_idx: the center of the new volume on the original volume (in indices)
            only_2d: only do cropping on first two dimensions
    """
    _expand_cropsize = [x + 1 for x in crop_size] # to deal with boundary case
    if only_2d:
        assert len(crop_size) == 2, "Actual len {}".format(len(crop_size))
        assert len(referece_ctr_idx) == 2, "Actual len {}".format(len(referece_ctr_idx))
        _expand_cropsize.append(ori_vol.shape[-1])
        
    image_patch = np.ones(tuple(_expand_cropsize)) * padval

    half_size = tuple( [int(x * 1.0 / 2) for x in _expand_cropsize] )
    _min_idx = [0,0,0]
    _max_idx = list(ori_vol.shape)

    # bias of actual cropped size to the beginning and the end of this volume
    _bias_start = [0,0,0]
    _bias_end = [0,0,0]

    for dim,hsize in enumerate(half_size):
        if dim == 2 and only_2d:
            break

        _bias_start[dim] = np.min([hsize, referece_ctr_idx[dim]])
        _bias_end[dim] = np.min([hsize, ori_vol.shape[dim] - referece_ctr_idx[dim]])

        _min_idx[dim] = referece_ctr_idx[dim] - _bias_start[dim]
        _max_idx[dim] = referece_ctr_idx[dim] + _bias_end[dim]
        
    if only_2d:
        image_patch[ half_size[0] - _bias_start[0]: half_size[0] +_bias_end[0], \
                half_size[1] - _bias_start[1]: half_size[1] +_bias_end[1], ... ] = \
                ori_vol[ referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] +_bias_end[0], \
                referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] +_bias_end[1], ... ]

        image_patch = image_patch[ 0: crop_size[0], 0: crop_size[1], : ]
    # then goes back to original volume
    else:
        image_patch[ half_size[0] - _bias_start[0]: half_size[0] +_bias_end[0], \
                half_size[1] - _bias_start[1]: half_size[1] +_bias_end[1], \
                half_size[2] - _bias_start[2]: half_size[2] +_bias_end[2] ] = \
                ori_vol[ referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] +_bias_end[0], \
                referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] +_bias_end[1], \
                referece_ctr_idx[2] - _bias_start[2]: referece_ctr_idx[2] +_bias_end[2] ]

        image_patch = image_patch[ 0: crop_size[0], 0: crop_size[1], 0: crop_size[2] ]
    return image_patch

   
    
def copy_spacing_ori(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst


def resize_image(image, img_size=(64, 64, 64), **kwargs):
    assert isinstance(image, (np.ndarray, np.generic))
    assert image.ndim - 1 == len(img_size) or image.ndim == len(
        img_size
    ), "Example size doesnt fit image size"

    rank = len(img_size)

    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.0))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    return np.pad(image[slicer], to_padding, **kwargs)