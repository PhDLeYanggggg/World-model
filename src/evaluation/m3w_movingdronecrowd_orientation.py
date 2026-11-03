"""Source-audit axis canonicalization, not metric camera-motion correction."""
import numpy as np


def canonical_rgb_boxes(rgb,boxes):
    image,rectangles=np.asarray(rgb),np.asarray(boxes,float)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or rectangles.shape!=(len(rectangles),4):
        raise ValueError('RGB source image and aligned head rectangles required')
    height,width=image.shape[:2]
    transform=np.eye(3)
    if width>=height:
        return image,rectangles.copy(),transform,0
    # Pixel-center convention of a lossless 90-degree counterclockwise rotation.
    transform=np.array([[0.,1.,0.],[-1.,0.,width-1.],[0.,0.,1.]])
    result=np.column_stack((rectangles[:,1],width-1-rectangles[:,2],
                            rectangles[:,3],width-1-rectangles[:,0]))
    return np.rot90(image,k=1).copy(),result,transform,90
