import torchvision.transforms as T


class CustomAugmentationTransform:
    def __init__(
        self,
        apply_augmentation=True,
        rotation_degrees=30,
        stretch_scale_range=(0.8, 1.2),
        stretch_shear_degrees=(-10, 10),
    ):
        """
        Custom transform for image augmentation including rotation and stretching.
        "Stretching" is achieved via scaling and shearing using RandomAffine.

        Args:
            apply_augmentation (bool): Whether to apply augmentations.
            rotation_degrees (float or sequence): Degrees for random rotation.
                                                 Example: 30 or (-30, 30).
            stretch_scale_range (tuple): Min and max scale for random affine transformation.
                                         Example: (0.8, 1.2).
            stretch_shear_degrees (float or sequence): Range for random shear.
                                                     If float, shear is (-degrees, degrees) for x-axis.
                                                     If sequence of 2, shear_x is (min, max).
                                                     If sequence of 4, shear_x (min, max), shear_y (min, max).
                                                     Example: 10 or (-10, 10) or (-10, 10, -5, 5).
        """
        self.apply_augmentation = apply_augmentation

        if self.apply_augmentation:
            self.transform = T.Compose(
                [
                    T.RandomRotation(degrees=rotation_degrees),
                    T.RandomAffine(
                        degrees=0,
                        scale=stretch_scale_range,
                        shear=stretch_shear_degrees,
                    ),
                ]
            )
        else:
            self.transform = None

    def __call__(self, image_tensor):
        """
        Applies the augmentation to the image tensor.
        Args:
            image_tensor (torch.Tensor): A C x H x W tensor (e.g., 4 x H x W).
        Returns:
            torch.Tensor: The augmented image tensor.
        """
        if not self.apply_augmentation or not self.transform:
            return image_tensor
        return self.transform(image_tensor)
