from tensorpack.dataflow import *
import cv2
from data_parsing.inaturalist.inaturalist_dataflow import iNaturalist

mean_inat = [0.454, 0.474, 0.367]
std_inat = [0.237, 0.230, 0.249]


def get_inat_data(data_partition='train', batch_size=1):
    print("Getting inaturalist data for", data_partition)
    ds = iNaturalist('/mnt/mount_sda2/src/inaturalist/', data_partition, shuffle=True)
    # ds.reset_state()
    if data_partition == 'train':
        # augmentations = [imgaug.Resize(224), imgaug.Flip(horiz=True), imgaug.MapImage(lambda x: (x - mean_inat) / std_inat)]
        # augmentations = [imgaug.RandomCrop(224), imgaug.Flip(horiz=True), imgaug.MapImage(lambda x: (x - mean_inat) / std_inat)]
        augmentations = [imgaug.GoogleNetRandomCropAndResize(target_shape=224), imgaug.Flip(horiz=True), imgaug.MapImage(lambda x: (x - mean_inat) / std_inat)]
        # augmentations = [imgaug.Resize(224), imgaug.Flip(horiz=True)]
    elif data_partition == 'val':
        augmentations = [imgaug.Resize(224), imgaug.MapImage(lambda x: (x - mean_inat) / std_inat)]
    ds = AugmentImageComponent(ds, augmentations)
    # # TODO: for validation, don't parallelize like this.
    if data_partition != 'val':
        ds = MultiProcessRunnerZMQ(ds, num_proc=16)  # 32 may be possible
    ds = BatchData(ds, batch_size)

    # augmentor = imgaug.AugmentorList(augmentations)
    # ds = MultiThreadMapData(ds,
    #                         num_thread=8,
    #                         map_func=lambda dp: [augmentor.augment(dp[0])] + dp[1:], buffer_size=200)
    # if data_partition != 'val':
    #     ds = MultiProcessRunnerZMQ(ds, num_proc=16)
    # ds = BatchData(ds, batch_size)

    return ds


if __name__ == '__main__':
    df = get_inat_data()
    # df.reset_state()
    # for ds in df:
    #     imgs = ds[0]
    #     print("Going")
    #     for img in imgs:
    #         print("img", img)
    #         cv2.imshow('image', img)
    #         cv2.waitKey(1000)
    #         cv2.destroyAllWindows()
    TestDataSpeed(df, size=1000).start()
    print("Truly goodbye")
