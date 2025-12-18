import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any


def find_nearest_value(value, arr: np.ndarray, return_idx: bool = True):
    idx = np.argmin(np.abs(arr - value))
    if return_idx:
        return idx
    return arr[idx]


def find_nearest_equal_or_smaller_value(value,
                                        sorted_arr: np.ndarray,
                                        return_idx: bool = True):
    index = np.searchsorted(sorted_arr, value)
    index = index if index < len(sorted_arr) else len(sorted_arr) - 1
    if return_idx:
        return index
    return sorted_arr[index]


def get_pixel_range_prob_dict(pixel_num_list: list, pixel_prob_list: list):
    if len(pixel_num_list) == len(pixel_prob_list):
        pixel_num_list.append(pixel_num_list[-1] * 4)
    assert len(pixel_prob_list) == len(pixel_num_list) - 1
    pixel_prob_list = np.array(pixel_prob_list)
    pixel_prob_list = pixel_prob_list / pixel_prob_list.sum()
    prob_dict = OrderedDict({})
    for i in range(len(pixel_prob_list)):
        pixel_range = (pixel_num_list[i], pixel_num_list[i + 1])
        prob_dict[pixel_range] = pixel_prob_list[i]
    return prob_dict


def get_bucket_config_from_side_list_and_ar_list(
    side_list: tuple,
    ar_value_arr: list,
    ar_max: float = 2.0,
    ar_min: float = 0.5,
    pixel_max: int = 1536**2,
):
    if isinstance(ar_value_arr, list):
        ar_value_arr = np.array(ar_value_arr)
    ar_list = []
    ar_dict = {}

    for h in side_list:
        assert h % 16 == 0
        for w in side_list:
            if h * w > pixel_max:
                continue
            ar = f'{h / w : .3f}'
            ar_list.append(ar)
            if ar not in ar_dict:
                ar_dict[ar] = []
            ar_dict[ar].append((h, w, h * w))

    ar_list = [
        ar for ar in ar_list if float(ar) <= ar_max and float(ar) >= ar_min
    ]
    ar_dict = {k: v for k, v in ar_dict.items() if k in ar_list}

    merged_ar_dict = {ar_value: [] for ar_value in ar_value_arr}
    for ar, size_list in ar_dict.items():
        ar_value = find_nearest_value(
            float(ar), ar_value_arr, return_idx=False)
        merged_ar_dict[ar_value] += size_list

    for ar, size_list in merged_ar_dict.items():
        merged_ar_dict[ar] = sorted(size_list, key=lambda x: x[2])
    return merged_ar_dict


def get_side_list_64_base():
    return [64 * i for i in range(4, 33)]


def get_default_side_list():
    return (144, 240, 256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1072,
            1280, 1440, 1920)


def get_default_ar_list():
    ar_list = [1, 1.5, 4 / 3, 16 / 9, 2]
    ar_list = ar_list + [1 / ar for ar in ar_list[1:]]
    return sorted(ar_list)


def assign_probability(bucket_config, prob_dict):
    for ar, size_list in bucket_config.items():
        size_list = sorted(size_list, key=lambda x: x[2])
        size_list_with_prob = []
        for pixel_range, prob in prob_dict.items():
            in_range_size_list = [
                size for size in size_list
                if size[0] * size[1] >= pixel_range[0] and size[0] *
                size[1] < pixel_range[1]
            ]
            if len(in_range_size_list) == 0:
                continue
            in_range_size_list = [(*size, prob) for size in in_range_size_list]
            size_list_with_prob = size_list_with_prob + in_range_size_list
        bucket_config[ar] = (np.array(size_list_with_prob)[:, :2].astype(
            np.int64), np.array(size_list_with_prob)[:, -1])
        bucket_config[ar][
            1][:] = bucket_config[ar][1][:] / bucket_config[ar][1][:].sum()
    return bucket_config


class FlopEstimator(ABC):

    @abstractmethod
    def __init__(self, *size) -> None:
        pass

    @abstractmethod
    def __eq__(self, other) -> None:
        pass

    @abstractmethod
    def __lt__(self, other) -> None:
        pass

    @abstractmethod
    def __gt__(self, other) -> None:
        pass


class FlopEstimatorExact(FlopEstimator):
    common_factor = 16  # this value should be at least 16
    max_thw = 16934400  # 720 * 480 * 49
    max_t = 49
    t_tick_size = 8

    def __init__(self, t, h, w) -> None:
        self.size = self.net_size(t, h, w)
        self.thw = np.prod(self.size)

    def net_size(self, t, h, w):
        h = h // self.common_factor * self.common_factor
        w = w // self.common_factor * self.common_factor
        t = min(t, int(self.max_thw / (h * w)), self.max_t)
        t = (t + self.t_tick_size -
             1) // self.t_tick_size * self.t_tick_size - self.t_tick_size + 1
        t = max(t, 1)
        return t, h, w

    def __eq__(self, other) -> None:
        return self.size == other.size

    def __lt__(self, other) -> None:
        return self.thw < other.thw or self.size < other.size

    def __gt__(self, other) -> None:
        return self.thw > other.thw or self.size > other.size

    def __hash__(self, ):
        return hash(self.size)

    def __str__(self) -> str:
        return str(self.size)


class FlopEstimator2D1DNotExact(FlopEstimatorExact):
    common_factor = 16  # this value should be at least 16
    max_thw = 16934400  # 720 * 480 * 49
    max_t = 49
    max_hw = 1920 * 1920

    def __init__(self, t, h, w, hw_value_arr: np.ndarray,
                 t_arr: np.ndarray) -> None:
        self.size = self.net_size(t, h, w)
        t, h, w = self.size
        hw_idx = find_nearest_equal_or_smaller_value(
            h * w, hw_value_arr, return_idx=True)
        t_idx = find_nearest_equal_or_smaller_value(t, t_arr, return_idx=True)
        self.size_not_exact = (t_idx, hw_idx)
        self.thw = np.prod(self.size)

    def __eq__(self, other) -> None:
        return self.size_not_exact == other.size_not_exact

    def __lt__(self, other) -> None:
        return self.size_not_exact != other.size_not_exact and self.thw < other.thw

    def __gt__(self, other) -> None:
        return self.size_not_exact != other.size_not_exact and self.thw > other.thw

    def __hash__(self, ):
        return hash(self.size_not_exact)

    def __str__(self) -> str:
        return str(self.size)


# class FlopEstimator3DNotExact(FlopEstimator2D1DNotExact):
#     common_factor = 16  # this value should be at least 16
#     max_thw = 16934400  # 720 * 480 * 49
#     max_t = 49

#     def __init__(self, t, h, w, thw_value_arr: np.ndarray) -> None:
#         self.size = self.net_size(t, h, w)
#         t, h, w = self.size
#         thw_idx = find_nearest_equal_or_smaller_value(
#             h * w, thw_value_arr, return_idx=True)
#         self.size_not_exact = thw_idx


class BucketConfig:
    bucket_config = OrderedDict({})
    flop = FlopEstimatorExact

    @classmethod
    def from_class_name(cls, class_name, class_kwargs: dict):
        bucket_class = globals().get(class_name)
        if not (isinstance(bucket_class, BucketConfig)
                or issubclass(bucket_class, BucketConfig)):
            raise ValueError(f'wrong class name {class_name}')
        return bucket_class(**class_kwargs)

    def __init__(self, ) -> None:
        ar_value_arr = np.array(list(self.bucket_config.keys()))
        assert np.all(ar_value_arr[:-1] < ar_value_arr[1:])
        self.ar_value_arr = ar_value_arr
        for ar, (hwp, prob) in self.bucket_config.items():
            assert isinstance(hwp, np.ndarray)
            assert isinstance(prob, np.ndarray)
            if hwp.shape[1] == 2:
                hwp = np.stack([hwp[:, 0], hwp[:, 1], hwp[:, 0] * hwp[:, 1]],
                               axis=-1)
            else:
                assert np.all(hwp[:-1, 2] <= hwp[1:, 2])
            if len(hwp) != len(prob):
                raise ValueError(
                    f'wrong config of aspect ratio: {ar}, size_list: {hwp}, prob_list: {prob}'
                )
            prob[:] = prob[:] / prob.sum()
            self.bucket_config[ar] = (hwp, prob)
            print(f'aspect ratio: {ar}, size list: {self.bucket_config[ar]}')
        pass

    def preprocess(self, n_frame: int, height: int, width: int,
                   rnd_state: np.random.RandomState):
        """process original height width, sample a (height, width) in
        bucket_config with the closed aspect ratio."""
        ar = height / width
        tgt_ar_idx = find_nearest_value(ar, self.ar_value_arr, return_idx=True)
        h_w_pixel_list, prob_list = self.bucket_config[
            self.ar_value_arr[tgt_ar_idx]]
        if h_w_pixel_list[-1, -1] > height * width:
            down_num = np.sum(h_w_pixel_list[:, -1] <= height * width)
            h_w_pixel_list = h_w_pixel_list[:down_num]
            prob_list = prob_list[:down_num] / prob_list[:down_num].sum()
        selected_idx = rnd_state.choice(
            len(prob_list), size=None, replace=False, p=prob_list)
        h, w = h_w_pixel_list[selected_idx][:2]
        return h, w

    def __call__(self, n_frame: int, height: int, width: int,
                 rnd_state: np.random.RandomState) -> Any:
        tgt_h, tgt_w = self.preprocess(n_frame, height, width, rnd_state)
        return self.flop(n_frame, tgt_h, tgt_w)


class BucketConfigHardCoded1(BucketConfig):

    def __init__(self, ) -> None:
        self.bucket_config = {
            # aspect_ratio: (height width pixel value with shape of (n, 3), probability with shape of (n, ) )
            0.667: (np.array([[480, 720, 480 * 720]],
                             dtype=np.int64), np.array([1.0])),
            1.0: (np.array([[384, 384, 384**2], [512, 512, 512**2]],
                           dtype=np.int64), np.array([0.4, 0.6])),
            1.5: (np.array([[720, 480, 480 * 720]],
                           dtype=np.int64), np.array([1.0])),
        }
        super().__init__()
        pass


class BucketConfigHardCoded2(BucketConfig):

    def __init__(self, ) -> None:
        self.bucket_config = {
            # aspect_ratio: (height width pixel value with shape of (n, 3), probability with shape of (n, ) )
            0.667: (np.array([[240, 360], [360, 540], [480, 720]],
                             dtype=np.int64), np.array([0.2, 0.5, 0.3])),
            1.0: (np.array([[256, 256], [384, 384], [512, 512]],
                           dtype=np.int64), np.array([0.2, 0.5, 0.3])),
            1.5: (np.array([[240, 360], [360, 540], [480, 720]],
                           dtype=np.int64)[:, ::-1], np.array([0.2, 0.5,
                                                               0.3])),
        }
        super().__init__()
        pass


class BucketConfigHardCoded3(BucketConfig):

    def __init__(self, ) -> None:
        self.bucket_config = {
            # aspect_ratio: (height width pixel value with shape of (n, 3), probability with shape of (n, ) )
            0.667: (np.array([[240, 360], [360, 540], [480, 720], [720, 1080]],
                             dtype=np.int64), np.array([
                                 0.2,
                                 0.5,
                                 0.2,
                                 0.1,
                             ])),
            1.0: (np.array([[256, 256], [384, 384], [512, 512], [768, 768]],
                           dtype=np.int64), np.array([0.2, 0.5, 0.2, 0.1])),
            1.5:
            (np.array([[240, 360], [360, 540], [480, 720], [720, 1080]],
                      dtype=np.int64)[:, ::-1], np.array([0.2, 0.5, 0.2,
                                                          0.1])),
        }
        super().__init__()
        pass


class BucketConfigFromSideList(BucketConfig):

    def __init__(self, side_list: list, ar_list: list, ar_max: float,
                 ar_min: float, pixel_num_list: list,
                 pixel_prob_list: list) -> None:
        assert len(pixel_num_list) == len(pixel_prob_list) or len(
            pixel_num_list) == (
                len(pixel_prob_list) + 1)
        prob_dict = get_pixel_range_prob_dict(pixel_num_list, pixel_prob_list)
        bucket_config = get_bucket_config_from_side_list_and_ar_list(
            side_list, ar_list, ar_max=ar_max, ar_min=ar_min)
        self.bucket_config = assign_probability(bucket_config, prob_dict)
        super().__init__()
        pass


class DefaultBucketConfig(BucketConfigFromSideList):

    def __init__(self, ) -> None:
        side_list = get_default_side_list()
        ar_list = get_default_ar_list()
        ar_max = 2.0
        ar_min = 0.5
        pixel_num_list = [
            144**2, 256**2, 384**2, 512**2, 480 * 720, 768**2, 4096**2
        ]
        pixel_prob_list = [0.05, 0.4, 0.4, 0.1, 0.03, 0.02]
        self.side_list = side_list
        super().__init__(
            side_list=side_list,
            ar_list=ar_list,
            ar_max=ar_max,
            ar_min=ar_min,
            pixel_num_list=pixel_num_list,
            pixel_prob_list=pixel_prob_list,
        )


class DefaultBucketConfigNotExact(DefaultBucketConfig):
    flop = FlopEstimator2D1DNotExact

    def __init__(self,
                 pixel_grading_num: int = 100,
                 frame_grading_num: int = 13) -> None:
        super().__init__()
        self.pixel_grading_arr = np.linspace(
            self.side_list[0]**2,
            self.side_list[-1]**2,
            pixel_grading_num,
            dtype=np.int64)
        self.frame_grading_arr = np.arange(frame_grading_num) * 4 + 1

    def __call__(self, n_frame: int, height: int, width: int,
                 rnd_state: np.random.RandomState) -> Any:
        tgt_h, tgt_w = self.preprocess(n_frame, height, width, rnd_state)
        flop = FlopEstimator2D1DNotExact(n_frame, tgt_h, tgt_w,
                                         self.pixel_grading_arr,
                                         self.frame_grading_arr)
        return flop


class BucketConfig3AR(BucketConfigFromSideList):

    def __init__(self, ) -> None:
        side_list = get_default_side_list()
        ar_list = sorted([0.6667, 1.0, 1.5])
        ar_max = 2.0
        ar_min = 0.5
        pixel_num_list = [
            144**2, 256**2, 384**2, 512**2, 480 * 720, 768**2, 4096**2
        ]
        pixel_prob_list = [0.05, 0.4, 0.4, 0.1, 0.03, 0.02]
        self.side_list = side_list
        super().__init__(
            side_list=side_list,
            ar_list=ar_list,
            ar_max=ar_max,
            ar_min=ar_min,
            pixel_num_list=pixel_num_list,
            pixel_prob_list=pixel_prob_list,
        )


class DefaultBucketConfig3ARNotExact(BucketConfig3AR):
    flop = FlopEstimator2D1DNotExact

    def __init__(self,
                 pixel_grading_num: int = 100,
                 frame_grading_num: int = 13) -> None:
        super().__init__()
        self.pixel_grading_arr = np.linspace(
            self.side_list[0]**2,
            self.side_list[-1]**2,
            pixel_grading_num,
            dtype=np.int64)
        self.frame_grading_arr = np.arange(frame_grading_num) * 4 + 1

    def __call__(self, n_frame: int, height: int, width: int,
                 rnd_state: np.random.RandomState) -> Any:
        tgt_h, tgt_w = self.preprocess(n_frame, height, width, rnd_state)
        flop = FlopEstimator2D1DNotExact(n_frame, tgt_h, tgt_w,
                                         self.pixel_grading_arr,
                                         self.frame_grading_arr)
        return flop